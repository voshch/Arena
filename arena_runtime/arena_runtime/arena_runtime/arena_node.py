from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import typing

import ament_index_python.packages
import arena_runtime_msgs.msg
import arena_runtime_msgs.srv
import rclpy
import rclpy.lifecycle
import rclpy.qos
import rclpy.subscription
import rclpy.timer
import rosgraph_msgs.msg
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.Time import Time
from lifecycle_msgs.msg import State as LcState
from lifecycle_msgs.msg import Transition, TransitionEvent
from rcl_interfaces.msg import ParameterDescriptor
from rclpy import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from arena_runtime.constants import SimSimulator
from arena_runtime.holds import HoldRegistry
from arena_runtime.lockstep import ChannelSpec, LockstepConfig, LockstepScheduler, _parse_channel_spec
from arena_runtime.registry import EnvRegistry, _extent_eq, sweep_verdict
from arena_runtime.sim import LifecycleRegistry, SimLifecycle
from arena_runtime.sim.dummy_simulator import DummyHost

_LATCHED = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)


def _tee_subprocess_output(
    pipe: typing.IO[bytes],
    log_file: typing.IO[bytes],
    echo_prefix: bytes | None,
) -> None:
    """Drain a subprocess pipe to a log file, optionally echoing each line to parent stdout."""
    try:
        for line in iter(pipe.readline, b""):
            log_file.write(line)
            log_file.flush()
            if echo_prefix is not None:
                sys.stdout.buffer.write(echo_prefix + line)
                sys.stdout.buffer.flush()
    finally:
        log_file.close()


@dataclasses.dataclass
class ManagedEnv:
    """Per-env state owned by ArenaNode: subscriptions, optional spawn future, optional wrapper Popen."""

    env_id: int
    namespace: str
    transition_sub: rclpy.subscription.Subscription
    heartbeat_sub: rclpy.subscription.Subscription
    spawn_ready: asyncio.Future[None] | None = None
    popen: subprocess.Popen | None = None
    log_path: pathlib.Path | None = None
    tee_thread: threading.Thread | None = None

    async def dispose(self, node: ArenaNode, *, grace_seconds: float) -> None:
        """Drop subs, cancel pending spawn future, reap the wrapper pgid."""
        for sub in (self.transition_sub, self.heartbeat_sub):
            with contextlib.suppress(Exception):
                node.destroy_subscription(sub)
        if self.spawn_ready is not None and not self.spawn_ready.done():
            self.spawn_ready.cancel()
        if self.popen is not None:
            await node._shutdown_pg(self.env_id, self.popen, grace_seconds=grace_seconds)
        if self.tee_thread is not None:
            await asyncio.get_running_loop().run_in_executor(None, self.tee_thread.join, 5.0)


_WINDOW_REASON = "unpause_window"


class ArenaNode(ArenaMixinNode, rclpy.lifecycle.LifecycleNode):
    _lifecycle: SimLifecycle
    _holds: HoldRegistry
    _env_registry: EnvRegistry
    _envs: dict[int, ManagedEnv]
    _heartbeat_timer: rclpy.timer.Timer | None
    _clock_task: asyncio.Task | None
    _windows: HoldRegistry
    _lockstep: LockstepScheduler

    def __init__(self) -> None:
        super().__init__("arena")
        self._envs = {}
        self._heartbeat_timer = None
        self._clock_task = None
        self._windows = HoldRegistry()

    def on_configure(self, state: rclpy.lifecycle.State) -> rclpy.lifecycle.TransitionCallbackReturn:
        return rclpy.lifecycle.TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: rclpy.lifecycle.State) -> rclpy.lifecycle.TransitionCallbackReturn:
        super().on_activate(state)
        return rclpy.lifecycle.TransitionCallbackReturn.SUCCESS

    async def setup(self) -> None:
        sim_name = self.rosparam[str].get("sim", SimSimulator.DUMMY.value)
        sim_key = SimSimulator(sim_name)
        physics_dt = self.rosparam[float].get("physics_dt", 0.0333)

        self._lifecycle = await LifecycleRegistry.get(sim_key, node=self, physics_dt=physics_dt)
        self._holds = HoldRegistry()
        self._lockstep = LockstepScheduler(self)

        if isinstance(self._lifecycle, DummyHost):
            self._clock_publisher = self.create_publisher(rosgraph_msgs.msg.Clock, '/clock', 10)
            self._clock_task = asyncio.create_task(self._publish_clock_loop(self._lifecycle))

        period = self.rosparam[float].get("heartbeat_period_sec", 1.0)
        self.rosparam[float].get("heartbeat_timeout_sec", 5.0)
        self.rosparam[float].get("reset_hold_timeout_sec", 30.0)
        self.rosparam[float].get("bootstrap_timeout_sec", 600.0)
        slot_buffer = self.rosparam[float].get("slot_buffer", 5.0)

        self._env_registry = EnvRegistry(slot_buffer=slot_buffer)

        self._pub_paused = self.create_publisher(
            Bool,
            self.service_namespace("state", "paused"),
            _LATCHED,
        )
        self._pub_holders = self.create_publisher(
            arena_runtime_msgs.msg.HoldRegistry,
            self.service_namespace("state", "holders"),
            _LATCHED,
        )
        self._pub_envs = self.create_publisher(
            arena_runtime_msgs.msg.EnvRegistry,
            self.service_namespace("state", "envs"),
            _LATCHED,
        )
        self._pub_shutdown_request = self.create_publisher(
            arena_runtime_msgs.msg.ShutdownRequest,
            self.service_namespace("shutdown_request"),
            10,
        )

        self.get_logger().info(f"waiting for {sim_name} services to come up")
        await self._lifecycle.ensure_ready()
        self.get_logger().info(f"{sim_name} services are up")

        register_env_name = str(self.service_namespace("register_env"))
        if any(name == register_env_name for name, _ in self.get_service_names_and_types()):
            raise RuntimeError(f"another arena runtime is already running ({register_env_name} exists); shut it down first")

        srv_cb_group = ReentrantCallbackGroup()

        self._srv_hold = self.create_service(
            arena_runtime_msgs.srv.LifecycleHold,
            self.service_namespace("sim_lifecycle", "hold"),
            self._cb_hold,
            callback_group=srv_cb_group,
        )
        self._srv_unpause_window = self.create_service(
            arena_runtime_msgs.srv.LifecycleUnpauseWindow,
            self.service_namespace("sim_lifecycle", "unpause_window"),
            self._cb_unpause_window,
            callback_group=srv_cb_group,
        )
        self._srv_step = self.create_service(
            arena_runtime_msgs.srv.LifecycleStep,
            self.service_namespace("sim_lifecycle", "step"),
            self._cb_step,
            callback_group=srv_cb_group,
        )
        self._srv_lockstep_start = self.create_service(
            arena_runtime_msgs.srv.LockstepStart,
            self.service_namespace("sim_lifecycle", "lockstep", "start"),
            self._cb_lockstep_start,
            callback_group=srv_cb_group,
        )
        self._srv_lockstep_register = self.create_service(
            arena_runtime_msgs.srv.LockstepRegister,
            self.service_namespace("sim_lifecycle", "lockstep", "register"),
            self._cb_lockstep_register,
            callback_group=srv_cb_group,
        )
        self._srv_lockstep_stop = self.create_service(
            Trigger,
            self.service_namespace("sim_lifecycle", "lockstep", "stop"),
            self._cb_lockstep_stop,
            callback_group=srv_cb_group,
        )
        self._srv_lockstep_pause = self.create_service(
            Trigger,
            self.service_namespace("sim_lifecycle", "lockstep", "pause"),
            self._cb_lockstep_pause,
            callback_group=srv_cb_group,
        )
        self._srv_lockstep_resume = self.create_service(
            Trigger,
            self.service_namespace("sim_lifecycle", "lockstep", "resume"),
            self._cb_lockstep_resume,
            callback_group=srv_cb_group,
        )
        self._srv_cleanup = self.create_service(
            arena_runtime_msgs.srv.CleanupEnv,
            self.service_namespace("cleanup_env"),
            self._cb_cleanup_env,
            callback_group=srv_cb_group,
        )
        self._srv_register_env = self.create_service(
            arena_runtime_msgs.srv.RegisterEnv,
            self.service_namespace("register_env"),
            self._cb_register_env,
            callback_group=srv_cb_group,
        )
        self._srv_spawn_env = self.create_service(
            arena_runtime_msgs.srv.SpawnEnv,
            self.service_namespace("spawn_env"),
            self._cb_spawn_env,
            callback_group=srv_cb_group,
        )
        self._srv_despawn_env = self.create_service(
            arena_runtime_msgs.srv.DespawnEnv,
            self.service_namespace("despawn_env"),
            self._cb_despawn_env,
            callback_group=srv_cb_group,
        )
        self._srv_confirm_world = self.create_service(
            arena_runtime_msgs.srv.ConfirmWorld,
            self.service_namespace("confirm_world"),
            self._cb_confirm_world,
            callback_group=srv_cb_group,
        )

        self._publish_state()
        self._publish_envs()

        self._heartbeat_timer = self.wall_timer(period, self._cb_heartbeat_check)

        self.get_logger().info(f"node initialized (sim={sim_name})")

        self.rosparam.declare_safe(
            "env_args",
            [],
            descriptor=ParameterDescriptor(type=Parameter.Type.STRING_ARRAY.value),
        )
        self.rosparam.declare_safe(
            "slot_buffer",
            slot_buffer,
            descriptor=ParameterDescriptor(type=Parameter.Type.DOUBLE.value),
        )

        env_n = self.rosparam[int].get("env_n", 0)
        if env_n > 0:
            asyncio.create_task(self._spawn_initial_envs(env_n))

        # bootstrap-only: read once here, runtime control goes through the services
        self.rosparam.declare_safe(
            "lockstep.channels",
            [],
            descriptor=ParameterDescriptor(type=Parameter.Type.STRING_ARRAY.value),
        )
        channels = list(self.get_parameter("lockstep.channels").value or [])
        bootstrap_ok = True
        if channels:
            try:
                specs = [_parse_channel_spec(e) for e in channels if e]
            except ValueError as e:
                self.get_logger().error(f"lockstep bootstrap channels invalid: {e}")
                bootstrap_ok = False
            else:
                self._lockstep.register("launch", "", specs)
        if bootstrap_ok and self.rosparam[bool].get("lockstep.autostart", False):
            rtf = self.rosparam[float].get("lockstep.target_rtf", 0.0)
            try:
                config = LockstepConfig(target_rtf=rtf, ungated=False)
            except ValueError as e:
                self.get_logger().error(f"lockstep autostart config invalid: {e}")
            else:
                await self._lockstep.start(config)
                if self.rosparam[bool].get("lockstep.paused", True):
                    self._lockstep.pause()
                    self.get_logger().warning("lockstep paused")

        self.trigger_configure()
        self.trigger_activate()

    async def _spawn_initial_envs(self, n: int) -> None:
        """Self-orchestrate the initial env fleet."""

        async def _spawn_one(i: int) -> None:
            req = arena_runtime_msgs.srv.SpawnEnv.Request()
            req.headless = False
            req.launch_args = []
            resp = arena_runtime_msgs.srv.SpawnEnv.Response()
            try:
                await self._cb_spawn_env(req, resp)
            except Exception as e:
                self.get_logger().error(f"spawn env {i} crashed: {e!r}")
                return
            if not resp.success:
                self.get_logger().error(f"spawn env {i} failed: {resp.error_msg}")

        await asyncio.gather(*(_spawn_one(i) for i in range(n)))

    async def _publish_clock_loop(self, lifecycle: DummyHost) -> None:
        """Publish simulated clock at ~100Hz using wall time, only when sim is dummy."""
        start = self.wall_time
        paused_duration = Time()
        pause_start: Time | None = None
        try:
            while True:
                if lifecycle.paused:
                    if pause_start is None:
                        pause_start = self.wall_time
                    await asyncio.sleep(0.01)
                    continue

                if pause_start is not None:
                    paused_duration += self.wall_time - pause_start
                    pause_start = None

                elapsed = self.wall_time - start - paused_duration
                self._clock_publisher.publish(elapsed.to_rosgraph_msg())
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.get_logger().error(f"clock loop crashed: {e!r}")

    def _publish_state(self) -> None:
        self._pub_paused.publish(Bool(data=not self._holds.is_empty()))
        self._pub_holders.publish(arena_runtime_msgs.msg.HoldRegistry(holds=self._holds.snapshot()))

    def _publish_envs(self) -> None:
        self._pub_envs.publish(arena_runtime_msgs.msg.EnvRegistry(envs=self._env_registry.snapshot()))

    def _make_managed_env(self, env_id: int, namespace: str) -> ManagedEnv:
        transition_topic = f"/{namespace}/transition_event"
        heartbeat_topic = f"/{namespace}/state/heartbeat"

        def _transition_cb(msg: TransitionEvent) -> None:
            self._on_transition_event(env_id, msg)

        def _heartbeat_cb(msg: arena_runtime_msgs.msg.Heartbeat) -> None:
            self._env_registry.update_heartbeat(env_id, msg.stamp)

        transition_sub = self.create_subscription(TransitionEvent, transition_topic, _transition_cb, 10)
        heartbeat_sub = self.create_subscription(arena_runtime_msgs.msg.Heartbeat, heartbeat_topic, _heartbeat_cb, 10)

        env = ManagedEnv(
            env_id=env_id,
            namespace=namespace,
            transition_sub=transition_sub,
            heartbeat_sub=heartbeat_sub,
        )
        self._envs[env_id] = env
        return env

    def _on_transition_event(self, env_id: int, msg: TransitionEvent) -> None:
        goal_state_id = msg.goal_state.id
        ready = goal_state_id == LcState.PRIMARY_STATE_ACTIVE
        self._env_registry.update_ready(env_id, ready)
        self._publish_envs()
        if ready:
            env = self._envs.get(env_id)
            if env is not None and env.spawn_ready is not None:
                fut = env.spawn_ready

                def _resolve() -> None:
                    if not fut.done():
                        fut.set_result(None)

                self.event_loop.call_soon_threadsafe(_resolve)
        if goal_state_id == LcState.PRIMARY_STATE_INACTIVE:
            env = self._envs.get(env_id)
            if env is not None:
                asyncio.run_coroutine_threadsafe(
                    self._activate_env(env_id, env.namespace),
                    self.event_loop,
                )
        if goal_state_id == LcState.PRIMARY_STATE_FINALIZED:
            asyncio.run_coroutine_threadsafe(
                self._evict(env_id, "lifecycle_finalized"),
                self.event_loop,
            )

    async def _activate_env(self, env_id: int, namespace: str) -> None:
        try:
            ok = await self.change_lifecycle_state_async(
                f"/{namespace}",
                Transition.TRANSITION_ACTIVATE,
                timeout=None,
            )
            if not ok:
                self.get_logger().error(f"failed to activate env_{env_id}")
        except Exception as e:
            self.get_logger().error(f"activate env_{env_id} raised: {e!r}")

    def _cb_heartbeat_check(self) -> None:
        asyncio.run_coroutine_threadsafe(
            self._check_heartbeats(),
            self.event_loop,
        )

    async def _check_heartbeats(self) -> None:
        timeout = self.rosparam[float].get("heartbeat_timeout_sec", 5.0)
        reset_timeout = self.rosparam[float].get("reset_hold_timeout_sec", 30.0)
        bootstrap_timeout = self.rosparam[float].get("bootstrap_timeout_sec", 600.0)
        now = self.wall_time
        for env_id, record in self._env_registry.items():
            elapsed = (now - Time.from_msg(record.last_heartbeat)).to_seconds()
            env = self._envs.get(env_id)
            process_alive = env.popen.poll() is None if env is not None and env.popen is not None else None
            # an env mid-reset blocks its loop for seconds; tolerate longer while it holds "reset", but still cap it
            reason = sweep_verdict(
                record,
                elapsed=elapsed,
                has_reset_hold=self._holds.has(record.fqn, "reset"),
                process_alive=process_alive,
                heartbeat_timeout=timeout,
                reset_timeout=reset_timeout,
                bootstrap_timeout=bootstrap_timeout,
            )
            if reason is not None:
                await self._evict(env_id, reason)

    async def _evict(self, env_id: int, reason: str) -> None:
        if not self._env_registry.start_eviction(env_id):
            return

        record = self._env_registry.get(env_id)
        if record is None:
            return

        self._publish_envs()

        await self._force_release_window(record.fqn)

        self._holds.release_all(record.fqn)
        if self._holds.is_empty() and self._windows.is_empty():
            await self._lifecycle.unpause()
        self._publish_state()

        self._lockstep.drop_env(record.fqn)

        self._publish_shutdown_request(env_id, reason)

        env = self._envs.pop(env_id, None)
        if env is not None:
            if env.popen is not None:
                await env.dispose(self, grace_seconds=15.0)
            else:
                await env.dispose(self, grace_seconds=0.0)
                await asyncio.sleep(2.0)  # give external env time to observe shutdown_request

        await self._lifecycle.cleanup_namespace(self._lifecycle.env_prefix(env_id))

        self._env_registry.complete_eviction(env_id)
        self._publish_envs()

        self.get_logger().info(f"evicted env_{env_id} ({reason})")

    async def _acquire_hold(self, caller: str, reason: str) -> int:
        """Acquire a hold, pausing the sim on the empty->nonempty edge if no window is open."""
        was_empty = self._holds.is_empty()
        count = self._holds.acquire(caller, reason)
        if was_empty and self._windows.is_empty():
            await self._lifecycle.pause()
        return count

    async def _release_hold(self, caller: str, reason: str) -> int:
        """Release a hold, unpausing the sim on the nonempty->empty edge if no window is open."""
        count = self._holds.release(caller, reason)
        if self._holds.is_empty() and self._windows.is_empty():
            await self._lifecycle.unpause()
        return count

    async def _cb_hold(
        self,
        request: arena_runtime_msgs.srv.LifecycleHold.Request,
        response: arena_runtime_msgs.srv.LifecycleHold.Response,
    ) -> arena_runtime_msgs.srv.LifecycleHold.Response:
        caller = request.caller_id
        reason = request.reason

        if request.action == arena_runtime_msgs.srv.LifecycleHold.Request.ACQUIRE:
            count = await self._acquire_hold(caller, reason)
        elif request.action == arena_runtime_msgs.srv.LifecycleHold.Request.RELEASE:
            count = await self._release_hold(caller, reason)
        else:
            self.get_logger().warning(f"unknown action {request.action}; ignoring")
            count = self._holds.total_count()

        self._publish_state()

        response.paused = not self._holds.is_empty()
        response.hold_count = count
        return response

    async def _cb_unpause_window(
        self,
        request: arena_runtime_msgs.srv.LifecycleUnpauseWindow.Request,
        response: arena_runtime_msgs.srv.LifecycleUnpauseWindow.Response,
    ) -> arena_runtime_msgs.srv.LifecycleUnpauseWindow.Response:
        if request.action == arena_runtime_msgs.srv.LifecycleUnpauseWindow.Request.ACQUIRE:
            was_empty = self._windows.is_empty()
            self._windows.acquire(request.caller_id, _WINDOW_REASON)
            if was_empty:
                try:
                    unpaused = await self._lifecycle.unpause()
                except Exception as e:
                    self._windows.release(request.caller_id, _WINDOW_REASON)
                    response.success = False
                    response.error_msg = f"unpause failed: {e!r}"
                    return response
                if not unpaused:
                    self._windows.release(request.caller_id, _WINDOW_REASON)
                    response.success = False
                    response.error_msg = "unpause returned failure"
                    return response
            response.success = True
            response.error_msg = ""
            return response

        if request.action == arena_runtime_msgs.srv.LifecycleUnpauseWindow.Request.RELEASE:
            if not self._windows.has(request.caller_id, _WINDOW_REASON):
                response.success = False
                response.error_msg = f"caller {request.caller_id} does not hold a window"
                return response
            self._windows.release(request.caller_id, _WINDOW_REASON)
            if self._windows.is_empty() and not self._holds.is_empty():
                await self._lifecycle.pause()
            response.success = True
            response.error_msg = ""
            return response

        self.get_logger().warning(f"unknown unpause_window action {request.action}; ignoring")
        response.success = False
        response.error_msg = f"unknown action {request.action}"
        return response

    async def _cb_step(
        self,
        request: arena_runtime_msgs.srv.LifecycleStep.Request,
        response: arena_runtime_msgs.srv.LifecycleStep.Response,
    ) -> arena_runtime_msgs.srv.LifecycleStep.Response:
        if self._holds.is_empty() or not self._windows.is_empty():
            response.success = False
            response.advanced = 0.0
            response.error_msg = "step requires an active hold and no open unpause window"
            return response

        try:
            response.advanced = await self._lifecycle.step_seconds(request.seconds)
        except NotImplementedError:
            response.success = False
            response.advanced = 0.0
            response.error_msg = "unsupported for this simulator"
            return response
        except RuntimeError as e:
            response.success = False
            response.advanced = 0.0
            response.error_msg = str(e)
            return response

        response.success = True
        response.error_msg = ""
        return response

    async def _cb_lockstep_start(
        self,
        request: arena_runtime_msgs.srv.LockstepStart.Request,
        response: arena_runtime_msgs.srv.LockstepStart.Response,
    ) -> arena_runtime_msgs.srv.LockstepStart.Response:
        try:
            config = LockstepConfig(target_rtf=request.target_rtf, ungated=request.ungated)
        except ValueError as e:
            response.success = False
            response.error_msg = str(e)
            return response
        await self._lockstep.start(config)
        response.success = True
        response.error_msg = ""
        return response

    async def _cb_lockstep_register(
        self,
        request: arena_runtime_msgs.srv.LockstepRegister.Request,
        response: arena_runtime_msgs.srv.LockstepRegister.Response,
    ) -> arena_runtime_msgs.srv.LockstepRegister.Response:
        registration = request.registration
        try:
            self._lockstep.register(
                registration.caller,
                registration.env,
                [ChannelSpec.from_msg(ch) for ch in registration.channels],
            )
        except ValueError as e:
            response.success = False
            response.error_msg = str(e)
            return response
        response.success = True
        response.error_msg = ""
        return response

    async def _cb_lockstep_stop(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        await self._lockstep.stop()
        response.success = True
        response.message = ""
        return response

    async def _cb_lockstep_pause(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        response.success = self._lockstep.pause()
        response.message = "" if response.success else "lockstep not running"
        return response

    async def _cb_lockstep_resume(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        response.success = self._lockstep.resume()
        response.message = "" if response.success else "lockstep not running"
        return response

    async def _force_release_window(self, caller_id: str) -> None:
        """Force-release all unpause windows held by a caller, e.g. on eviction."""
        if not self._windows.has(caller_id, _WINDOW_REASON):
            return
        self._windows.release_all(caller_id)
        if self._windows.is_empty() and not self._holds.is_empty():
            await self._lifecycle.pause()

    async def _cb_cleanup_env(
        self,
        request: arena_runtime_msgs.srv.CleanupEnv.Request,
        response: arena_runtime_msgs.srv.CleanupEnv.Response,
    ) -> arena_runtime_msgs.srv.CleanupEnv.Response:
        prefix = self._lifecycle.env_prefix(request.env_id)
        removed = await self._lifecycle.cleanup_namespace(prefix)
        self.get_logger().info(f"cleanup env_{request.env_id} ('{prefix}') removed {removed}")
        response.success = True
        response.removed = removed
        response.error_msg = ""
        return response

    async def _cb_register_env(
        self,
        request: arena_runtime_msgs.srv.RegisterEnv.Request,
        response: arena_runtime_msgs.srv.RegisterEnv.Response,
    ) -> arena_runtime_msgs.srv.RegisterEnv.Response:
        requested = request.env_id if request.env_id != 0xFFFF else None
        requested_ns = request.ns or None

        try:
            env_id, namespace = self._env_registry.reserve(
                requested_env_id=requested,
                requested_ns=requested_ns,
                now=self.wall_time.to_msg(),
            )
        except ValueError as e:
            response.success = False
            response.error_msg = str(e)
            return response

        self._make_managed_env(env_id, namespace)

        self._publish_envs()
        self.get_logger().info(f"registered env_{env_id} (ns={namespace})")

        response.success = True
        response.env_id = env_id
        response.ns = namespace
        response.sim = self.rosparam[str].get("sim", SimSimulator.DUMMY.value)
        response.error_msg = ""
        return response

    def _publish_shutdown_request(self, env_id: int, reason: str) -> None:
        msg = arena_runtime_msgs.msg.ShutdownRequest(env_id=env_id, reason=reason)
        self._pub_shutdown_request.publish(msg)

    async def _shutdown_pg(
        self,
        env_id: int,
        popen: subprocess.Popen,
        *,
        grace_seconds: float,
    ) -> None:
        """Reap a wrapper-python pgid: optional grace, SIGTERM-pgid, SIGKILL-pgid.

        The wrapper is started with start_new_session=True so killpg targets its
        own group, not arena_node's.
        """
        loop = asyncio.get_running_loop()
        if popen.poll() is not None:
            return
        if grace_seconds > 0:
            try:
                await asyncio.wait_for(loop.run_in_executor(None, popen.wait), timeout=grace_seconds)
                return
            except TimeoutError:
                self.get_logger().warning(f"env_{env_id} did not exit within {grace_seconds:.1f}s, sending SIGTERM")
        try:
            os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as e:
            self.get_logger().warning(f"SIGTERM env_{env_id} failed: {e!r}")
        try:
            await asyncio.wait_for(loop.run_in_executor(None, popen.wait), timeout=2.0)
            return
        except TimeoutError:
            self.get_logger().warning(f"env_{env_id} did not exit after SIGTERM, sending SIGKILL")
        try:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception as e:
            self.get_logger().warning(f"SIGKILL env_{env_id} failed: {e!r}")

    async def teardown(self) -> None:
        envs = list(self._envs.values())
        self._envs.clear()
        if not envs:
            return
        self.get_logger().info(f"reaping {len(envs)} managed env(s)")
        for env in envs:
            with contextlib.suppress(Exception):
                self._publish_shutdown_request(env.env_id, "arena_node_teardown")
        await asyncio.gather(
            *(env.dispose(self, grace_seconds=2.0) for env in envs),
            return_exceptions=True,
        )

    _LAUNCH_WRAPPER_SCRIPT = """
import json
import os
import sys

payload = json.loads(os.environ.pop('ARENA_LAUNCH_PAYLOAD'))
sys.argv = [sys.argv[0]]

import launch
import launch.actions
import launch.launch_description_sources

src = launch.launch_description_sources.PythonLaunchDescriptionSource(payload['launchfile'])
ld = launch.LaunchDescription([
    launch.actions.IncludeLaunchDescription(src, launch_arguments=payload['args']),
])
ls = launch.LaunchService()
ls.include_launch_description(ld)
sys.exit(ls.run())
"""

    async def _cb_spawn_env(
        self,
        request: arena_runtime_msgs.srv.SpawnEnv.Request,
        response: arena_runtime_msgs.srv.SpawnEnv.Response,
    ) -> arena_runtime_msgs.srv.SpawnEnv.Response:
        sim_name = self.rosparam[str].get("sim", SimSimulator.DUMMY.value)
        requested_ns = request.ns or None

        try:
            env_id, namespace = self._env_registry.reserve(
                requested_env_id=None,
                requested_ns=requested_ns,
                now=self.wall_time.to_msg(),
            )
        except ValueError as e:
            response.success = False
            response.error_msg = f"reserve failed: {e}"
            return response

        env = self._make_managed_env(env_id, namespace)
        env.spawn_ready = self.event_loop.create_future()
        self._publish_envs()

        launchfile = os.path.join(
            ament_index_python.packages.get_package_share_directory("task_generator"),
            "launch",
            "task_generator.launch.py",
        )
        per_spawn_fixed: dict[str, str] = {
            "env.managed": "true",
            "env.id": str(env_id),
            "env.ns": namespace,
            "sim": sim_name,
        }

        merged: dict[str, str] = {}
        for source in (self.get_parameter("env_args").value or [], request.launch_args):
            for s in source:
                if ":=" in s:
                    k, v = s.split(":=", 1)
                    merged[k] = v
        merged.update(per_spawn_fixed)

        launch_args: list[list[str]] = [[k, v] for k, v in merged.items()]

        proc_env = {
            **os.environ,
            "ARENA_LAUNCH_PAYLOAD": json.dumps(
                {
                    "launchfile": launchfile,
                    "args": launch_args,
                }
            ),
        }

        log_dir = pathlib.Path.home() / ".ros" / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"arena_env_{env_id}_{ts}.log"
        env.log_path = log_path

        try:
            log_file = log_path.open("wb")
        except Exception as e:
            self._envs.pop(env_id, None)
            await env.dispose(self, grace_seconds=0.0)
            self._env_registry.free(env_id)
            self._publish_envs()
            response.success = False
            response.error_msg = f"open log file failed: {e!r}"
            return response

        try:
            env.popen = subprocess.Popen(
                [sys.executable, "-c", self._LAUNCH_WRAPPER_SCRIPT],
                env=proc_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as e:
            log_file.close()
            self._envs.pop(env_id, None)
            await env.dispose(self, grace_seconds=0.0)
            self._env_registry.free(env_id)
            self._publish_envs()
            response.success = False
            response.error_msg = f"popen failed: {e!r}"
            return response

        echo_prefix = None if request.headless else f"[env_{env_id}] ".encode()
        env.tee_thread = threading.Thread(
            target=_tee_subprocess_output,
            args=(env.popen.stdout, log_file, echo_prefix),
            name=f"arena_env_{env_id}_tee",
            daemon=True,
        )
        env.tee_thread.start()

        self.get_logger().info(f"env_{env_id} spawned (ns={namespace}), logs at {log_path}")

        try:
            await self.await_forever(env.spawn_ready, f"env_{env_id} spawn ACTIVE")
        except (Exception, asyncio.CancelledError) as e:
            if isinstance(e, asyncio.CancelledError) and not env.spawn_ready.cancelled():
                raise
            self._envs.pop(env_id, None)
            await env.dispose(self, grace_seconds=0.0)
            self._env_registry.free(env_id)
            self._publish_envs()
            response.success = False
            response.error_msg = f"spawn ACTIVE wait failed: {e!r} (see {log_path} for details)"
            response.log_path = str(log_path)
            return response

        self.get_logger().info(f"env_{env_id} reached ACTIVE")

        response.success = True
        response.env_id = env_id
        response.ns = namespace
        response.log_path = str(log_path)
        response.error_msg = ""
        return response

    async def _cb_despawn_env(
        self,
        request: arena_runtime_msgs.srv.DespawnEnv.Request,
        response: arena_runtime_msgs.srv.DespawnEnv.Response,
    ) -> arena_runtime_msgs.srv.DespawnEnv.Response:
        env_id = request.env_id
        record = self._env_registry.get(env_id)
        if record is None:
            response.success = False
            response.error_msg = f"env_id {env_id} not registered"
            return response
        self._publish_shutdown_request(env_id, "despawn_env")
        self.get_logger().info(f"despawn env_{env_id}")
        response.success = True
        response.error_msg = ""
        return response

    async def _cb_confirm_world(
        self,
        request: arena_runtime_msgs.srv.ConfirmWorld.Request,
        response: arena_runtime_msgs.srv.ConfirmWorld.Response,
    ) -> arena_runtime_msgs.srv.ConfirmWorld.Response:
        record = self._env_registry.get(request.env_id)
        if record is None:
            response.success = False
            response.reallocated = False
            response.error_msg = f"env_id {request.env_id} not registered"
            return response

        was_placed = record.placed
        prior_extent = record.extent

        try:
            placement = self._env_registry.place(request.env_id, request.extent)
        except ValueError as e:
            response.success = False
            response.reallocated = False
            response.error_msg = str(e)
            return response

        self._publish_envs()

        reallocated = (not was_placed) or (not _extent_eq(prior_extent, request.extent))
        self.get_logger().info(f"placed env_{request.env_id} (reallocated={reallocated})")

        response.success = True
        response.reference = list(placement.reference)
        response.slot_extent = list(placement.slot_extent)
        response.prespawn = list(placement.prespawn)
        response.reallocated = reallocated
        response.error_msg = ""
        return response


def main(args: list[str] | None = None) -> None:
    del args
    ArenaNode.run_main()
