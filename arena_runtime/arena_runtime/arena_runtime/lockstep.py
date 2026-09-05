from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import typing

import arena_runtime_msgs.msg
import rclpy.qos
import rclpy.subscription
from arena_rclpy_mixins.Time import Time
from rosidl_runtime_py.utilities import get_message

from arena_runtime.lockstep_gates import Channel, ChannelRegistry, ChannelSpec, GateLedger, parse_channel_entry, step_slice

if typing.TYPE_CHECKING:
    from arena_runtime.arena_node import ArenaNode

_LOCKSTEP_REASON = "lockstep"
_STATUS_PERIOD = 1.0
_STALL_AFTER = 1.0

_LATCHED = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)


def spec_from_msg(msg: arena_runtime_msgs.msg.LockstepChannel) -> ChannelSpec:
    return ChannelSpec(name=msg.name, topic_template=msg.topic, type_name=msg.type, period_s=msg.period_s, hard=msg.hard)


def spec_to_msg(spec: ChannelSpec) -> arena_runtime_msgs.msg.LockstepChannel:
    return arena_runtime_msgs.msg.LockstepChannel(
        name=spec.name,
        topic=spec.topic_template,
        type=spec.type_name,
        period_s=spec.period_s,
        hard=spec.hard,
    )


def _validate_spec(spec: ChannelSpec) -> None:
    try:
        get_message(spec.type_name)
    except (AttributeError, LookupError, ModuleNotFoundError, ValueError) as e:
        raise ValueError(f"lockstep channel {spec.name!r}: cannot resolve type {spec.type_name!r}") from e


def _parse_channel_spec(entry: str) -> ChannelSpec:
    spec = parse_channel_entry(entry)
    _validate_spec(spec)
    return spec


@dataclasses.dataclass(frozen=True)
class LockstepConfig:
    """Run parameters only. The gate set comes from the registration registry."""

    target_rtf: float
    ungated: bool

    def __post_init__(self) -> None:
        if self.target_rtf < 0.0:
            raise ValueError("target_rtf must be >= 0")


class LockstepScheduler:
    """Owns the lockstep step/wait loop: holds the sim paused and steps it exactly one
    channel period at a time, gating on HARD channels' latest header.stamp."""

    def __init__(self, node: ArenaNode) -> None:
        self._node = node
        self._config: LockstepConfig | None = None
        self._task: asyncio.Task | None = None
        self._resume_task: asyncio.Task | None = None
        self._measured_rtf = 0.0
        self._waiting: list[str] = []
        self._resume = asyncio.Event()
        self._resume.set()
        self._arrived = asyncio.Event()
        self._registry = ChannelRegistry()
        self._pub_status = node.create_publisher(
            arena_runtime_msgs.msg.LockstepStatus,
            node.service_namespace("state", "lockstep"),
            _LATCHED,
        )
        self._publish_status()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def owns_current_task(self) -> bool:
        """True when called from inside the loop task itself."""
        return self._task is not None and asyncio.current_task() is self._task

    def register(self, caller: str, env: str, channels: typing.Sequence[ChannelSpec]) -> None:
        """Replace the caller's registration (empty channels clears it). Raises ValueError with no state change."""
        for spec in channels:
            _validate_spec(spec)
        self._registry.register(caller, env, channels)
        self._publish_status()
        self._wake()

    def drop_env(self, env: str) -> None:
        """Drop all registrations owned by a despawned env."""
        if self._registry.drop_env(env):
            self._publish_status()
            self._wake()

    def _wake(self) -> None:
        """Let a gate wait re-read the ledger: a message arrived or the registry changed."""
        self._node.event_loop.call_soon_threadsafe(self._arrived.set)

    async def start(self, config: LockstepConfig) -> None:
        if self.running:
            # reconfigure: swap the task under the existing hold, no unpause blip
            await self._cancel_task()
        else:
            fqn = self._node.get_fully_qualified_name()
            await self._node._acquire_hold(fqn, _LOCKSTEP_REASON)
            self._node._publish_state()
        self._config = config
        self._resume.set()
        self._task = asyncio.create_task(self._run(config))

    async def stop(self) -> None:
        if self.running:
            await self._cancel_task()
            fqn = self._node.get_fully_qualified_name()
            await self._node._release_hold(fqn, _LOCKSTEP_REASON)
            self._node._publish_state()
            self._resume_task = asyncio.create_task(self._ensure_resumed())
        self._config = None
        self._measured_rtf = 0.0
        self._resume.set()
        self._publish_status()

    def pause(self) -> bool:
        """Suspend stepping at the next tick boundary. The run and its hold stay intact."""
        if not self.running:
            return False
        self._resume.clear()
        self._publish_status()
        return True

    def resume(self) -> bool:
        if not self.running:
            return False
        self._resume.set()
        self._publish_status()
        return True

    async def _ensure_resumed(self) -> None:
        """A step request in flight during cancel re-pauses gz after its chunk drains.
        Once the clock quiesces, reassert whatever the hold ledger implies."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        last = self._node.sim_time
        quiet = 0.0
        while loop.time() < deadline:
            await asyncio.sleep(0.1)
            if self._node.sim_time > last:
                last = self._node.sim_time
                quiet = 0.0
                continue
            quiet += 0.1
            if quiet >= 0.5:
                if self._node._holds.is_empty() and self._node._windows.is_empty():
                    await self._node._lifecycle.unpause()
                return

    async def _cancel_task(self) -> None:
        task = self._task
        self._task = None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    def _foreign_hold_active(self) -> bool:
        fqn = self._node.get_fully_qualified_name()
        return any(entry.caller_id != fqn or entry.reason != _LOCKSTEP_REASON for entry in self._node._holds.snapshot())

    async def _await_quiet_clock(self) -> None:
        """A cancelled predecessor (reconfigure, prior client) may have left the sim
        draining queued step iterations, so baseline only once the clock is quiet."""
        last_seen = self._node.sim_time
        quiet = 0.0
        while quiet < 0.3:
            await asyncio.sleep(0.05)
            if self._node.sim_time > last_seen:
                last_seen = self._node.sim_time
                quiet = 0.0
            else:
                quiet += 0.05

    async def _idle_while_owned(self) -> None:
        """An open unpause window (robot bringup, teleport) or a foreign hold (env reset)
        owns the sim, and a step now would advance it under the holder."""
        reassert = 0.0
        while not self._node._windows.is_empty() or self._foreign_hold_active():
            # a chunk in flight when a window opened re-pauses gz at its
            # end, clobbering the window's unpause, so re-assert while idling
            if not self._node._windows.is_empty():
                if reassert <= 0.0:
                    await self._node._lifecycle.unpause()
                    reassert = 0.5
                reassert -= 0.05
            else:
                reassert = 0.0
            await asyncio.sleep(0.05)

    async def _run(self, config: LockstepConfig) -> None:
        dt = self._node.rosparam[float].get('physics_dt', 0.0333)
        await self._await_quiet_clock()
        ledger = GateLedger(dt=dt, base=self._node.sim_time.to_seconds())
        subs: dict[str, rclpy.subscription.Subscription] = {}
        loop = asyncio.get_running_loop()
        window_start = loop.time()
        window_advance = 0.0
        try:
            while True:
                await self._resume.wait()
                iter_start = loop.time()
                sim_before = ledger.now
                await self._idle_while_owned()

                # the window/hold ran the sim outside this loop's bookkeeping, so
                # rebase or every gate is satisfied by the previous window's stamp
                drift = ledger.rebase(self._node.sim_time.to_seconds())
                if drift > 0.0:
                    self._node.get_logger().info(f"lockstep: sim advanced {drift:.3f}s outside the run, rebasing")
                    sim_before = ledger.now

                if not config.ungated:
                    removed, added = ledger.refresh(self._desired())
                    for ch in removed:
                        self._node.destroy_subscription(subs.pop(ch.topic))
                    for ch in added:
                        subs[ch.topic] = self._subscribe(ledger, ch)

                if config.ungated or not ledger.channels:
                    ledger.advance(await self._node._lifecycle.step_seconds(step_slice(config.target_rtf, dt)))
                    self._node._lifecycle.record_success()
                else:
                    ledger.advance(await self._node._lifecycle.step_seconds(ledger.next_delta()))
                    self._node._lifecycle.record_success()
                    due = ledger.due()
                    hard_due = [ch for ch in due if ch.hard]
                    if hard_due:
                        await self._await_hard(ledger, hard_due)
                    ledger.complete(due)

                window_advance += ledger.now - sim_before
                elapsed = loop.time() - window_start
                if elapsed >= _STATUS_PERIOD:
                    self._measured_rtf = window_advance / elapsed
                    window_start = loop.time()
                    window_advance = 0.0
                    self._publish_status(tick=Time.from_float(ledger.tick))

                if config.target_rtf > 0.0:
                    lag = (ledger.now - sim_before) / config.target_rtf - (loop.time() - iter_start)
                    if lag > 0.0:
                        await asyncio.sleep(lag)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._node.get_logger().error(f"lockstep scheduler crashed: {e!r}")
            await self._node._lifecycle.record_failure(f"lockstep step: {e!r}")
            fqn = self._node.get_fully_qualified_name()
            await self._node._release_hold(fqn, _LOCKSTEP_REASON)
            self._node._publish_state()
            self._config = None
            self._measured_rtf = 0.0
            self._publish_status()
        finally:
            self._waiting = []
            for sub in subs.values():
                self._node.destroy_subscription(sub)

    def _desired(self) -> dict[str, tuple[ChannelSpec, str]]:
        return self._registry.desired(env_id for env_id, record in self._node._env_registry.items() if not record.draining)

    def _subscribe(self, ledger: GateLedger, channel: Channel) -> rclpy.subscription.Subscription:
        loop = self._node.event_loop

        def _cb(msg: object) -> None:
            ledger.observe(channel, Time.from_msg(msg.header.stamp).to_seconds())
            loop.call_soon_threadsafe(self._arrived.set)

        return self._node.create_subscription(get_message(channel.spec.type_name), channel.topic, _cb, 10)

    async def _await_hard(self, ledger: GateLedger, hard_due: list[Channel]) -> None:
        start = self._node.wall_time
        next_warn = 10.0
        stalled = False
        version = self._registry.version
        due_time = Time.from_float(ledger.tick)
        while True:
            self._arrived.clear()
            if self._registry.version != version:
                # a deregistered channel (env despawn) must not stall the tick forever
                version = self._registry.version
                desired = self._desired()
                hard_due = [ch for ch in hard_due if ch.topic in desired]
            waiting = ledger.waiting(hard_due)
            if not waiting:
                break
            elapsed = (self._node.wall_time - start).to_seconds()
            if elapsed >= _STALL_AFTER and not stalled:
                stalled = True
                self._publish_stall(ledger, due_time, hard_due)
            if elapsed >= next_warn:
                for ch in waiting:
                    self._node.get_logger().warning(f"lockstep: waiting on {ch.name} ({ch.topic}) for {elapsed:.0f}s")
                self._publish_stall(ledger, due_time, hard_due)
                next_warn += 10.0
            threshold = next_warn if stalled else min(_STALL_AFTER, next_warn)
            try:
                await asyncio.wait_for(self._arrived.wait(), timeout=max(threshold - elapsed, 0.01))
            except TimeoutError:
                pass
        self._waiting = []
        if stalled:
            self._publish_status(tick=due_time, arrived=[ch.name for ch in hard_due])

    def _publish_stall(self, ledger: GateLedger, due_time: Time, hard_due: list[Channel]) -> None:
        waiting_on = [ch.name for ch in ledger.waiting(hard_due)]
        arrived = [ch.name for ch in hard_due if ch.name not in waiting_on]
        self._waiting = waiting_on
        self._publish_status(tick=due_time, waiting_on=waiting_on, arrived=arrived)

    def _snapshot_registrations(self) -> list[arena_runtime_msgs.msg.LockstepRegistration]:
        return [
            arena_runtime_msgs.msg.LockstepRegistration(
                caller=caller,
                env=reg.env,
                channels=[spec_to_msg(spec) for spec in reg.channels],
            )
            for caller, reg in self._registry.items()
        ]

    def _publish_status(
        self,
        tick: Time | None = None,
        waiting_on: typing.Sequence[str] | None = None,
        arrived: typing.Sequence[str] = (),
    ) -> None:
        """Status reflects the live gate: a publish from register/drop mid-stall keeps waiting_on."""
        config = self._config
        self._pub_status.publish(
            arena_runtime_msgs.msg.LockstepStatus(
                active=config is not None,
                paused=config is not None and not self._resume.is_set(),
                ungated=config.ungated if config is not None else False,
                target_rtf=config.target_rtf if config is not None else 0.0,
                measured_rtf=self._measured_rtf,
                tick=(tick if tick is not None else Time()).to_msg(),
                registrations=self._snapshot_registrations(),
                waiting_on=list(waiting_on) if waiting_on is not None else list(self._waiting),
                arrived=list(arrived),
            )
        )
