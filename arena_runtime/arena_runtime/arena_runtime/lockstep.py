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

if typing.TYPE_CHECKING:
    from arena_runtime.arena_node import ArenaNode

_LOCKSTEP_REASON = "lockstep"
_STATUS_PERIOD = 1.0
_STALL_AFTER = 1.0

_LATCHED = rclpy.qos.QoSProfile(
    depth=1,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclasses.dataclass(frozen=True)
class ChannelSpec:
    name: str
    topic_template: str
    type_name: str
    period_s: float
    hard: bool

    @classmethod
    def from_msg(cls, msg: arena_runtime_msgs.msg.LockstepChannel) -> ChannelSpec:
        return cls(name=msg.name, topic_template=msg.topic, type_name=msg.type, period_s=msg.period_s, hard=msg.hard)

    def to_msg(self) -> arena_runtime_msgs.msg.LockstepChannel:
        return arena_runtime_msgs.msg.LockstepChannel(
            name=self.name,
            topic=self.topic_template,
            type=self.type_name,
            period_s=self.period_s,
            hard=self.hard,
        )


def _validate_spec(spec: ChannelSpec) -> None:
    try:
        get_message(spec.type_name)
    except (AttributeError, LookupError, ModuleNotFoundError, ValueError) as e:
        raise ValueError(f"lockstep channel {spec.name!r}: cannot resolve type {spec.type_name!r}") from e
    if spec.period_s <= 0.0:
        raise ValueError(f"lockstep channel {spec.name!r}: period must be > 0")


def _parse_channel_spec(entry: str) -> ChannelSpec:
    try:
        name, topic, type_name, period_s, role = entry.split('|')
    except ValueError as e:
        raise ValueError(f"lockstep channel {entry!r}: expected name|topic|type|period_s|role") from e
    if role not in ('hard', 'soft'):
        raise ValueError(f"lockstep channel {name!r}: unknown role {role!r}")
    try:
        period = float(period_s)
    except ValueError as e:
        raise ValueError(f"lockstep channel {name!r}: bad period {period_s!r}") from e
    spec = ChannelSpec(name=name, topic_template=topic, type_name=type_name, period_s=period, hard=role == 'hard')
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


@dataclasses.dataclass(frozen=True)
class _Registration:
    env: str
    channels: tuple[ChannelSpec, ...]


def _env_match(reg_env: str, env: str) -> bool:
    a = reg_env.strip('/')
    b = env.strip('/')
    return a == b or a.startswith(b + '/') or b.startswith(a + '/')


def _step_slice(target_rtf: float, dt: float) -> float:
    if target_rtf <= 0.0:
        return 1.0
    return min(max(target_rtf * 0.05, dt), 1.0)


@dataclasses.dataclass
class _Channel:
    spec: ChannelSpec
    name: str
    topic: str
    hard: bool
    period: float
    next_due: float
    sub: rclpy.subscription.Subscription | None = None
    latest_stamp: Time | None = None


class LockstepScheduler:
    """Owns the lockstep step/wait loop: holds the sim paused and steps it exactly one
    channel period at a time, gating on HARD channels' latest header.stamp."""

    def __init__(self, node: ArenaNode) -> None:
        self._node = node
        self._config: LockstepConfig | None = None
        self._task: asyncio.Task | None = None
        self._measured_rtf = 0.0
        self._resume = asyncio.Event()
        self._resume.set()
        self._registrations: dict[str, _Registration] = {}
        self._registry_version = 0
        self._pub_status = node.create_publisher(
            arena_runtime_msgs.msg.LockstepStatus,
            node.service_namespace("state", "lockstep"),
            _LATCHED,
        )
        self._publish_status()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def register(self, caller: str, env: str, channels: typing.Sequence[ChannelSpec]) -> None:
        """Replace the caller's registration (empty channels clears it). Raises ValueError with no state change."""
        if not caller:
            raise ValueError("lockstep register: caller must be nonempty")
        for spec in channels:
            _validate_spec(spec)
        if channels:
            self._registrations[caller] = _Registration(env=env, channels=tuple(channels))
        else:
            self._registrations.pop(caller, None)
        self._registry_version += 1
        self._publish_status()

    def drop_env(self, env: str) -> None:
        """Drop all registrations owned by a despawned env."""
        dropped = [caller for caller, reg in self._registrations.items() if reg.env and _env_match(reg.env, env)]
        for caller in dropped:
            del self._registrations[caller]
        if dropped:
            self._registry_version += 1
            self._publish_status()

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
            await self._ensure_resumed()
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
        return any(
            entry.caller_id != fqn or entry.reason != _LOCKSTEP_REASON
            for entry in self._node._holds.snapshot()
        )

    async def _run(self, config: LockstepConfig) -> None:
        dt = self._node.rosparam[float].get('physics_dt', 0.0333)
        # a cancelled predecessor (reconfigure, prior client) may have left the sim
        # draining queued step iterations, so baseline only once the clock is quiet
        last_seen = self._node.sim_time
        quiet = 0.0
        while quiet < 0.3:
            await asyncio.sleep(0.05)
            if self._node.sim_time > last_seen:
                last_seen = self._node.sim_time
                quiet = 0.0
            else:
                quiet += 0.05
        base = self._node.sim_time
        now = 0.0
        channels: dict[str, _Channel] = {}
        loop = asyncio.get_running_loop()
        window_start = loop.time()
        window_advance = 0.0
        try:
            while True:
                await self._resume.wait()
                iter_start = loop.time()
                sim_before = now
                # an open unpause window (robot bringup, teleport) or a foreign hold
                # (env reset) owns the sim, and a step now would advance it under the holder
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

                # the window/hold ran the sim outside this loop's bookkeeping, so
                # rebase or every gate is satisfied by the previous window's stamp
                drift = (self._node.sim_time - base).to_seconds() - now
                if drift > dt / 2.0:
                    self._node.get_logger().info(f"lockstep: sim advanced {drift:.3f}s outside the run, rebasing")
                    now += drift
                    sim_before = now
                    for ch in channels.values():
                        ch.next_due = now + ch.period

                if not config.ungated:
                    self._refresh_channels(channels, dt, now)

                if config.ungated or not channels:
                    now += await self._node._lifecycle.step_seconds(_step_slice(config.target_rtf, dt))
                else:
                    due = min(ch.next_due for ch in channels.values())
                    delta = due - now
                    if delta <= 0.0:
                        delta = dt
                    now += await self._node._lifecycle.step_seconds(delta)
                    due_time = base + Time.from_float(now)

                    due_channels = [ch for ch in channels.values() if ch.next_due <= now + 1e-9]
                    hard_due = [ch for ch in due_channels if ch.hard]
                    if hard_due:
                        # a periodic producer promises one sample per period window,
                        # and its timer may fire mid-burst and reschedule past the
                        # frozen clock (rcl timers reschedule from fire time), so
                        # gate on window coverage, not tick freshness. dt/2 rejects
                        # the previous window's boundary sample
                        gates = {
                            ch.topic: base + Time.from_float(ch.next_due - ch.period + dt / 2.0)
                            for ch in hard_due
                        }
                        await self._await_hard(due_time, gates, hard_due)
                    for ch in due_channels:
                        ch.next_due += ch.period

                window_advance += now - sim_before
                elapsed = loop.time() - window_start
                if elapsed >= _STATUS_PERIOD:
                    self._measured_rtf = window_advance / elapsed
                    window_start = loop.time()
                    window_advance = 0.0
                    self._publish_status(tick=base + Time.from_float(now))

                if config.target_rtf > 0.0:
                    lag = (now - sim_before) / config.target_rtf - (loop.time() - iter_start)
                    if lag > 0.0:
                        await asyncio.sleep(lag)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._node.get_logger().error(f"lockstep scheduler crashed: {e!r}")
            fqn = self._node.get_fully_qualified_name()
            await self._node._release_hold(fqn, _LOCKSTEP_REASON)
            self._node._publish_state()
            self._config = None
            self._measured_rtf = 0.0
            self._publish_status()
        finally:
            for ch in channels.values():
                if ch.sub is not None:
                    self._node.destroy_subscription(ch.sub)

    def _desired(self) -> dict[str, tuple[ChannelSpec, str]]:
        """Union of all registrations, keyed by resolved topic (first registrant wins)."""
        env_ids = [env_id for env_id, record in self._node._env_registry.items() if not record.draining]

        desired: dict[str, tuple[ChannelSpec, str]] = {}
        for reg in self._registrations.values():
            for spec in reg.channels:
                if not reg.env and '{env}' in spec.topic_template:
                    for env_id in env_ids:
                        topic = spec.topic_template.replace('{env}', f'env_{env_id}')
                        desired.setdefault(topic, (spec, f"{spec.name}/env_{env_id}"))
                else:
                    desired.setdefault(spec.topic_template, (spec, spec.name))
        return desired

    def _refresh_channels(self, channels: dict[str, _Channel], dt: float, now: float) -> None:
        desired = self._desired()

        for topic in list(channels):
            ch = channels[topic]
            if desired.get(topic) != (ch.spec, ch.name):
                del channels[topic]
                if ch.sub is not None:
                    self._node.destroy_subscription(ch.sub)

        for topic, (spec, display) in desired.items():
            if topic in channels:
                continue
            # raw period, floored to one sim step: snapping to the sim grid would
            # break 1:1 fire-per-window pacing for grid-timer producers (engine)
            period = max(spec.period_s, dt)
            channel = _Channel(
                spec=spec,
                name=display,
                topic=topic,
                hard=spec.hard,
                period=period,
                next_due=now + period,
            )
            msg_type = get_message(spec.type_name)
            channel.sub = self._node.create_subscription(msg_type, topic, self._make_callback(channel), 10)
            channels[topic] = channel

    def _make_callback(self, channel: _Channel) -> typing.Callable[[object], None]:
        def _cb(msg: object) -> None:
            channel.latest_stamp = Time.from_msg(msg.header.stamp)

        return _cb

    async def _await_hard(self, due_time: Time, gates: dict[str, Time], hard_due: list[_Channel]) -> None:
        start = self._node.wall_time
        next_warn = 10.0
        stalled = False
        version = self._registry_version
        while True:
            if self._registry_version != version:
                # a deregistered channel (env despawn) must not stall the tick forever
                version = self._registry_version
                desired = self._desired()
                hard_due = [ch for ch in hard_due if ch.topic in desired]
            waiting = [ch for ch in hard_due if ch.latest_stamp is None or ch.latest_stamp < gates[ch.topic]]
            if not waiting:
                break
            elapsed = (self._node.wall_time - start).to_seconds()
            if elapsed >= _STALL_AFTER and not stalled:
                stalled = True
                self._publish_stall(due_time, gates, hard_due)
            if elapsed >= next_warn:
                for ch in waiting:
                    self._node.get_logger().warning(f"lockstep: waiting on {ch.name} ({ch.topic}) for {elapsed:.0f}s")
                self._publish_stall(due_time, gates, hard_due)
                next_warn += 10.0
            await asyncio.sleep(0.01)
        if stalled:
            self._publish_status(tick=due_time, arrived=[ch.name for ch in hard_due])

    def _publish_stall(self, due_time: Time, gates: dict[str, Time], hard_due: list[_Channel]) -> None:
        waiting_on = [ch.name for ch in hard_due if ch.latest_stamp is None or ch.latest_stamp < gates[ch.topic]]
        arrived = [ch.name for ch in hard_due if ch.name not in waiting_on]
        self._publish_status(tick=due_time, waiting_on=waiting_on, arrived=arrived)

    def _snapshot_registrations(self) -> list[arena_runtime_msgs.msg.LockstepRegistration]:
        return [
            arena_runtime_msgs.msg.LockstepRegistration(
                caller=caller,
                env=reg.env,
                channels=[spec.to_msg() for spec in reg.channels],
            )
            for caller, reg in self._registrations.items()
        ]

    def _publish_status(
        self,
        tick: Time | None = None,
        waiting_on: typing.Sequence[str] = (),
        arrived: typing.Sequence[str] = (),
    ) -> None:
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
                waiting_on=list(waiting_on),
                arrived=list(arrived),
            )
        )
