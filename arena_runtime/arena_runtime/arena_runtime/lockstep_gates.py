"""ROS-free lockstep bookkeeping: channel specs, the per-caller registry and one run's gate ledger.
Ledger times are sim seconds, `now` relative to the run's `base`, observed stamps absolute."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

_NS = 1_000_000_000


@dataclasses.dataclass(frozen=True)
class ChannelSpec:
    name: str
    topic_template: str
    type_name: str
    period_s: float
    hard: bool

    def __post_init__(self) -> None:
        if self.period_s <= 0.0:
            raise ValueError(f"lockstep channel {self.name!r}: period must be > 0")


def parse_channel_entry(entry: str) -> ChannelSpec:
    """`name|topic|type|period_s|role`. Type resolution is the caller's job."""
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
    return ChannelSpec(name=name, topic_template=topic, type_name=type_name, period_s=period, hard=role == 'hard')


def env_match(reg_env: str, env: str) -> bool:
    a = reg_env.strip('/')
    b = env.strip('/')
    return a == b or a.startswith(b + '/') or b.startswith(a + '/')


def step_slice(target_rtf: float, dt: float) -> float:
    """Ungated step size: 5% of a target-rtf second, clamped to [dt, 1.0], a full second when unbounded."""
    if target_rtf <= 0.0:
        return 1.0
    return min(max(target_rtf * 0.05, dt), 1.0)


@dataclasses.dataclass(frozen=True)
class Registration:
    env: str
    channels: tuple[ChannelSpec, ...]


class ChannelRegistry:
    """Per-caller registrations. Every mutation bumps `version` so a blocked gate can re-filter."""

    def __init__(self) -> None:
        self._entries: dict[str, Registration] = {}
        self.version = 0

    def items(self) -> Iterable[tuple[str, Registration]]:
        return self._entries.items()

    def register(self, caller: str, env: str, channels: Sequence[ChannelSpec]) -> None:
        """Replace the caller's registration, empty channels clears it."""
        if not caller:
            raise ValueError("lockstep register: caller must be nonempty")
        if channels:
            self._entries[caller] = Registration(env=env, channels=tuple(channels))
        else:
            self._entries.pop(caller, None)
        self.version += 1

    def drop_env(self, env: str) -> list[str]:
        """Drop every registration owned by a despawned env, returning the dropped callers."""
        dropped = [caller for caller, reg in self._entries.items() if reg.env and env_match(reg.env, env)]
        for caller in dropped:
            del self._entries[caller]
        if dropped:
            self.version += 1
        return dropped

    def desired(self, env_ids: Iterable[int]) -> dict[str, tuple[ChannelSpec, str]]:
        """Union of all registrations keyed by resolved topic, first registrant wins."""
        ids = list(env_ids)
        desired: dict[str, tuple[ChannelSpec, str]] = {}
        for reg in self._entries.values():
            for spec in reg.channels:
                if not reg.env and '{env}' in spec.topic_template:
                    for env_id in ids:
                        topic = spec.topic_template.replace('{env}', f'env_{env_id}')
                        desired.setdefault(topic, (spec, f"{spec.name}/env_{env_id}"))
                else:
                    desired.setdefault(spec.topic_template, (spec, spec.name))
        return desired


@dataclasses.dataclass
class Channel:
    spec: ChannelSpec
    name: str
    topic: str
    period: float
    next_due: float
    latest_stamp: float | None = None

    @property
    def hard(self) -> bool:
        return self.spec.hard


class GateLedger:
    def __init__(self, dt: float, base: float) -> None:
        self.dt = dt
        self.base = base
        self.now = 0.0
        self.channels: dict[str, Channel] = {}

    @property
    def tick(self) -> float:
        return self.base + self.now

    def rebase(self, sim_now: float) -> float:
        """Absorb sim time that advanced outside the run (windows, foreign holds), every
        window restarts from the new now. Returns the drift absorbed, 0.0 within half a step."""
        drift = (sim_now - self.base) - self.now
        if drift <= self.dt / 2.0:
            return 0.0
        self.now += drift
        for ch in self.channels.values():
            ch.next_due = self.now + ch.period
        return drift

    def refresh(self, desired: Mapping[str, tuple[ChannelSpec, str]]) -> tuple[list[Channel], list[Channel]]:
        """Reconcile against the desired set, returning (removed, added) for the subscription owner."""
        removed = [self.channels.pop(topic) for topic, ch in list(self.channels.items()) if desired.get(topic) != (ch.spec, ch.name)]
        added: list[Channel] = []
        for topic, (spec, display) in desired.items():
            if topic in self.channels:
                continue
            # raw period, floored to one sim step
            period = max(spec.period_s, self.dt)
            channel = Channel(spec=spec, name=display, topic=topic, period=period, next_due=self.now + period)
            self.channels[topic] = channel
            added.append(channel)
        return removed, added

    def next_delta(self) -> float:
        """Sim seconds to the earliest due channel, at least one step."""
        due = min(ch.next_due for ch in self.channels.values())
        delta = due - self.now
        return self.dt if delta <= 0.0 else delta

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def due(self) -> list[Channel]:
        return [ch for ch in self.channels.values() if ch.next_due <= self.now + 1e-9]

    def covers(self, channel: Channel) -> bool:
        """Latest stamp within one period of the ledger's tick, at clock resolution."""
        if channel.latest_stamp is None:
            return False
        return round(channel.latest_stamp * _NS) > round((self.now - channel.period) * _NS)

    def observe(self, channel: Channel, stamp: float) -> None:
        channel.latest_stamp = stamp - self.base

    def waiting(self, channels: Iterable[Channel]) -> list[Channel]:
        return [ch for ch in channels if not self.covers(ch)]

    def complete(self, channels: Iterable[Channel]) -> None:
        for ch in channels:
            ch.next_due += ch.period
