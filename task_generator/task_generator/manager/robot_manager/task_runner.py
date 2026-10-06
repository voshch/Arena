"""Episode-wide phase list of one robot and the judge that advances it, ROS-independent."""

from __future__ import annotations

import typing
from collections.abc import Sequence

import attrs
from arena_simulation_setup.shared.conditions import EpisodeCondition, OnFailure
from arena_simulation_setup.shared.judge import ConditionMonitor, PhaseMonitor, Sample, Zones
from arena_simulation_setup.shared.task import TaskPhase

Outcome = typing.Literal["pending", "met", "failed", "dropped"]


@attrs.define
class PhaseState:
    index: int
    phase: TaskPhase
    monitor: PhaseMonitor
    outcome: Outcome = "pending"
    reason: str = ""
    dispatched: bool = False


@attrs.define
class ScopedCondition:
    """A condition judged over a span of phases: a request's or the episode's."""

    id: str
    condition: EpisodeCondition
    monitor: ConditionMonitor
    first: int
    last: int | None = None

    def covers(self, index: int | None) -> bool:
        if index is None:
            return False
        return index >= self.first and (self.last is None or index <= self.last)


@attrs.define
class Transition:
    """What one tick decided: the phase that ended, how, and whether the task stops."""

    ended: PhaseState | None = None
    stop: bool = False
    abort: str | None = None
    violated: list[ScopedCondition] = attrs.field(factory=list)


@attrs.define
class TaskRunner:
    robot: str
    phases: list[PhaseState] = attrs.field(factory=list)
    active: int | None = None
    scoped: list[ScopedCondition] = attrs.field(factory=list)
    violated: list[str] = attrs.field(factory=list)
    changed: bool = False
    _requests: int = 0

    def begin_episode(self) -> None:
        self.phases.clear()
        self.active = None
        self.scoped.clear()
        self.violated.clear()
        self._requests = 0
        self.changed = True

    def set_episode_conditions(self, conditions: Sequence[EpisodeCondition]) -> None:
        self.scoped = [s for s in self.scoped if not s.id.startswith("e")]
        self.scoped.extend(ScopedCondition(id=f"e{i}", condition=c, monitor=ConditionMonitor(c, self.robot), first=0) for i, c in enumerate(conditions))

    def submit(self, phases: Sequence[TaskPhase], conditions: Sequence[EpisodeCondition] = ()) -> list[PhaseState]:
        """Append resolved phases, dropping whatever of the previous request had not finished."""
        for state in self.phases:
            if state.outcome == "pending":
                state.outcome = "dropped"
        for scoped in self.scoped:
            if scoped.last is None and not scoped.id.startswith("e"):
                scoped.last = len(self.phases) - 1
        first = len(self.phases)
        states = [PhaseState(index=first + i, phase=phase, monitor=PhaseMonitor(phase, self.robot)) for i, phase in enumerate(phases)]
        self.phases.extend(states)
        request = self._requests
        self._requests += 1
        self.scoped.extend(ScopedCondition(id=f"r{request}:{i}", condition=c, monitor=ConditionMonitor(c, self.robot), first=first) for i, c in enumerate(conditions))
        self.active = first if states else None
        self.changed = True
        return states

    def stop(self) -> None:
        for state in self.phases:
            if state.outcome == "pending":
                state.outcome = "dropped"
        self.active = None
        self.changed = True

    @property
    def active_state(self) -> PhaseState | None:
        return None if self.active is None else self.phases[self.active]

    @property
    def done(self) -> bool:
        return self.active is None

    @property
    def waiting(self) -> bool:
        state = self.active_state
        return state is not None and state.monitor.waiting

    def outcomes(self, outcome: Outcome) -> list[int]:
        return [s.index for s in self.phases if s.outcome == outcome]

    def step(self, sample: Sample, zones: Zones, action_done: bool | None, failure: tuple[int | None, str | None], signal: str | None = None) -> Transition:
        """Judge one sample. `failure` is the adapter's (status, reason) for the active phase, read when it ends, `signal` the robot's latched signal."""
        transition = Transition()
        state = self.active_state
        for scoped in self.scoped:
            if scoped.covers(self.active) and not scoped.monitor.decided and scoped.monitor.update(sample, zones) is False:
                self._violate(scoped, transition)
        if state is None:
            return transition
        met = state.monitor.step(sample, zones, action_done, signal)
        for i, monitor in enumerate(state.monitor.conditions):
            if monitor.decided and monitor.verdict is False:
                self._violate(ScopedCondition(id=f"{state.index}:{i}", condition=state.phase.conditions[i], monitor=monitor, first=state.index, last=state.index), transition)
        if transition.stop:
            return transition
        if state.monitor.misfire is not None:
            self._end_phase(state, True, state.monitor.misfire, transition)
            if not transition.stop:
                self._apply_policy("abort_episode", state.monitor.misfire, transition)
            return transition
        if not met:
            return transition
        status, reason = failure
        failed = status is not None and status != 0
        self._end_phase(state, failed, reason or "", transition)
        return transition

    def fail_active(self, reason: str) -> Transition:
        """End the active phase as failed without waiting for the judge."""
        transition = Transition()
        state = self.active_state
        if state is not None:
            self._end_phase(state, True, reason, transition)
        return transition

    def _end_phase(self, state: PhaseState, failed: bool, reason: str, transition: Transition) -> None:
        for i, verdict in enumerate(state.monitor.finish()):
            if verdict is False:
                scoped = ScopedCondition(id=f"{state.index}:{i}", condition=state.phase.conditions[i], monitor=state.monitor.conditions[i], first=state.index, last=state.index)
                self._violate(scoped, transition)
        state.outcome = "failed" if failed else "met"
        state.reason = reason if failed else ""
        transition.ended = state
        self.changed = True
        if failed and state.phase.on_failure != "continue":
            self._apply_policy(state.phase.on_failure, reason or f"phase {state.index} failed on robot {self.robot}", transition)
            return
        if transition.stop:
            return
        nxt = state.index + 1
        self.active = nxt if nxt < len(self.phases) and self.phases[nxt].outcome == "pending" else None
        if self.active is None:
            for scoped in self.scoped:
                if not scoped.id.startswith("e") and not scoped.monitor.decided:
                    if scoped.monitor.finish() is False:
                        self._violate(scoped, transition)

    def _violate(self, scoped: ScopedCondition, transition: Transition) -> None:
        if scoped.id in self.violated:
            return
        self.violated.append(scoped.id)
        transition.violated.append(scoped)
        self.changed = True
        if scoped.condition.on_failure != "continue":
            self._apply_policy(scoped.condition.on_failure, f"condition {scoped.id} ({scoped.condition.p}) violated on robot {self.robot}", transition)

    def _apply_policy(self, policy: OnFailure, reason: str, transition: Transition) -> None:
        self.stop()
        transition.stop = True
        if policy == "abort_episode":
            transition.abort = reason

    def finish_episode(self) -> None:
        """Close every open condition scope at episode end."""
        transition = Transition()
        for scoped in self.scoped:
            if not scoped.monitor.decided and scoped.monitor.finish() is False:
                self._violate(scoped, transition)

    def serialize(self) -> dict:
        return {
            "phases": [s.phase.serialize() for s in self.phases],
            "conditions": [{"id": s.id, "from": s.first, "to": s.last, **s.condition.serialize()} for s in self.scoped],
        }


__all__ = ["Outcome", "PhaseState", "ScopedCondition", "TaskRunner", "Transition"]
