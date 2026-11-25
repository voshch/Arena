"""Per-agent view of a task layer's events: who owns which task and what each agent is doing."""

from __future__ import annotations

from collections.abc import Mapping

import attrs

_FINISHED = ("met", "failed", "dropped")


@attrs.frozen
class AgentEntry:
    name: str
    task: str = ""
    state: str = "idle"
    zones: tuple[str, ...] = ()
    since: float = 0.0


@attrs.define
class TaskBoard:
    """Folds stamped task events into one entry per agent, `changed` set whenever an entry differs."""

    changed: bool = False
    _owner: dict[str, str] = attrs.Factory(dict)
    _status: dict[str, str] = attrs.Factory(dict)
    _zones: dict[str, tuple[str, ...]] = attrs.Factory(dict)
    _finished: dict[str, str] = attrs.Factory(dict)
    _entries: dict[str, AgentEntry] = attrs.Factory(dict)

    def reset(self) -> None:
        self.changed = self.changed or bool(self._entries)
        self._owner.clear()
        self._status.clear()
        self._zones.clear()
        self._finished.clear()
        self._entries.clear()

    def agents(self) -> list[AgentEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.name)

    def apply(self, stamped: Mapping[str, object]) -> None:
        kind = str(stamped["kind"])
        task = str(stamped["task"])
        robot = str(stamped["robot"])
        t = float(stamped["t"])  # pyright: ignore[reportArgumentType]
        detail = stamped["detail"]
        assert isinstance(detail, Mapping)
        touched: list[str] = []
        if kind == "allocated":
            assignments = detail.get("assignments", {})
            assert isinstance(assignments, Mapping)
            touched.extend(str(a["member"]) for a in assignments.values() if isinstance(a, Mapping) and a.get("member"))
        elif kind == "reassigned":
            touched.append(str(detail.get("from", "")))
            self._set(task, robot, "assigned")
        elif kind == "started":
            zones = detail.get("zones", ())
            assert isinstance(zones, (list, tuple))
            self._zones[task] = tuple(str(zone) for zone in zones)
            self._set(task, robot, "working")
        elif kind in _FINISHED:
            self._set(task, robot, kind)
            if robot:
                self._finished[robot] = task
        else:
            self._set(task, robot, self._status.get(task) or "assigned")
        touched.append(robot)
        for agent in dict.fromkeys(touched):
            if agent:
                self._refresh(agent, t)

    def _set(self, task: str, robot: str, status: str) -> None:
        if robot:
            self._owner[task] = robot
        self._status[task] = status

    def _current(self, agent: str) -> tuple[str, str]:
        owned = [task for task, owner in self._owner.items() if owner == agent]
        working = [task for task in owned if self._status[task] == "working"]
        if working:
            return working[-1], "working"
        assigned = [task for task in owned if self._status[task] == "assigned"]
        if assigned:
            return assigned[0], "assigned"
        finished = self._finished.get(agent, "")
        if finished in owned:
            return finished, self._status[finished]
        return "", "idle"

    def _refresh(self, agent: str, t: float) -> None:
        task, state = self._current(agent)
        zones = self._zones.get(task, ())
        previous = self._entries.get(agent)
        if previous is not None and (previous.task, previous.state, previous.zones) == (task, state, zones):
            return
        self._entries[agent] = AgentEntry(agent, task, state, zones, t)
        self.changed = True
