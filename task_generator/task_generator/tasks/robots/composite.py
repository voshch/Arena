"""Composite + Null TM_Robots: multi-TM fan-out and idle sink."""

from __future__ import annotations

import asyncio
import typing

import attrs
from arena_runtime.env.task_layer import TaskContext
from arena_runtime.shared import Pose

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.robots import TM_Robots


@attrs.define
class _ScopedRobotsView:
    """Thin proxy exposing only a subset of a parent's robots."""

    _inner: typing.Any  # real RobotsManager
    _names: frozenset[str]

    def __getattr__(self, item: str) -> object:
        return getattr(self._inner, item)

    @property
    def managers(self) -> dict:
        all_mgrs = self._inner.managers
        return {n: all_mgrs[n] for n in self._names if n in all_mgrs}


def _scoped_ctx(parent: TaskContext, robot_names: typing.Iterable[str]) -> TaskContext:
    """Build a TaskContext whose robots expose only the named subset."""
    names = frozenset(robot_names)
    scoped_mgr = _ScopedRobotsView(parent.robots_manager, names)
    return attrs.evolve(parent, robots_manager=scoped_mgr)  # type: ignore[arg-type]


class TM_Composite(TM_Robots):
    """Task mode that fans out to multiple sub-TMs, each scoped to a disjoint robot subset."""

    _sub_modes: list[TM_Robots]

    def __init__(self, *, sub_modes: typing.Sequence[TM_Robots], **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        super().__init__(**kwargs)
        self._sub_modes = list(sub_modes)

    async def reset(self) -> None:
        await super().reset()
        await asyncio.gather(*(m.reset() for m in self._sub_modes))

    async def teardown(self) -> None:
        await asyncio.gather(*(m.teardown() for m in self._sub_modes))

    @property
    def start_poses(self) -> dict:
        merged: dict = {}
        for m in self._sub_modes:
            merged.update(m.start_poses)
        return merged

    @property
    async def done(self) -> bool:
        if not self._sub_modes:
            return False
        results = await asyncio.gather(*(m.done for m in self._sub_modes))
        return all(results)

    def goal_editing_robots(self) -> frozenset[str]:
        return frozenset().union(*(m.goal_editing_robots() for m in self._sub_modes))

    async def set_position(self, pose: Pose):
        robots = self._ctx.robots
        if not robots:
            return
        last = next(reversed(robots))
        for m in self._sub_modes:
            if last in m._ctx.robots:
                await m.set_position(pose)
                return

    async def set_goal(self, pose: Pose):
        for m in self._sub_modes:
            await m.set_goal(pose)


class TM_Null(TM_Robots):
    """Idle sink for unallocated robots."""

    async def reset(self) -> None:
        await super().reset()

    @property
    async def done(self) -> bool:
        return False

    async def set_position(self, pose: Pose):
        return None

    async def set_goal(self, pose: Pose):
        return None


# TM_Null is registered here via a sentinel string because the
# standard registry is keyed by the Modes.TM_Robots enum.
_COMPOSITE_EXTRA: dict[str, typing.Any] = {}


def _register_extra(kind_str: str, loader: typing.Callable[[], type[TM_Robots]]) -> None:
    _COMPOSITE_EXTRA[kind_str] = loader


def get_extra_tm_loader(kind_str: str) -> typing.Callable[[], type[TM_Robots]] | None:
    """Look up a TM loader registered by sentinel string (currently ``null``)."""
    return _COMPOSITE_EXTRA.get(kind_str)


def extra_tm_kinds() -> list[str]:
    """Sentinel strings with a registered TM loader, sorted."""
    return sorted(_COMPOSITE_EXTRA)


_register_extra("null", lambda: TM_Null)


__all__ = [
    "TM_Composite",
    "TM_Null",
    "_scoped_ctx",
    "extra_tm_kinds",
    "get_extra_tm_loader",
]
