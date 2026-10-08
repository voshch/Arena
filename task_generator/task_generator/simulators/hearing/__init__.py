"""Robot hearing axis: `robot.hearing:=<frontend>` selects the hearing layer run for every fleet robot."""

from __future__ import annotations

import abc
import os

from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry
from arena_rclpy_mixins.shared import Namespace
from arena_runtime._node import NodeInterface
from task_generator_msgs.msg import AdapterDisplay, RecordedTopic

NONE = "none"
FRONTENDS = ("bus", "srp", "seld")


def arena_hearing_share(hearing: str) -> str:
    """Share directory of arena_hearing, a RuntimeError naming the feature when it is not installed."""
    from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

    try:
        return get_package_share_directory("arena_hearing")
    except PackageNotFoundError as exc:
        raise RuntimeError(f"robot.hearing:={hearing} needs the arena_hearing package, install it with `arena feature hearing install`") from exc


def nav2_overlay(hearing: str) -> str:
    """Nav2 params overlay of the hearing layer, empty for none."""
    if hearing == NONE:
        return ""
    return os.path.join(arena_hearing_share(hearing), "config", "nav2_overlay.yaml")


class BaseHearing(NodeInterface, abc.ABC):
    """Node-side counterpart of the robot hearing layer."""

    def __init__(self, *args: object, namespace: Namespace, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._namespace = namespace

    def robot_displays(self, robot: str) -> tuple[AdapterDisplay, ...]:
        """Per-robot rviz displays, published as the robot's `_hearing` adapter entry."""
        del robot
        return ()

    def recorded_topics(self) -> tuple[RecordedTopic, ...]:
        """Topics the episode recorder subscribes to for this layer."""
        return ()


HearingRegistry = Registry[str, BaseHearing]()


@HearingRegistry.register(NONE)
async def none(**kwargs: object) -> BaseHearing:
    from .noop import NoopHearing

    return NoopHearing(**kwargs)


async def _arena(**kwargs: object) -> BaseHearing:
    from .arena import ArenaHearing

    return ArenaHearing(**kwargs)


for _frontend in FRONTENDS:
    HearingRegistry.register(_frontend)(_arena)
