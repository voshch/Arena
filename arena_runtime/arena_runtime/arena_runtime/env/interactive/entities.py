from __future__ import annotations

import typing

import attrs
from geometry_msgs.msg import Pose as PoseMsg
from visualization_msgs.msg import Marker

from arena_runtime.env.interactive.hub import Apply, planar_marker, visual
from arena_runtime.shared import Obstacle, Pose

if typing.TYPE_CHECKING:
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.interactive.hub import MarkerHub
    from arena_runtime.env.realizer import Realizer

_STATIC_COLOR = (0.2, 0.6, 0.9, 0.8)


class EntityHandles:
    """Interactive handles of runtime-spawned static obstacles, named by sim_path."""

    def __init__(self, markers: MarkerHub, realizer: Realizer, environment: EnvironmentManager) -> None:
        self._markers = markers
        self._realizer = realizer
        self._environment = environment
        self._obstacles: dict[str, Obstacle] = {}

    def add(self, obstacle: Obstacle) -> None:
        name = obstacle.sim_path
        self._obstacles[name] = obstacle
        pose = self._realizer.realize(obstacle.pose, self._level(obstacle)).to_msg()
        body = visual(Marker.CYLINDER, (0.6, 0.6, 0.02), _STATIC_COLOR)

        async def on_pose(moved: PoseMsg) -> None:
            await self._move(name, moved)

        self._markers.put(
            planar_marker(name, pose, description=obstacle.name, visuals=[body]),
            on_pose=on_pose,
            apply=Apply.LIVE,
        )

    def clear(self) -> None:
        self._obstacles.clear()

    def _level(self, obstacle: Obstacle) -> str:
        return self._realizer.placement_level(obstacle.placement_zone, obstacle.level)

    async def _move(self, name: str, pose_msg: PoseMsg) -> None:
        obstacle = self._obstacles[name]
        local = self._realizer.ezilear(Pose.from_msg(pose_msg), self._level(obstacle))
        moved = attrs.evolve(obstacle, pose=local)
        await self._environment.move_obstacles([moved])
        self._obstacles[name] = moved
