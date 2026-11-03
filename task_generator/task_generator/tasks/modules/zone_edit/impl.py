import asyncio
from collections.abc import Awaitable, Callable

import attrs
import geometry_msgs.msg as geometry_msgs
from arena_runtime.env.task_api import Apply, Menu, planar_marker, visual
from arena_runtime.shared import Position
from visualization_msgs.msg import Marker

from task_generator.tasks.modules import TM_Module
from task_generator.tasks.modules.zone_edit.zone_corners import write_zone_polygons

_PREFIX = "zone/"
_HANDLE_SCALE = 0.6
_HANDLE_COLOR = (1.0, 0.85, 0.1, 1.0)
_HANDLE_Z = 0.05


class Mod_ZoneEdit(TM_Module):
    def after_reset(self):
        interactive = self._ctx.interactive
        assert interactive is not None, "zone_edit needs the env's interactive context"
        hub = interactive.markers
        manager = self._ctx.world_manager
        root = manager.loaded_world_path
        menu = [Menu(f"Save world to {root}", self._save)] if root.parts else []
        for level_id, level in manager.world.levels.items():
            offset = self._ctx.realizer.realize(Position(x=0.0, y=0.0), level_id)
            for zone in level.zones:
                for index, corner in enumerate(zone.polygon):
                    pose = geometry_msgs.Pose()
                    pose.position.x = corner.x + offset.x
                    pose.position.y = corner.y + offset.y
                    pose.position.z = _HANDLE_Z
                    pose.orientation.w = 1.0
                    hub.put(
                        planar_marker(
                            f"{_PREFIX}{level_id}/{zone.name}/{index}",
                            pose,
                            description=zone.name if index == 0 else "",
                            scale=_HANDLE_SCALE,
                            rotate=False,
                            visuals=[visual(Marker.SPHERE, (0.25, 0.25, 0.25), _HANDLE_COLOR)],
                        ),
                        on_pose=self._mover(level_id, zone.name, index),
                        apply=Apply.LIVE,
                        menu=menu,
                    )

    def _mover(self, level_id: str, zone_name: str, index: int) -> Callable[[geometry_msgs.Pose], Awaitable[None]]:
        async def move(pose: geometry_msgs.Pose) -> None:
            manager = self._ctx.world_manager
            level = manager.world.levels[level_id]
            zone = next(z for z in level.zones if z.name == zone_name)
            offset = self._ctx.realizer.realize(Position(x=0.0, y=0.0), level_id)
            zone.polygon[index] = attrs.evolve(zone.polygon[index], x=pose.position.x - offset.x, y=pose.position.y - offset.y)
            manager.refresh_world_markers(self._ctx.simulator.semantics_snapshot())

        return move

    async def _save(self) -> None:
        manager = self._ctx.world_manager
        root = manager.loaded_world_path
        polygons = {zone.name: [(c.x, c.y) for c in zone.polygon] for level in manager.world.all_levels for zone in level.zones}
        changed = await asyncio.to_thread(write_zone_polygons, root, polygons, self._ctx.conf.KG_DERIVE.value or "auto")
        self._logger.info(f"saved {changed} polygon coordinates to {root}")
