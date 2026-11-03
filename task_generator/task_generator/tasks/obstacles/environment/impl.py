import math
import typing

import arena_simulation_setup.tree.configs.environment
import attrs
import numpy as np
import shapely
import shapely.ops
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_runtime.env.task_api import GridFrame, WorldOccupancy
from arena_runtime.shared import DynamicObstacle, Obstacle, Orientation, Pose, Position
from shapely.geometry import LineString, Point

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


@attrs.define()
class _ParsedConfig:
    static: list[Obstacle]
    dynamic: list[DynamicObstacle]


class TM_Environment(TM_Obstacles):
    _config: ROSParamT[_ParsedConfig]

    def merge_walls(self, door_threshold: float = 2.0) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        world = self._ctx.world_manager.world_compacted()
        all_walls = list(world.all_walls)
        horizontal, vertical = [], []

        # Classify walls
        for wall in all_walls:
            start = (wall.start.x, wall.start.y)
            end = (wall.end.x, wall.end.y)
            if start[1] == end[1]:  # Horizontal
                y = start[1]
                x1, x2 = sorted([start[0], end[0]])
                horizontal.append((y, x1, x2))
            elif start[0] == end[0]:  # Vertical
                x = start[0]
                y1, y2 = sorted([start[1], end[1]])
                vertical.append((x, y1, y2))

        # Merge horizontal walls
        merged_h = []
        walls_by_y = {}
        for y, x1, x2 in horizontal:
            walls_by_y.setdefault(y, []).append((x1, x2))
        for y, segments in walls_by_y.items():
            segments.sort()
            merged = []
            for seg in segments:
                if not merged:
                    merged.append(seg)
                else:
                    last = merged[-1]
                    if seg[0] - last[1] <= door_threshold:
                        merged[-1] = (last[0], max(last[1], seg[1]))
                    else:
                        merged.append(seg)
            merged_h.extend([(y, x1, x2) for x1, x2 in merged])

        # Merge vertical walls
        merged_v = []
        walls_by_x = {}
        for x, y1, y2 in vertical:
            walls_by_x.setdefault(x, []).append((y1, y2))
        for x, segments in walls_by_x.items():
            segments.sort()
            merged = []
            for seg in segments:
                if not merged:
                    merged.append(seg)
                else:
                    last = merged[-1]
                    if seg[0] - last[1] <= door_threshold:
                        merged[-1] = (last[0], max(last[1], seg[1]))
                    else:
                        merged.append(seg)
            merged_v.extend([(x, y1, y2) for y1, y2 in merged])

        return merged_h, merged_v

    def _create_rooms_from_walls(self, door_threshold: float = 4.0) -> list[shapely.Polygon]:
        merged_h, merged_v = self.merge_walls(door_threshold)
        diagonal = [wall for wall in self._ctx.world_manager.world_compacted().all_walls if wall.start.x != wall.end.x and wall.start.y != wall.end.y]
        lines = [
            *(LineString([(x1, y), (x2, y)]) for y, x1, x2 in merged_h),
            *(LineString([(x, y1), (x, y2)]) for x, y1, y2 in merged_v),
            *(LineString([(wall.start.x, wall.start.y), (wall.end.x, wall.end.y)]) for wall in diagonal),
        ]
        return list(shapely.ops.polygonize(shapely.ops.unary_union(lines)))

    @staticmethod
    def _footprint(frame: GridFrame, world_x: float, world_y: float, width: float, height: float, margin: float, rotation_deg: float) -> tuple[int, int, int, int]:
        """Half-open cell rows and cols covered by the rotated box's axis-aligned bounds grown by margin."""
        radians = math.radians(rotation_deg)
        cos, sin = abs(math.cos(radians)), abs(math.sin(radians))
        half_w = (width * cos + height * sin + margin) / 2.0
        half_h = (width * sin + height * cos + margin) / 2.0
        top, left = frame.pos2grid(Position(x=world_x - half_w, y=world_y + half_h))
        bottom, right = frame.pos2grid(Position(x=world_x + half_w, y=world_y - half_h))
        return int(top), int(bottom), int(left), int(right)

    def _is_region_free(
        self,
        occupancy_grid: np.ndarray,
        frame: GridFrame,
        world_x: float,
        world_y: float,
        width: float,
        height: float,
        rotation_deg: float = 0.0,
    ) -> bool:
        top, bottom, left, right = self._footprint(frame, world_x, world_y, width, height, 0.0, rotation_deg)
        if top < 0 or left < 0 or bottom > occupancy_grid.shape[0] or right > occupancy_grid.shape[1]:
            return False
        return bool(WorldOccupancy.empty(occupancy_grid[top:bottom, left:right]).all())

    def _mark_region_occupied(
        self,
        occupancy_grid: np.ndarray,
        frame: GridFrame,
        world_x: float,
        world_y: float,
        width: float,
        height: float,
        margin: float = 0.0,
        rotation_deg: float = 0.0,
    ) -> None:
        top, bottom, left, right = self._footprint(frame, world_x, world_y, width, height, margin, rotation_deg)
        occupancy_grid[max(top, 0) : max(bottom, 0), max(left, 0) : max(right, 0)] = WorldOccupancy.FULL

    def _parse_environment(self, environment_file: str) -> _ParsedConfig:
        rng = self._ctx.conf.RNG.stream("obstacles", "environment")

        environment = arena_simulation_setup.tree.configs.environment.EnvironmentIdentifier(environment_file).resolve_sync()

        static_obstacles: list[Obstacle] = []
        dynamic_obstacles: list[DynamicObstacle] = []

        world = self._ctx.world_manager.world_compacted()

        if zones := world.zones:
            rooms = [zone.shape() for zone in zones]
            zone_name_to_polygon = {zone.name: zone.shape() for zone in zones}
        else:
            rooms = self._create_rooms_from_walls()
            zone_name_to_polygon = {}

        groups = dict(environment)["groups"]

        group_room_pairs: list[tuple[dict, shapely.Polygon]] = []
        for room_polygon in rooms:
            eligible_groups = [g for g in groups if "zones" not in g or any(zone_name_to_polygon.get(z, shapely.Polygon()).equals(room_polygon) for z in g["zones"])]
            if eligible_groups:
                eligible_groups_sorted = sorted(eligible_groups, key=lambda g: g["name"])
                group_room_pairs.append((eligible_groups_sorted[rng.integers(len(eligible_groups_sorted))], room_polygon))

        world_map = self._ctx.world_manager.map_for_floor()
        frame = world_map.frame
        occupancy_grid = world_map.occupancy.grid.copy()

        for idx, (group, room_polygon) in enumerate(group_room_pairs):
            group_name = group["name"]
            width, height = group["size"]
            margin = group.get("margin", 0.5)

            n_groups = 0
            minx, miny, max_x, max_y = room_polygon.bounds

            step_x = width + margin
            step_y = height + margin
            y = miny + step_y / 2.0

            while y + step_y / 2.0 <= max_y:
                x = minx + step_x / 2.0
                while x + step_x / 2.0 <= max_x:
                    if room_polygon.contains(Point(x, y)):
                        rotation_deg = rng.choice(group.get("rotations", [0]))
                        if self._is_region_free(occupancy_grid, frame, x, y, width, height, rotation_deg=rotation_deg):
                            group_static_entities = group.get("entities", {}).get("static", [])
                            group_dynamic_entities = group.get("entities", {}).get("dynamic", [])
                            for j, entity in enumerate(group_static_entities):
                                ex_off, ey_off, e_theta = entity["position"]

                                radians = math.radians(rotation_deg)
                                rot_x = ex_off * math.cos(radians) - ey_off * math.sin(radians)
                                rot_y = ex_off * math.sin(radians) + ey_off * math.cos(radians)
                                rot_theta = math.fmod(e_theta + rotation_deg, 360) / 180 * math.pi

                                obstacle_x = x + rot_x
                                obstacle_y = y + rot_y

                                obs_name = f"G_{group_name}_{idx}_{n_groups}_{entity['model']}_{j}"
                                new_obstacle = Obstacle(
                                    name=obs_name,
                                    pose=Pose(
                                        Position(x=obstacle_x, y=obstacle_y),
                                        Orientation.from_yaw(rot_theta),
                                    ),
                                    model=entity["model"],
                                    extra={},
                                )
                                static_obstacles.append(new_obstacle)
                            for g, entity in enumerate(group_dynamic_entities):
                                ex_off, ey_off, e_theta = entity["position"]
                                radians = math.radians(rotation_deg)
                                rot_x = ex_off * math.cos(radians) - ey_off * math.sin(radians)
                                rot_y = ex_off * math.sin(radians) + ey_off * math.cos(radians)
                                rot_theta = math.fmod(e_theta + rotation_deg, 360) / 180 * math.pi

                                obstacle_x = x + rot_x
                                obstacle_y = y + rot_y

                                # strings are zone refs, lists are coordinates
                                resolver = world.point_resolver(rng)
                                resolved_waypoints = [resolver.resolve(wp) if isinstance(wp, str) else Position(x=wp[0], y=wp[1]) for wp in entity.get("waypoints", [])]

                                obs_name = f"G_{group_name}_{idx}_{n_groups}_{entity['model']}_{g}"
                                new_obstacle = DynamicObstacle(
                                    name=obs_name,
                                    pose=Pose(
                                        Position(x=obstacle_x, y=obstacle_y),
                                        Orientation.from_yaw(rot_theta),
                                    ),
                                    model=entity["model"],
                                    waypoints=resolved_waypoints,
                                    extra={},
                                )
                                dynamic_obstacles.append(new_obstacle)

                            self._mark_region_occupied(occupancy_grid, frame, x, y, width, height, margin=margin, rotation_deg=rotation_deg)

                            n_groups += 1
                    x += step_x
                y += step_y
        return _ParsedConfig(static=static_obstacles, dynamic=dynamic_obstacles)

    async def reset(self, *, seed: int) -> Obstacles:
        return self._config.value.static, self._config.value.dynamic

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        self._config = self.node.ROSParam[_ParsedConfig](
            self.namespace('file'),
            'default',
            parse=self._parse_environment,
        )
