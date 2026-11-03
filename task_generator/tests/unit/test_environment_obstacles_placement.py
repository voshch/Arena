from __future__ import annotations

import asyncio
from types import SimpleNamespace
import math
import typing

import numpy as np
import pytest
import shapely
import yaml

OFFICE = [(0.0, 0.0), (6.0, 0.0), (6.0, 4.0), (0.0, 4.0)]
LAB = [(10.0, 0.0), (16.0, 0.0), (16.0, 4.0), (10.0, 4.0)]
OFFICE_TILES = [(1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (1.0, 3.0), (3.0, 3.0), (5.0, 3.0)]
OUTER_BOX = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0), (0.0, 0.0)]
LEFT_HALF = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0), (10.0, 10.0)]
RIGHT_HALF = [(10.0, 0.0), (10.0, 10.0), (20.0, 0.0), (20.0, 10.0)]
BOX_GROUP = {"name": "box", "size": [1, 1], "margin": 1, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}


class Bench:
    def __init__(self, node, ctx, tmp_path) -> None:
        self.node = node
        self.ctx = ctx
        self.tmp_path = tmp_path
        self.configs = 0

    def world(
        self,
        *,
        zones: dict[str, list[tuple[float, float]]] | None = None,
        walls: typing.Sequence[typing.Sequence[tuple[float, float]]] = (),
        shape: tuple[int, int] = (60, 60),
        origin: tuple[float, float] = (0.0, 0.0),
        resolution: float = 0.5,
        occupied: typing.Sequence[tuple[slice, slice]] = (),
    ) -> None:
        from arena_rclpy_mixins.Time import Time
        from arena_simulation_setup.tree.World import Level, World, Zone
        from arena_simulation_setup.tree.World.model import WallPath

        from arena_runtime.env.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
        from arena_runtime.shared import Position

        level = Level(
            zones=[Zone(name=name, polygon=[Position(x, y) for x, y in polygon]) for name, polygon in (zones or {}).items()],
            walls=[WallPath(path=[Position(x, y) for x, y in path]) for path in walls],
        )
        grid = np.full(shape, WorldOccupancy.EMPTY, dtype=np.uint8)
        for cells in occupied:
            grid[cells] = WorldOccupancy.FULL
        world_map = WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(x=origin[0], y=origin[1]), resolution=resolution, time=Time())
        self.ctx.world_manager.update_world(world_map, World.from_levels(level))

    def mode(self, groups: list[dict] | None = None):
        from task_generator.modes import Modes
        from task_generator.tasks.registry import OBSTACLES_MODES

        key = Modes.TM_Obstacles.ENVIRONMENT
        tm = OBSTACLES_MODES.get(key)(node=self.node, ctx=self.ctx, namespace=OBSTACLES_MODES.meta(key).namespace)
        if groups is not None:
            self.load(tm, groups)
        return tm

    def load(self, tm, groups: list[dict]) -> None:
        from rclpy.parameter import Parameter

        self.configs += 1
        path = self.tmp_path / f"environment_{self.configs}.yaml"
        path.write_text(yaml.safe_dump({"groups": groups}))
        assert self.node.set_parameters([Parameter(tm._config.name, value=str(path))])[0].successful


@pytest.fixture
def on_bench(tmp_path):
    def run(body: typing.Callable[[Bench], typing.Awaitable[None]]) -> None:
        from arena_runtime.sim.dummy_simulator import DummySimulator

        from arena_runtime.env.constants import Constants

        from arena_runtime.env.environment_manager import EnvironmentManager
        from arena_runtime.env.realizer import Realizer
        from arena_runtime.env.robot_manager import RobotsManager
        from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS
        from arena_runtime.env.episode_node import EpisodeNode
        from arena_runtime.acoustics import AcousticsSimulatorRegistry
        from arena_runtime.humans.noop import NoopHumanSimulator
        from arena_runtime.env.task_layer import TaskConfiguration, TaskContext

        async def main() -> None:
            node = EpisodeNode()
            try:
                realizer = Realizer(level_origins={"0": (0.0, 0.0)})
                sim = DummySimulator(node=node, namespace=node._namespace, realizer=realizer)
                acoustics = await AcousticsSimulatorRegistry.get(Constants.AcousticsSimulator.NONE, node=node, namespace=node._namespace)
                human = NoopHumanSimulator(node=node, namespace=node._namespace, simulator=sim, realizer=realizer, acoustics=acoustics)
                env = EnvironmentManager(node=node, simulator=sim, human_simulator=human, realizer=realizer)
                world = WorldManagerROS(node=node, environment_manager=env)
                ctx = TaskContext(
                    environment_manager=env,
                    robots_manager=RobotsManager(node=node, environment_manager=env),
                    world_manager=world,
                    realizer=realizer,
                    simulator=sim,
                    acoustics_simulator=acoustics,
                    reset_lock=asyncio.Lock(),
                    conf=TaskConfiguration.of(node.conf),
                    episode=SimpleNamespace(),
                    interactive=SimpleNamespace(),
                )
                await body(Bench(node, ctx, tmp_path))
            finally:
                node._heartbeat_stop.set()
                node.destroy_node()

        asyncio.run(main())

    return run


def _xy(entity) -> tuple[float, float]:
    return entity.pose.position.x, entity.pose.position.y


def _heading(entity) -> tuple[float, float]:
    yaw = entity.pose.orientation.to_yaw()
    return math.cos(yaw), math.sin(yaw)


def _corners(room) -> list[tuple[float, float]]:
    return sorted(room.exterior.coords[:-1])


def _inside(polygon, x: float, y: float) -> bool:
    return shapely.Polygon(polygon).covers(shapely.Point(x, y))


def test_group_tiles_a_zone_on_a_size_plus_margin_grid_with_rotated_offsets(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "desk", "size": [1, 1], "margin": 1, "rotations": [90], "entities": {"static": [{"position": [1.0, 0.0, 90.0], "model": "chair"}, {"position": [0.0, 0.0, 0.0], "model": "table"}]}}])
        static, dynamic = await tm.reset(seed=0)
        assert dynamic == []
        assert [entity.name for entity in static] == [name for n in range(6) for name in (f"G_desk_0_{n}_chair_0", f"G_desk_0_{n}_table_1")]
        assert [entity.model.name for entity in static] == ["chair", "table"] * 6
        chairs, tables = static[0::2], static[1::2]
        assert [_xy(table) for table in tables] == pytest.approx(OFFICE_TILES)
        assert [_xy(chair) for chair in chairs] == pytest.approx([(x, y + 1.0) for x, y in OFFICE_TILES])
        assert all(_heading(chair) == pytest.approx((-1.0, 0.0), abs=1e-9) for chair in chairs)
        assert all(_heading(table) == pytest.approx((0.0, 1.0), abs=1e-9) for table in tables)

    on_bench(body)


def test_group_without_margin_or_rotations_defaults_to_half_meter_and_no_turn(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "crate", "size": [2.5, 1.5], "entities": {"static": [{"position": [0.0, 0.5, 0.0], "model": "box"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([(1.5, 1.5), (4.5, 1.5), (1.5, 3.5), (4.5, 3.5)])
        assert all(_heading(entity) == pytest.approx((1.0, 0.0), abs=1e-9) for entity in static)

    on_bench(body)


def test_dynamic_entities_resolve_zone_and_coordinate_waypoints(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "walk", "size": [4, 2], "margin": 0, "rotations": [0], "entities": {"dynamic": [{"position": [0.5, 0.0, 0.0], "model": "human", "waypoints": ["office", [2.0, 3.0]]}]}}])
        static, dynamic = await tm.reset(seed=0)
        assert static == []
        assert [entity.name for entity in dynamic] == ["G_walk_0_0_human_0", "G_walk_0_1_human_0"]
        assert [_xy(entity) for entity in dynamic] == pytest.approx([(2.5, 1.0), (2.5, 3.0)])
        for entity in dynamic:
            sampled, fixed = entity.waypoints
            assert _inside(OFFICE, sampled.x, sampled.y)
            assert (fixed.x, fixed.y) == (2.0, 3.0)

    on_bench(body)


def test_dynamic_entities_without_waypoints_stand_still(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([{"name": "idle", "size": [4, 2], "margin": 0, "entities": {"dynamic": [{"position": [0.0, 0.0, 0.0], "model": "human"}]}}])
        _, dynamic = await tm.reset(seed=0)
        assert [entity.waypoints for entity in dynamic] == [[], []]

    on_bench(body)


def test_zone_restricted_group_only_lands_in_its_zones(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE, "lab": LAB})
        tm = bench.mode([{"name": "bench", "size": [1, 1], "margin": 1, "zones": ["lab"], "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "stool"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([(x + 10.0, y) for x, y in OFFICE_TILES])
        assert {entity.name.split("_")[2] for entity in static} == {"0"}

    on_bench(body)


def test_tiles_outside_a_concave_zone_are_skipped(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"ell": [(0.0, 0.0), (6.0, 0.0), (6.0, 2.0), (2.0, 2.0), (2.0, 6.0), (0.0, 6.0)]})
        tm = bench.mode([{"name": "box", "size": [1, 1], "margin": 1, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([(1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (1.0, 3.0), (1.0, 5.0)])

    on_bench(body)


def test_without_zones_a_box_inside_the_walls_is_a_room_and_a_hole_in_the_outer_room(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[OUTER_BOX, [(2.0, 2.0), (6.0, 2.0), (6.0, 5.0), (2.0, 5.0), (2.0, 2.0)]])
        tm = bench.mode([BOX_GROUP])
        inner, outer = sorted(tm._create_rooms_from_walls(), key=lambda room: room.area)
        assert inner.equals(shapely.box(2.0, 2.0, 6.0, 5.0))
        assert outer.equals(shapely.box(0.0, 0.0, 20.0, 10.0).difference(inner))
        static, _ = await tm.reset(seed=0)
        by_room: dict[str, list[tuple[float, float]]] = {}
        for entity in static:
            by_room.setdefault(entity.name.split("_")[2], []).append(_xy(entity))
        small, large = sorted(by_room.values(), key=len)
        assert small == pytest.approx([(3.0, 3.0), (5.0, 3.0)])
        assert sorted(large) == pytest.approx(sorted((x, y) for x in range(1, 20, 2) for y in range(1, 10, 2) if not inner.covers(shapely.Point(x, y))))
        assert len(large) == 46

    on_bench(body)


def test_reloading_the_file_parameter_replaces_the_obstacles(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        tm = bench.mode([BOX_GROUP])
        assert len((await tm.reset(seed=0))[0]) == 6
        bench.load(tm, [{"name": "big", "size": [4, 2], "margin": 0, "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "box"}]}}])
        assert [_xy(entity) for entity in (await tm.reset(seed=1))[0]] == pytest.approx([(2.0, 1.0), (2.0, 3.0)])

    on_bench(body)


def test_merge_walls_joins_collinear_segments_across_door_gaps(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[[(0.0, 0.0), (3.0, 0.0)], [(8.0, 0.0), (4.0, 0.0)], [(20.0, 0.0), (25.0, 0.0)], [(0.0, 0.0), (0.0, 2.0)], [(0.0, 9.0), (0.0, 5.0)], [(0.0, 0.0), (3.0, 3.0)]])
        tm = bench.mode()
        assert tm.merge_walls(2.0) == ([(0.0, 0.0, 8.0), (0.0, 20.0, 25.0)], [(0.0, 0.0, 2.0), (0.0, 5.0, 9.0)])
        assert tm.merge_walls(4.0) == ([(0.0, 0.0, 8.0), (0.0, 20.0, 25.0)], [(0.0, 0.0, 9.0)])
        assert tm.merge_walls() == tm.merge_walls(2.0)

    on_bench(body)


def test_parsing_an_environment_leaves_the_world_map_untouched(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE})
        before = bench.ctx.world_manager.map_for_floor().occupancy.grid.copy()
        tm = bench.mode([BOX_GROUP])
        assert len((await tm.reset(seed=0))[0]) == 6
        after = bench.ctx.world_manager.map_for_floor().occupancy.grid
        assert sorted({int(value) for value in after[after != before]}) == []

    on_bench(body)


def test_wall_rooms_split_by_a_divider_are_both_found(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[OUTER_BOX, [(10.0, 0.0), (10.0, 10.0)]])
        rooms = bench.mode()._create_rooms_from_walls()
        assert sorted(_corners(room) for room in rooms) == [LEFT_HALF, RIGHT_HALF]

    on_bench(body)


def test_a_plain_wall_box_is_one_room(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[OUTER_BOX])
        tm = bench.mode([BOX_GROUP])
        assert [_corners(room) for room in tm._create_rooms_from_walls()] == [[(0.0, 0.0), (0.0, 10.0), (20.0, 0.0), (20.0, 10.0)]]
        static, _ = await tm.reset(seed=0)
        assert sorted(_xy(entity) for entity in static) == pytest.approx(sorted((x, y) for x in range(1, 20, 2) for y in range(1, 10, 2)))

    on_bench(body)


def test_wall_rooms_close_across_door_gaps_up_to_the_threshold(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[[(4.0, 0.0), (0.0, 0.0), (0.0, 10.0), (20.0, 10.0), (20.0, 0.0), (5.0, 0.0)], [(10.0, 0.0), (10.0, 4.0)], [(10.0, 5.0), (10.0, 10.0)]])
        assert sorted(_corners(room) for room in bench.mode()._create_rooms_from_walls()) == [LEFT_HALF, RIGHT_HALF]
        bench.world(walls=[[(4.0, 0.0), (0.0, 0.0), (0.0, 10.0), (20.0, 10.0), (20.0, 0.0), (9.0, 0.0)], [(10.0, 0.0), (10.0, 10.0)]])
        assert [_corners(room) for room in bench.mode()._create_rooms_from_walls()] == [RIGHT_HALF]

    on_bench(body)


def test_non_axis_aligned_walls_bound_rooms(on_bench):
    async def body(bench: Bench):
        bench.world(walls=[[(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (0.0, 0.0)]])
        assert [_corners(room) for room in bench.mode()._create_rooms_from_walls()] == [[(0.0, 0.0), (0.0, 4.0), (4.0, 0.0)]]

    on_bench(body)


def test_tiles_whose_group_footprint_is_occupied_in_the_map_are_skipped(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE}, occupied=[np.s_[57:59, 5:7]])
        static, _ = await bench.mode([BOX_GROUP]).reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([tile for tile in OFFICE_TILES if tile != (3.0, 1.0)])

    on_bench(body)


def test_an_occupied_cell_in_the_margin_alone_does_not_block_a_tile(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE}, occupied=[np.s_[57:58, 4:5]])
        static, _ = await bench.mode([BOX_GROUP]).reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx(OFFICE_TILES)

    on_bench(body)


def test_map_cells_follow_the_map_resolution_origin_and_row_flip(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"office": OFFICE}, shape=(40, 48), origin=(-2.0, -1.0), resolution=0.25, occupied=[np.s_[23:24, 20:21]])
        static, _ = await bench.mode([BOX_GROUP]).reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx([tile for tile in OFFICE_TILES if tile != (3.0, 3.0)])

    on_bench(body)


def test_rotated_groups_are_checked_against_their_turned_footprint(on_bench):
    async def body(bench: Bench):
        bench.world(zones={"lab": LAB})
        tm = bench.mode([{"name": "bar", "size": [1, 3], "margin": 1, "rotations": [90], "entities": {"static": [{"position": [0.0, 0.0, 0.0], "model": "bar"}]}}])
        static, _ = await tm.reset(seed=0)
        assert [entity.name for entity in static] == ["G_bar_0_0_bar_0", "G_bar_0_1_bar_0"]
        assert [_xy(entity) for entity in static] == pytest.approx([(11.0, 2.0), (15.0, 2.0)])

    on_bench(body)


def test_placed_groups_block_later_rooms_without_touching_the_map(on_bench):
    async def body(bench: Bench):
        from arena_runtime.env.task_api import WorldOccupancy

        bench.world(zones={"office": OFFICE, "office_copy": OFFICE})
        static, _ = await bench.mode([BOX_GROUP]).reset(seed=0)
        assert [_xy(entity) for entity in static] == pytest.approx(OFFICE_TILES)
        assert {entity.name.split("_")[2] for entity in static} == {"0"}
        assert bool(WorldOccupancy.empty(bench.ctx.world_manager.map_for_floor().occupancy.grid).all())

    on_bench(body)


def test_region_checks_and_marks_use_the_grid_frame(on_bench):
    async def body(bench: Bench):
        from arena_runtime.env.task_api import GridFrame, WorldOccupancy
        from arena_runtime.shared import Position

        bench.world()
        tm = bench.mode()
        frame = GridFrame(shape=(8, 12), origin=Position(x=-1.0, y=2.0), resolution=0.5)
        grid = np.full((8, 12), WorldOccupancy.EMPTY, dtype=np.uint8)
        assert tm._is_region_free(grid, frame, 1.0, 4.0, 1.0, 1.0)
        assert not tm._is_region_free(grid, frame, -1.0, 4.0, 1.0, 1.0)
        assert not tm._is_region_free(grid, frame, 1.0, 2.2, 1.0, 1.0)
        assert not tm._is_region_free(grid, frame, 4.8, 4.0, 1.0, 1.0)
        assert not tm._is_region_free(grid, frame, 1.0, 5.8, 1.0, 1.0)
        tm._mark_region_occupied(grid, frame, 1.0, 4.0, 2.0, 1.0, margin=1.0, rotation_deg=90.0)
        expected = np.zeros((8, 12), dtype=bool)
        expected[1:7, 2:6] = True
        assert WorldOccupancy.full(grid).tolist() == expected.tolist()
        assert not tm._is_region_free(grid, frame, 1.0, 4.0, 1.0, 1.0)
        assert tm._is_region_free(grid, frame, 3.0, 4.0, 1.0, 1.0)
        tm._mark_region_occupied(grid, frame, -1.0, 2.0, 1.0, 1.0)
        expected[7, 0] = True
        assert WorldOccupancy.full(grid).tolist() == expected.tolist()

    on_bench(body)
