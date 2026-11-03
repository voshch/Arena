"""Tests for WorldManager.get_positions_on_map and its grid helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import shapely
    import shapely.affinity
    from arena_rclpy_mixins.Time import Time
    from arena_simulation_setup.shared import Elevator, Stairs
    from arena_simulation_setup.shared.entities import Obstacle
    from arena_simulation_setup.tree.World import Level, World, WorldView, Zone
    from arena_simulation_setup.utils.geometry import Pose
    from arena_simulation_setup.utils.geometry import Position as GeoPosition
    from arena_runtime.env.world_manager.utils import MultiLevelMap, WorldLayers, WorldMap, WorldOccupancy
    from arena_runtime.env.world_manager.world_manager import (
        WorldManager,
        _occupancy_to_available,
        _sample_grid_positions,
        _zone_mask,
    )
    from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS
    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.shared import Position, PositionRadius
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")


def empty_grid(h: int, w: int) -> np.ndarray:
    return np.full((h, w), WorldOccupancy.EMPTY, dtype=np.uint8)


def make_map(grid: np.ndarray, resolution: float = 0.05, origin: tuple[float, float] = (0.0, 0.0)) -> WorldMap:
    return WorldMap(
        occupancy=WorldLayers(walls=WorldOccupancy(grid.copy())),
        origin=Position(x=origin[0], y=origin[1]),
        resolution=resolution,
        time=Time(),
    )


def grid_distance_to_occupied(occupancy: np.ndarray, row: int, col: int) -> float:
    occ = np.argwhere(~WorldOccupancy.not_full(occupancy))
    if len(occ) == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(occ - np.array([row, col]), axis=1)))


class TestOccupancyToAvailable:
    def test_returned_indices_in_input_frame(self):
        grid = empty_grid(20, 20)
        for sd in (1.0, 3.0, 5.0):
            avail = _occupancy_to_available(grid, safe_dist_cells=sd)
            assert avail.size > 0
            assert int(avail[:, 0].max()) < 20
            assert int(avail[:, 1].max()) < 20

    def test_off_map_treated_as_occupied(self):
        grid = empty_grid(20, 20)
        avail = _occupancy_to_available(grid, safe_dist_cells=3.0)
        for r, c in avail:
            assert r >= 3 and r <= 16
            assert c >= 3 and c <= 16

    def test_excludes_cells_within_safe_dist_of_wall(self):
        grid = empty_grid(30, 30)
        grid[:, 10] = WorldOccupancy.FULL
        avail = _occupancy_to_available(grid, safe_dist_cells=2.5)
        for _, c in avail:
            assert abs(int(c) - 10) >= 2.5 - 1e-6

    def test_accepts_diagonal_cell_outside_disc(self):
        grid = empty_grid(20, 20)
        grid[10, 10] = WorldOccupancy.FULL
        avail = _occupancy_to_available(grid, safe_dist_cells=2.0)
        coords = {(int(r), int(c)) for r, c in avail}
        assert (12, 12) in coords  # L2 = 2*sqrt(2) ≈ 2.83 > 2; L∞ box would have rejected

    def test_full_grid_returns_empty(self):
        grid = np.full((20, 20), WorldOccupancy.FULL, dtype=np.uint8)
        avail = _occupancy_to_available(grid, safe_dist_cells=1.0)
        assert len(avail) == 0


class TestSampleGridPositions:
    def test_returns_n_distinct_cells(self):
        grid = empty_grid(40, 40)
        cells = _sample_grid_positions(grid, n=8, safe_dist_cells=2.0, rng=np.random.default_rng(0))
        assert cells.shape == (8, 2)
        assert len({tuple(c) for c in cells}) == 8

    @pytest.mark.parametrize("safe_dist_cells", [1.0, 2.5, 5.0])
    def test_clearance_from_walls(self, safe_dist_cells):
        grid = empty_grid(40, 60)
        grid[:, 30] = WorldOccupancy.FULL
        rng = np.random.default_rng(1)
        cells = _sample_grid_positions(grid, n=5, safe_dist_cells=safe_dist_cells, rng=rng)
        for r, c in cells:
            d = grid_distance_to_occupied(grid, int(r), int(c))
            assert d >= safe_dist_cells - 1e-6, f"cell ({r},{c}) only {d} from wall, need {safe_dist_cells}"

    def test_pairwise_distance_at_least_safe_dist(self):
        grid = empty_grid(80, 80)
        sd = 5.0
        cells = _sample_grid_positions(grid, n=20, safe_dist_cells=sd, rng=np.random.default_rng(2))
        diffs = cells[:, None, :] - cells[None, :, :]
        dists = np.linalg.norm(diffs, axis=2).astype(float)
        np.fill_diagonal(dists, np.inf)
        assert dists.min() >= sd - 1e-6

    def test_raises_when_no_room(self):
        grid = empty_grid(20, 20)
        with pytest.raises(RuntimeError):
            _sample_grid_positions(grid, n=5, safe_dist_cells=15.0, rng=np.random.default_rng(0))

    def test_raises_when_more_requested_than_available(self):
        grid = empty_grid(20, 20)
        # safe_dist_cells=6: feasible count is small; ask for many.
        with pytest.raises(RuntimeError):
            _sample_grid_positions(grid, n=200, safe_dist_cells=6.0, rng=np.random.default_rng(0))

    def test_deterministic_under_same_seed(self):
        grid = empty_grid(40, 40)
        a = _sample_grid_positions(grid, n=8, safe_dist_cells=3.0, rng=np.random.default_rng(42))
        b = _sample_grid_positions(grid, n=8, safe_dist_cells=3.0, rng=np.random.default_rng(42))
        assert np.array_equal(a, b)

    def test_handles_narrow_passage(self):
        h, w = 50, 50
        grid = np.full((h, w), WorldOccupancy.FULL, dtype=np.uint8)
        grid[:, 23:28] = WorldOccupancy.EMPTY  # width-5 vertical corridor
        cells = _sample_grid_positions(grid, n=3, safe_dist_cells=2.0, rng=np.random.default_rng(0))
        for _, c in cells:
            assert 23 <= int(c) <= 27


class TestGetPositionsOnMap:
    @staticmethod
    def make_wm(grid: np.ndarray, resolution: float = 0.05, seed: int = 0) -> WorldManager:
        wm = WorldManager.__new__(WorldManager)
        wm._map = make_map(grid, resolution=resolution)
        wm._stair_masks = {}
        episode_rng = EpisodeRng()
        episode_rng.reseed(seed)
        fake_node = SimpleNamespace(conf=SimpleNamespace(General=SimpleNamespace(RNG=episode_rng)))
        wm._NodeInterface__node = fake_node  # NodeInterface uses name-mangled storage
        return wm

    def test_returns_n_positions(self):
        wm = self.make_wm(empty_grid(60, 60))
        out = wm.get_positions_on_map(n=4, safe_dist=0.4, forbid=False)
        assert len(out) == 4
        for p in out:
            assert isinstance(p, Position)

    def test_world_pairwise_distance_respects_safe_dist(self):
        wm = self.make_wm(empty_grid(80, 80))
        safe_dist = 0.5
        out = wm.get_positions_on_map(n=8, safe_dist=safe_dist, forbid=False)
        coords = np.array([[p.x, p.y] for p in out])
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        assert dists.min() >= safe_dist - wm.resolution

    def test_forbid_persists_across_calls(self):
        wm = self.make_wm(empty_grid(80, 80), seed=7)
        safe_dist = 0.4
        first = wm.get_positions_on_map(n=1, safe_dist=safe_dist, forbid=True)
        second = wm.get_positions_on_map(n=1, safe_dist=safe_dist, forbid=False)
        d = float(np.linalg.norm(np.array([first[0].x - second[0].x, first[0].y - second[0].y])))
        assert d >= safe_dist - wm.resolution

    def test_forbid_false_does_not_persist(self):
        wm = self.make_wm(empty_grid(80, 80), seed=11)
        before = WorldOccupancy.full(wm.map.occupancy.grid).sum()
        wm.get_positions_on_map(n=3, safe_dist=0.3, forbid=False)
        after = WorldOccupancy.full(wm.map.occupancy.grid).sum()
        assert before == after

    def test_raises_when_map_too_tight(self):
        wm = self.make_wm(empty_grid(20, 20))
        with pytest.raises(RuntimeError):
            wm.get_positions_on_map(n=5, safe_dist=2.0, forbid=False)

    def test_forbidden_zones_enforced(self):
        wm = self.make_wm(empty_grid(80, 80), seed=3)
        zone = PositionRadius(x=2.0, y=2.0, radius=0.5)
        out = wm.get_positions_on_map(n=3, safe_dist=0.3, forbidden_zones=[zone], forbid=False)
        assert len(out) == 3
        for p in out:
            assert not (1.5 <= p.x <= 2.5 and 1.5 <= p.y <= 2.5), f"{p.x},{p.y} inside forbidden zone"


class TestCoordinateTransforms:
    def test_round_trip_asymmetric(self):
        wm = make_map(empty_grid(40, 80), resolution=0.05, origin=(10.0, -5.0))
        for x, y in [(10.5, -4.9), (12.0, -4.0), (13.5, -3.1)]:
            row, col = wm.tf_pos2grid(Position(x=x, y=y))
            back = wm.tf_grid2pos((row, col))
            assert back.x == pytest.approx(x, abs=wm.resolution)
            assert back.y == pytest.approx(y, abs=wm.resolution)

    def test_orientation(self):
        h, w, res, ox, oy = 40, 80, 0.05, 10.0, -5.0
        wm = make_map(empty_grid(h, w), resolution=res, origin=(ox, oy))
        max_y = oy + h * res
        assert tuple(int(v) for v in wm.tf_pos2grid(Position(x=ox, y=max_y))) == (0, 0)
        assert tuple(int(v) for v in wm.tf_pos2grid(Position(x=ox, y=oy))) == (h, 0)


class TestZoneMask:
    def test_matches_pointwise_contains(self):
        wmap = make_map(empty_grid(60, 80), resolution=0.1, origin=(-1.0, 0.5))
        polys = [shapely.Polygon([(0.0, 1.0), (3.0, 1.0), (3.0, 2.5), (0.0, 2.5)]), shapely.Polygon([(4.0, 3.0), (6.0, 3.0), (5.0, 5.0)])]
        mask = _zone_mask(wmap, polys)
        rows, cols = np.mgrid[0:60, 0:80]
        expected = np.zeros((60, 80), dtype=bool)
        for r, c in zip(rows.ravel(), cols.ravel(), strict=True):
            p = wmap.tf_grid2pos((int(r), int(c)))
            expected[r, c] = any(poly.contains(shapely.Point(p.x, p.y)) for poly in polys)
        assert np.array_equal(mask, expected)
        assert mask.sum() > 0


class TestRenderedSampling:
    @staticmethod
    def make_wm_from_level(level: Level, seed: int = 0) -> WorldManager:
        wm = WorldManager.__new__(WorldManager)
        wm._map = WorldMap.from_world_description(level, resolution=0.05, time=Time())
        wm._multi_map = None
        wm._zone_masks = {}
        wm._stair_masks = {}
        episode_rng = EpisodeRng()
        episode_rng.reseed(seed)
        wm._NodeInterface__node = SimpleNamespace(conf=SimpleNamespace(General=SimpleNamespace(RNG=episode_rng)))
        return wm

    def test_positions_land_inside_rendered_zone(self):
        corners = [GeoPosition(10, 5), GeoPosition(30, 5), GeoPosition(30, 15), GeoPosition(10, 15), GeoPosition(10, 5)]
        level = Level(zones=[Zone(name="room", polygon=corners)])
        wm = self.make_wm_from_level(level)
        out = wm.get_positions_on_map(n=10, safe_dist=0.3, forbid=False)
        assert len(out) == 10
        for p in out:
            assert 10 <= p.x <= 30, f"x={p.x} outside rendered zone [10,30]"
            assert 5 <= p.y <= 15, f"y={p.y} outside rendered zone [5,15]"


class TestStairsExclusion:
    RES = 0.05

    def make_wm(self, world: World) -> WorldManager:
        wm = WorldManager.__new__(WorldManager)
        wm._map = make_map(empty_grid(100, 100), resolution=self.RES)
        wm._map.level_origins = {}
        wm._multi_map = None
        episode_rng = EpisodeRng()
        episode_rng.reseed(0)
        wm._NodeInterface__node = SimpleNamespace(conf=SimpleNamespace(General=SimpleNamespace(RNG=episode_rng)))
        wm.update_world(wm._map, world)
        return wm

    def test_samples_never_land_on_a_stair_half(self):
        stair = Stairs(name="stairs_low", position=GeoPosition(2.5, 2.5), size=[4.0, 4.0], mouth_side="-x", to="stairs_high")
        wm = self.make_wm(World(levels={"0": Level(id="0", stairs=[stair])}))
        out = wm.get_positions_on_map(n=40, safe_dist=0.05, forbid=False)
        assert len(out) == 40
        interior = shapely.Polygon([(corner.x, corner.y) for corner in stair.corners()]).buffer(-self.RES)
        assert not any(interior.contains(shapely.Point(p.x, p.y)) for p in out)


class TestStaticFootprints:
    """update_world rasterizes static entity footprints, or a fixed square when unannotated."""

    RES = 0.05

    @staticmethod
    def make_obstacle(x: float, y: float) -> Obstacle:
        return Obstacle(name="thing", model="box", pose=Pose(position=GeoPosition(x, y)))

    def make_wm(self, h: int = 100, w: int = 100, level_origins: dict | None = None) -> WorldManager:
        wm = WorldManager.__new__(WorldManager)
        wm._map = make_map(empty_grid(h, w), resolution=self.RES)
        wm._map.level_origins = level_origins or {}
        wm._multi_map = None
        return wm

    @staticmethod
    def world_with(*obstacles: Obstacle, level_id: str = "0") -> World:
        return World(levels={level_id: Level(id=level_id, objects=list(obstacles))})

    @staticmethod
    def occupied_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
        rows, cols = np.nonzero(~WorldOccupancy.not_full(grid))
        return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1

    def test_poly2mask_matches_rect_cells(self):
        wm = self.make_wm()
        mask = wm._map.tf_poly2mask(shapely.box(1.0, 1.0, 2.0, 1.5))
        rows, cols = np.nonzero(mask)
        assert (int(rows.min()), int(rows.max()) + 1) == (100 - 30, 100 - 20)
        assert (int(cols.min()), int(cols.max()) + 1) == (20, 40)
        assert int(mask.sum()) == 10 * 20

    def test_diagonal_footprint_is_not_its_envelope(self):
        wm = self.make_wm()
        poly = shapely.affinity.rotate(shapely.box(1.5, 2.25, 3.5, 2.75), 45)
        wm.update_world(wm._map, self.world_with(self.make_obstacle(2.5, 2.5)), static_footprints=[poly])
        occupied = int((~WorldOccupancy.not_full(wm._map.occupancy.grid)).sum())
        area_cells = 1.0 / (self.RES**2)
        assert 0.85 * area_cells <= occupied <= 1.2 * area_cells

    def test_unannotated_keeps_unit_radius_square(self):
        wm = self.make_wm()
        wm.update_world(wm._map, self.world_with(self.make_obstacle(2.5, 2.5)), static_footprints=[None])
        r0, r1, c0, c1 = self.occupied_bbox(wm._map.occupancy.grid)
        assert (r1 - r0, c1 - c0) == (40, 40)
        assert (c0, c1) == (30, 70)

    def test_rect_footprint_keeps_aspect(self):
        wm = self.make_wm()
        poly = shapely.box(1.0, 2.0, 3.0, 2.5)
        wm.update_world(wm._map, self.world_with(self.make_obstacle(2.0, 2.25)), static_footprints=[poly])
        r0, r1, c0, c1 = self.occupied_bbox(wm._map.occupancy.grid)
        assert (c0, c1) == (20, 60)
        assert (r1 - r0) == 10

    def test_rotated_footprint_swaps_axes(self):
        wm = self.make_wm()
        poly = shapely.affinity.rotate(shapely.box(1.0, 2.0, 3.0, 2.5), 90)
        wm.update_world(wm._map, self.world_with(self.make_obstacle(2.0, 2.25)), static_footprints=[poly])
        r0, r1, c0, c1 = self.occupied_bbox(wm._map.occupancy.grid)
        assert (c1 - c0) == 10
        assert (r1 - r0) == 40

    def test_missing_footprints_fall_back(self):
        wm = self.make_wm()
        wm.update_world(wm._map, self.world_with(self.make_obstacle(2.5, 2.5), self.make_obstacle(1.0, 1.0)), static_footprints=[shapely.box(2.0, 2.0, 3.0, 3.0), None])
        grid = wm._map.occupancy.grid
        assert not WorldOccupancy.not_full(grid)[100 - 30, 30]
        assert not WorldOccupancy.not_full(grid)[100 - 25, 25]
        assert WorldOccupancy.not_full(grid)[100 - 70, 70]

    def test_footprint_count_mismatch_raises(self):
        wm = self.make_wm()
        with pytest.raises(ValueError):
            wm.update_world(wm._map, self.world_with(self.make_obstacle(1.0, 1.0)), static_footprints=[])

    def test_no_footprints_marks_every_entity(self):
        wm = self.make_wm()
        wm.update_world(wm._map, self.world_with(self.make_obstacle(1.0, 1.0)))
        assert not WorldOccupancy.not_full(wm._map.occupancy.grid)[100 - 20, 20]

    def test_level_origin_shifts_combined_map_only(self):
        wm = self.make_wm(level_origins={"1": (2.0, 0.0)})
        level_map = make_map(empty_grid(100, 100), resolution=self.RES)
        wm._multi_map = MultiLevelMap({"1": level_map})
        poly = shapely.box(0.5, 0.5, 1.0, 1.0)
        wm.update_world(wm._map, self.world_with(self.make_obstacle(0.75, 0.75), level_id="1"), multi_level_map=wm._multi_map, static_footprints=[poly])
        _, _, c0, c1 = self.occupied_bbox(wm._map.occupancy.grid)
        assert (c0, c1) == (50, 60)
        _, _, lc0, lc1 = self.occupied_bbox(level_map.occupancy.grid)
        assert (lc0, lc1) == (10, 20)

    def test_single_frame_map_has_no_level_offset(self):
        wm = self.make_wm()
        wm.update_world(wm._map, self.world_with(self.make_obstacle(0.75, 0.75)), static_footprints=[shapely.box(0.5, 0.5, 1.0, 1.0)])
        _, _, c0, c1 = self.occupied_bbox(wm._map.occupancy.grid)
        assert (c0, c1) == (10, 20)


class TestElevatorHops:
    """level_hops rides cabin to cabin over the world's elevator destinations."""

    @staticmethod
    def make_wm(world: World, level_origins: dict[str, tuple[float, float]]) -> WorldManager:
        wm = WorldManager.__new__(WorldManager)
        wm._map = make_map(empty_grid(10, 10))
        wm._map.level_origins = level_origins
        wm._world = world
        return wm

    @staticmethod
    def lift(name: str, x: float, *destinations: str) -> Elevator:
        return Elevator(name=name, position=GeoPosition(x, 1.0), size=[2.0, 2.0, 2.0], destinations=list(destinations))

    def stacked(self) -> World:
        return World(
            levels={
                "0": Level(id="0", elevators=[self.lift("a_0", 1.0, "a_1")]),
                "1": Level(id="1", elevators=[self.lift("a_1", 1.0, "a_0"), self.lift("b_1", 5.0, "b_2")]),
                "2": Level(id="2", elevators=[self.lift("b_2", 5.0, "b_1")]),
            }
        )

    def test_route_boards_each_shaft_on_its_source_level(self):
        wm = self.make_wm(self.stacked(), {"0": (0.0, 0.0), "1": (20.0, 0.0), "2": (40.0, 0.0)})
        hops = wm.level_hops("0", "2")
        assert [(h.board_position.x, h.board_position.y) for h in hops] == [(1.0, 1.0), (25.0, 1.0)]
        assert all(wm.elevator_ride(h, 0.3).boarding_radius == pytest.approx(1.0 - 0.05 - 0.025 - 0.3) for h in hops)

    def test_route_is_empty_on_the_same_level_or_without_a_shaft(self):
        wm = self.make_wm(self.stacked(), {"0": (0.0, 0.0), "1": (20.0, 0.0), "2": (40.0, 0.0)})
        assert wm.level_hops("1", "1") == []
        assert wm.level_hops("0", "3") == []

    def test_destination_on_an_unloaded_level_is_no_route(self):
        world = self.stacked()
        del world.levels["2"]
        wm = self.make_wm(world, {"0": (0.0, 0.0), "1": (20.0, 0.0)})
        assert wm.level_hops("1", "2") == []


class TestLevelOfPoint:
    def test_point_maps_to_the_level_whose_zones_hold_it(self):
        room = [GeoPosition(0, 0), GeoPosition(4, 0), GeoPosition(4, 3), GeoPosition(0, 3)]
        wm = WorldManager.__new__(WorldManager)
        wm._map = make_map(empty_grid(10, 10))
        wm._map.level_origins = {"0": (0.0, 0.0), "1": (10.0, 0.0)}
        wm._world = World(levels={"0": Level(id="0", zones=[Zone(name="a", polygon=room)]), "1": Level(id="1", zones=[Zone(name="b", polygon=room)])})
        assert wm.level_of_point(2.0, 1.0) == "0"
        assert wm.level_of_point(12.0, 1.0) == "1"
        assert wm.level_of_point(7.0, 1.0) == ""


class TestPreviewMaps:
    def test_level_maps_load_from_the_world_preview_maps(self, tmp_path):
        room = [GeoPosition(0, 0), GeoPosition(4, 0), GeoPosition(4, 3), GeoPosition(0, 3)]
        view = WorldView(tmp_path / "one_room")
        view.save(World(levels={"0": Level(id="0", zones=[Zone(name="room", polygon=room)])}))
        maps = WorldManagerROS._load_multi_level_map(SimpleNamespace(), view, ["0", "1"])
        assert maps is not None
        assert list(maps.level_ids) == ["0"]
        level_map = maps.select_map("0")
        assert level_map.shape[0] > 0 and level_map.shape[1] > 0
        assert level_map.origin.x < 0.0 and level_map.origin.y < 0.0
