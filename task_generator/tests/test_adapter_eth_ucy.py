"""ETH/UCY importer: transposed sequence files into replay scenarios in a world of the walked area, read by the native loaders."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

NATIVE_WORLD = "sbend_corridor"
NATIVE_SCENARIO = "default"
WALKERS = {
    "zara01": [(1, 0, 40, (0.0, 0.0), (4.0, 0.0)), (2, 10, 60, (4.0, 1.0), (0.0, 1.0))],
    "zara02": [(5, 0, 50, (0.0, -1.0), (0.0, 3.0))],
    "eth": [(7, 780, 810, (1.0, 2.0), (1.0, 5.0)), (8, 790, 830, (3.0, 2.0), (3.0, 6.0))],
}


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")
    pytest.importorskip("arena_simulation_setup.shared.track")


def write_sequence(path: Path, walkers: list[tuple[int, int, int, tuple[float, float], tuple[float, float]]]) -> None:
    """A SocNavBench sequence file: rows frame, id, y, x, one column per sample, a walker per straight line."""
    columns = []
    for ident, first, last, (x0, y0), (x1, y1) in walkers:
        for frame in range(first, last + 1):
            u = (frame - first) / (last - first)
            columns.append((frame, ident, round(y0 + u * (y1 - y0), 4), round(x0 + u * (x1 - x0), 4)))
    columns.sort()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(zip(*columns, strict=True))


def seed(cache: Path) -> Path:
    from task_generator.adapters.eth_ucy import SEQUENCES

    for item, walkers in WALKERS.items():
        write_sequence(cache / SEQUENCES[item].file, walkers)
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from task_generator.adapters.eth_ucy import ETHUCY

    return ETHUCY(cache=seed(tmp_path / "cache"))


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _view(out: Path, world: str):
    from arena_simulation_setup.tree.World import WorldIdentifier

    return WorldIdentifier(world).load(out / world)


def _level(view):
    return view.load().levels["0"]


def test_registry_target_is_the_class() -> None:
    from arena_assets.importers import IMPORTERS

    from task_generator.adapters.eth_ucy import ETHUCY

    assert IMPORTERS.load("scenario", "eth_ucy") is ETHUCY
    assert ETHUCY.meta.kind == "scenario"


def test_items_and_unknown_item(importer) -> None:
    assert list(importer.items()) == ["eth", "hotel", "univ", "zara01", "zara02"]
    with pytest.raises(KeyError, match="items: eth, hotel, univ, zara01, zara02"):
        importer.fetch("zara03")


def test_sequence_rows_are_frame_id_y_x_counted_from_the_first_frame(tmp_path: Path) -> None:
    from task_generator.adapters.eth_ucy import parse_sequence

    path = tmp_path / "eth.csv"
    write_sequence(path, WALKERS["eth"])
    bodies = {body.name: body for body in parse_sequence(path, 15.0)}
    assert sorted(bodies) == ["ped_7", "ped_8"]
    assert (bodies["ped_7"].t[0], bodies["ped_7"].t[-1]) == pytest.approx((0.0, 2.0))
    assert bodies["ped_8"].t[0] == pytest.approx(10 / 15)
    assert (bodies["ped_7"].x[0], bodies["ped_7"].y[0], bodies["ped_7"].y[-1]) == pytest.approx((1.0, 2.0, 5.0))


def test_scenario_is_a_native_world_with_a_track_per_pedestrian(importer, tmp_path: Path) -> None:
    out = tmp_path / "worlds"
    target = importer.import_item("zara01", out)
    assert target == out / "eth_ucy_zara" / "scenarios" / "zara01"
    assert _files(out / "eth_ucy_zara") == ["ATTRIBUTION.yaml", "map/0.png", "map/0.yaml", "scenarios/zara01/ATTRIBUTION.yaml", "scenarios/zara01/ped_1.csv", "scenarios/zara01/ped_2.csv", "scenarios/zara01/scenario.yaml", "world.yaml"]
    view = _view(out, "eth_ucy_zara")
    assert list(view.load(validate=True).levels) == ["0"]
    assert [scenario.name for scenario in view.scenario.listall()] == ["zara01"]


def test_scenario_entries_carry_the_keys_of_a_native_entry(importer, tmp_path: Path) -> None:
    import yaml
    from arena_simulation_setup.tree.World import WorldIdentifier

    native_view = WorldIdentifier(NATIVE_WORLD).resolve_sync().scenario(NATIVE_SCENARIO).resolve_sync()
    native = yaml.safe_load(Path(native_view.scenario_path).read_text())
    imported = yaml.safe_load((importer.import_item("zara01", tmp_path / "worlds") / "scenario.yaml").read_text())
    assert [entry["name"] for entry in imported["humans"]] == ["ped_1", "ped_2"]
    for entry in imported["humans"]:
        assert set(entry) == set(native["humans"][0]) | {"velocity", "waypoint_mode", "stays", "track"}
    assert imported["robots"] == []


def test_tracks_replay_the_recorded_positions_at_the_sequence_frame_rate(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import Track

    scenario = importer.import_item("zara01", tmp_path / "worlds")
    walker = Track.load(scenario / "ped_2.csv")
    assert (walker.start, walker.end) == pytest.approx((10 / 25, 60 / 25))
    middle = walker.state(35 / 25)
    assert (middle.x, middle.y) == pytest.approx((2.0, 1.0), abs=1e-3)
    assert math.hypot(middle.vx, middle.vy) == pytest.approx(4.0 / 2.0, rel=1e-3)
    assert math.cos(middle.yaw) == pytest.approx(-1.0, abs=1e-6)


def test_scenario_loads_as_native_pedestrians_walking_their_tracks(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.shared.track import Track, track_file
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    from task_generator.adapters.eth_ucy import MODEL

    out = tmp_path / "worlds"
    directory = importer.import_item("zara01", out)
    scenario = _view(out, "eth_ucy_zara").scenario("zara01").resolve_sync().load()
    assert scenario.objects == []
    assert scenario.robots == []
    assert sorted(ped.name for ped in scenario.humans) == ["ped_1", "ped_2"]
    for ped in scenario.humans:
        assert type(ped) is DynamicObstacle
        assert isinstance(ped.model, HumanIdentifier)
        assert ped.model.name == MODEL
        assert ped.included_from == directory
        assert track_file(ped) == directory / f"{ped.name}.csv"
        track = Track.load(track_file(ped))
        assert (ped.pose.position.x, ped.pose.position.y) == pytest.approx((track.x[0], track.y[0]), abs=1e-3)
        assert math.remainder(ped.pose.orientation.to_yaw() - track.yaw[0], 2.0 * math.pi) == pytest.approx(0.0, abs=1e-3)
        assert ped.velocity == pytest.approx({"ped_1": 2.5, "ped_2": 2.0}[ped.name], abs=0.01)
        assert ped.agent == {"agent_type": "adult", "desired_velocity": ped.velocity}
        assert ped.waypoint_mode == "once"
        assert len(ped.waypoints) >= 1
        last = ped.waypoints[-1]
        assert math.hypot(last.x - track.x[-1], last.y - track.y[-1]) < 1e-3


def test_agent_block_parses_in_the_humansim_adapter(importer, tmp_path: Path) -> None:
    pytest.importorskip("arena_humansim_msgs.msg")
    pytest.importorskip("arena_humansim.core.agents")
    from arena_humansim_msgs.msg import Waypoints

    from task_generator.simulators.human.arena_humansim.agent import ArenaHumanDynamicObstacle

    out = tmp_path / "worlds"
    importer.import_item("zara01", out)
    view = _view(out, "eth_ucy_zara")
    for ped in view.scenario("zara01").resolve_sync().load().humans:
        parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(ped)
        assert parsed.agent_type == "adult"
        assert parsed.waypoint_mode == Waypoints.MODE_ONCE
        assert parsed.desired_velocity_min == parsed.desired_velocity_max == ped.velocity
        assert parsed.sample_params(np.random.default_rng(0)).desired_velocity == pytest.approx(ped.velocity)


def test_resample_holds_the_heading_while_standing_and_leaves_long_gaps_open() -> None:
    from task_generator.adapters.eth_ucy import MAX_GAP_S, MIN_HEADING_SPEED, TRACK_HZ, Body, resample

    t = np.round(np.arange(1, 801) / 100.0, 2)
    x = np.select([t <= 2.0, t <= 3.0, t <= 4.0], [0.0, t - 2.0, 1.0], 1.0 - (t - 6.0))
    y = np.select([t <= 3.0, t <= 4.0], [0.0, t - 3.0], 1.0)
    seen = (t <= 4.0) | (t > 6.0)
    track = resample(Body("walker", t[seen], x[seen], y[seen]), 0.01, TRACK_HZ, MAX_GAP_S, MIN_HEADING_SPEED)
    assert MAX_GAP_S < 2.0
    assert track.spans == [(0.0, pytest.approx(3.9)), (pytest.approx(6.0), pytest.approx(7.9))]
    assert track.state(5.0) is None
    held = track.t < 1.5
    assert held.sum() > TRACK_HZ
    assert track.x[held] == pytest.approx(np.zeros(held.sum()), abs=1e-9)
    assert track.y[held] == pytest.approx(np.zeros(held.sum()), abs=1e-9)
    assert np.all(track.yaw[held] == 0.0)
    walking_x = track.state(2.5)
    assert walking_x.yaw == pytest.approx(0.0, abs=1e-6)
    assert (walking_x.vx, walking_x.vy) == pytest.approx((1.0, 0.0), abs=1e-3)
    walking_y = track.state(3.5)
    assert walking_y.yaw == pytest.approx(math.pi / 2, abs=1e-6)
    assert (walking_y.vx, walking_y.vy) == pytest.approx((0.0, 1.0), abs=1e-3)
    walking_back = track.state(7.0)
    assert abs(walking_back.yaw) == pytest.approx(math.pi, abs=1e-6)
    assert (walking_back.vx, walking_back.vy) == pytest.approx((-1.0, 0.0), abs=1e-3)


def _stays_track():
    from task_generator.adapters.eth_ucy import MAX_GAP_S, MIN_HEADING_SPEED, TRACK_HZ, Body, resample

    t = np.round(np.arange(0, 91) / 10.0, 1)
    x = np.where(t <= 7.0, 0.0, (t - 7.0) * 0.5)
    y = np.clip(t - 2.0, 0.0, 4.0)
    second = np.round(np.arange(120, 141) / 10.0, 1)
    third = np.round(np.arange(141, 162) / 10.0, 1)
    short = np.round(np.arange(180, 186) / 10.0, 1)
    body = Body(
        "walker",
        np.concatenate([t, second, third, short]),
        np.concatenate([x, 5.0 + (second - 12.0), 20.0 + (third - 14.1), np.full(len(short), 30.0)]),
        np.concatenate([y, np.full(len(second) + len(third) + len(short), 5.0)]),
    )
    return resample(body, 0.0, TRACK_HZ, MAX_GAP_S, MIN_HEADING_SPEED)


def test_stays_split_at_gaps_and_jumps_and_time_their_legs() -> None:
    from task_generator.adapters.eth_ucy import stays

    first, second, third = stays(_stays_track())
    assert first == {"start": 0.0, "pose": [0.0, 0.0, 1.571], "velocity": [0.0, 0.0], "legs": [[0.0, 0.0, 0.1, 2.0], [0.0, 4.0, 1.0, 1.0], [1.0, 4.0, 0.5, 0.0]]}
    assert second == {"start": 12.0, "pose": [5.0, 5.0, 0.0], "velocity": [1.0, 0.0], "legs": [[7.0, 5.0, 1.0, 0.0]]}
    assert third == {"start": 14.1, "pose": [20.0, 5.0, 0.0], "velocity": [1.0, 0.0], "legs": [[22.0, 5.0, 1.0, 0.0]]}


def test_stays_match_the_timed_desire_of_the_humansim_importer(tmp_path: Path) -> None:
    data = pytest.importorskip("arena_humansim.importer.data")
    from task_generator.adapters.eth_ucy import stays

    track = _stays_track()
    (tmp_path / "walker.csv").write_bytes(track.dumps())
    desires = [data.desire(data.Appearance("walker", t, xy), pace="timed") for t, xy in data.load_track(tmp_path / "walker.csv") if len(t) >= 2 and t[-1] - t[0] >= data.MIN_STAY_S]
    ours = stays(track)
    assert len(ours) == len(desires) == 3
    for stay, wanted in zip(ours, desires, strict=True):
        assert stay["start"] == pytest.approx(wanted.spawn_t, abs=1e-3)
        assert stay["pose"] == pytest.approx(list(wanted.pose), abs=1e-3)
        assert stay["velocity"] == pytest.approx(list(wanted.velocity), abs=1e-3)
        assert np.asarray(stay["legs"]) == pytest.approx(np.asarray([[leg.x, leg.y, leg.speed, leg.dwell] for leg in wanted.legs]), abs=1e-2)


def test_a_sequence_is_moved_into_the_frame_of_its_scene(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import Track

    from task_generator.adapters.eth_ucy import SEQUENCES

    assert [item for item, entry in SEQUENCES.items() if entry.shift != (0.0, 0.0)] == ["zara02"]
    dx, dy = SEQUENCES["zara02"].shift
    walker = Track.load(importer.import_item("zara02", tmp_path / "worlds") / "ped_5.csv")
    first, last = walker.state(walker.start), walker.state(walker.end)
    assert (first.x, first.y) == pytest.approx((0.0 + dx, -1.0 + dy), abs=1e-3)
    assert (last.x, last.y) == pytest.approx((0.0 + dx, 3.0 + dy), abs=1e-3)


def test_floor_holds_every_walked_path_with_a_footprint(importer, tmp_path: Path) -> None:
    import shapely

    from task_generator.adapters.eth_ucy import FOOTPRINT_M, SEQUENCES

    out = tmp_path / "worlds"
    importer.import_item("zara01", out)
    level = _level(_view(out, "eth_ucy_zara"))
    floor = shapely.union_all([shapely.Polygon([(corner.x, corner.y) for corner in zone.polygon]) for zone in level.zones])
    walls = shapely.MultiLineString([((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)) for wall in level.all_walls])
    for item in ("zara01", "zara02"):
        dx, dy = SEQUENCES[item].shift
        for _, _, _, start, end in WALKERS[item]:
            path = shapely.LineString([(start[0] + dx, start[1] + dy), (end[0] + dx, end[1] + dy)])
            assert floor.buffer(1e-6).covers(path)
            assert walls.distance(path) >= FOOTPRINT_M - 0.06


def test_floor_is_one_plain_polygon_with_room_to_turn_in_every_corner() -> None:
    import shapely

    from task_generator.adapters.eth_ucy import FOOTPRINT_M, Body, walked_area

    crowd = [Body(f"ped_{i}", np.array([0.0, 1.0]), np.array([0.0, 8.0]), np.array([0.2 * i, 0.2 * i])) for i in range(11)]
    stray = Body("stray", np.array([0.0, 1.0, 2.0]), np.array([8.0, 11.0, 11.5]), np.array([1.0, 6.0, 6.2]))
    floor = walked_area([*crowd, stray])
    assert floor.geom_type == "Polygon"
    assert not floor.interiors
    assert len(floor.exterior.coords) - 1 <= 16
    walked = shapely.union_all([shapely.LineString(np.column_stack([body.x, body.y])) for body in (*crowd, stray)])
    assert floor.covers(walked)
    assert floor.exterior.distance(walked) >= FOOTPRINT_M - 1e-9
    reached = floor.buffer(-FOOTPRINT_M).buffer(FOOTPRINT_M)
    assert max(piece.area for piece in shapely.get_parts(floor.difference(reached))) < 0.1


def test_floor_refuses_a_scene_nobody_walked() -> None:
    from task_generator.adapters.eth_ucy import walked_area

    with pytest.raises(ValueError, match="no pedestrian walked the scene"):
        walked_area([])


def test_both_sequences_of_a_scene_share_one_world_in_either_order(tmp_path: Path) -> None:
    from task_generator.adapters.eth_ucy import ETHUCY

    worlds = []
    for order in (("zara01", "zara02"), ("zara02", "zara01")):
        importer = ETHUCY(cache=seed(tmp_path / "-".join(order) / "cache"))
        out = tmp_path / "-".join(order) / "worlds"
        for item in order:
            importer.import_item(item, out)
        assert sorted(scenario.name for scenario in _view(out, "eth_ucy_zara").scenario.listall()) == ["zara01", "zara02"]
        worlds.append((out / "eth_ucy_zara" / "world.yaml").read_bytes())
    assert worlds[0] == worlds[1]


def test_attribution_names_the_dataset_authors_and_the_converter(importer, tmp_path: Path) -> None:
    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    from task_generator.adapters.eth_ucy import REVISION

    out = tmp_path / "worlds"
    importer.import_item("eth", out)
    attribution = read_attribution(out / "eth_ucy_eth" / ATTRIBUTION_FILE)
    assert REVISION in attribution.url
    assert attribution.license == "unspecified"
    assert "Pellegrini" in attribution.authors and "SocNavBench" in attribution.authors
    assert "ICCV 2009" in attribution.citation


def test_cli_scenario_import_brings_its_world_where_the_resolver_looks(tmp_path: Path) -> None:
    cache = seed(tmp_path / "cache")
    env = {**os.environ, "ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "imported")}
    target = tmp_path / "imported" / "worlds" / "eth_ucy_eth"
    imported = subprocess.run([sys.executable, "-m", "arena_assets", "import", "scenario", "eth_ucy", "eth", "--cache", str(cache)], env=env, capture_output=True, text=True, check=True)
    assert imported.stdout.splitlines()[-1] == str(target / "scenarios" / "eth")
    code = "from arena_simulation_setup.tree.World import WorldIdentifier\nview = WorldIdentifier('eth_ucy_eth').resolve_sync()\nprint(view.path)\nprint(*sorted(scenario.name for scenario in view.scenario.listall()))"
    resolved = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert resolved.stdout.splitlines() == [str(target), "eth"]


@pytest.mark.network
def test_live_eth_walks_at_pedestrian_speed_on_its_15_fps_clock(tmp_path: Path) -> None:
    from task_generator.adapters.eth_ucy import ETHUCY, SEQUENCES, parse_sequence

    bodies = parse_sequence(ETHUCY(cache=tmp_path / "cache").provider.get(SEQUENCES["eth"].file, tmp_path / "eth.csv"), SEQUENCES["eth"].fps)
    assert len(bodies) == 360
    speeds = np.concatenate([np.hypot(np.diff(body.x), np.diff(body.y)) / np.diff(body.t) for body in bodies])
    assert 1.2 < float(np.median(speeds[speeds > 0.3])) < 1.8


@pytest.mark.network
def test_live_zara_sequences_walk_the_same_pavement_in_the_scene_frame(tmp_path: Path) -> None:
    from task_generator.adapters.eth_ucy import ETHUCY, SEQUENCES, parse_sequence

    provider = ETHUCY(cache=tmp_path / "cache").provider
    walked = []
    for item in ("zara01", "zara02"):
        entry = SEQUENCES[item]
        bodies = parse_sequence(provider.get(entry.file, tmp_path / f"{item}.csv"), entry.fps, entry.shift)
        walked.append({(round(float(x) / 0.1), round(float(y) / 0.1)) for body in bodies for x, y in zip(body.x, body.y, strict=True)})
    assert len(walked[0] & walked[1]) / len(walked[0] | walked[1]) > 0.65


@pytest.mark.network
def test_live_floors_are_plain_polygons_around_every_recorded_step(tmp_path: Path) -> None:
    import shapely

    from task_generator.adapters.eth_ucy import ETHUCY, FOOTPRINT_M, SEQUENCES, parse_sequence, scene_items, walked_area

    provider = ETHUCY(cache=tmp_path / "cache").provider
    for scene in ("eth", "hotel", "univ", "zara"):
        bodies = []
        for item in scene_items(scene):
            entry = SEQUENCES[item]
            bodies += parse_sequence(provider.get(entry.file, tmp_path / f"{item}.csv"), entry.fps, entry.shift)
        floor = walked_area(bodies)
        walked = shapely.union_all([shapely.LineString(np.column_stack([body.x, body.y])) for body in bodies])
        assert floor.covers(walked), scene
        assert floor.exterior.distance(walked) >= FOOTPRINT_M - 1e-9, scene
        assert len(floor.exterior.coords) - 1 <= 30, scene
        reached = floor.buffer(-FOOTPRINT_M).buffer(FOOTPRINT_M)
        assert max(piece.area for piece in shapely.get_parts(floor.difference(reached))) < 0.1, scene
