"""SocNavBench importer: curated episodes into replay scenarios with the robot start and goal, read by the native loaders."""

from __future__ import annotations

import re
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from adapter_helpers import write_sequence

NATIVE_WORLD = "sbend_corridor"
NATIVE_SCENARIO = "default"
ETH_FIRST_FRAME = 780
LATE_START_S = (931 - ETH_FIRST_FRAME) / 15
PED_6_ENTERS_S = (1021 - ETH_FIRST_FRAME) / 15
WALKERS = {
    "eth": [
        (1, 780, 1080, (0.0, 0.0), (10.0, 0.0)),
        (2, 931, 1006, (0.0, 2.0), (5.0, 2.0)),
        (3, 915, 990, (0.0, 4.0), (5.0, 4.0)),
        (4, 961, 1036, (6.0, 2.0), (6.0, 7.0)),
        (6, 1021, 1201, (0.0, 6.0), (12.0, 6.0)),
        (7, 1066, 1141, (2.0, 8.0), (7.0, 8.0)),
        (8, 990, 1050, (3.0, 1.0), (3.0, 5.0)),
    ],
    "zara01": [(1, 0, 100, (0.0, 0.0), (4.0, 0.0)), (2, 10, 110, (4.0, 1.0), (0.0, 1.0))],
    "zara02": [(1, 6, 106, (0.0, -1.0), (0.0, 3.0)), (2, 50, 150, (1.0, 3.0), (1.0, -1.0))],
}
PLACEMENTS = """[dataset_params]


[eth] ; recommended map: ETH
file_name=eth/world_coordinate_inter_eth.csv
fps=15 ; framerate of data capture
spawn_delay_s = 0
offset=[1.4, 14.4, 0] ; (x, y, theta) translation
swapxy=True
flipxn=True
flipyn=False

[hotel]
file_name=hotel/world_coordinate_inter_hotel.csv
fps=15
spawn_delay_s = 0
offset=[12, 4.5, 0]
swapxy=True
flipxn=False
flipyn=False

[dhotel-top]
file_name=hotel/world_coordinate_inter_hotel.csv
fps=15
spawn_delay_s = 0
offset=[12, 17.5, 0]
swapxy=True
flipxn=False
flipyn=False

[dhotel-bot]
file_name=hotel/world_coordinate_inter_hotel.csv
fps=15
spawn_delay_s = 0
offset=[12, 4.5, 0]
swapxy=True
flipxn=False
flipyn=False

[zara01]
file_name=zara01/world_coordinate_inter.csv
fps=25
spawn_delay_s = 0
offset=[9.5, 3.0, 0]
swapxy=False
flipxn=True
flipyn=True

[zara02]
file_name=zara02/world_coordinate_inter.csv
fps=25
spawn_delay_s = 0
offset=[9.5, 1.85, 0]
swapxy=False
flipxn=True
flipyn=True

[univ]
#file_name=univ/world_coordinate_inter_adjustedFrameID.csv
file_name=univ/world_coordinate_inter.csv
fps=15
spawn_delay_s = 0
offset=[10, 6, 0]
swapxy=False
flipxn=True
flipyn=True
"""
EPISODES = [
    ("t_all", ["eth"], [(0, -1)], [[8, 5, 1.5], [12, 15, 1.5]], 60),
    ("t_late", ["eth"], [(1, -1)], [[5, 1, 1.5], [12.5, 15.5, 0]], 8),
    ("t_cross", ["zara01"], [(0, -1)], [[8.0, 8, 0], [15, 3, -0.2]], 60),
    ("t_shifted", ["zara02"], [(0, -1)], [[20, 13, -1.5], [1.5, 8, -1.5]], 50),
    ("t_univ", ["univ"], [(0, -1)], [[8.0, 8.5, 0], [15, 1, -0.2]], 60),
    ("t_hotel", ["hotel"], [(361, -1)], [[2, 5, 0], [15, 5, 0.2]], 60),
    ("t_dhotel", ["dhotel-top", "dhotel-bot"], [(140, -1), (180, -1)], [[8, 1, -3], [9, 22.5, 1.5]], 60),
]
IMPORTED = ["t_all", "t_late", "t_cross", "t_shifted"]
LIVE = {
    "t_ETH1": ("socnavbench_eth", 0.0, 14, 20.0),
    "t_ETH2": ("socnavbench_eth", 0.0, 32, 60.0),
    "t_eth_follow": ("socnavbench_eth", 609.0, 75, 60.0),
    "t_eth_against": ("socnavbench_eth", 609.0, 75, 60.0),
    "t_eth_dense_flow": ("socnavbench_eth", 627.8, 66, 60.0),
    "t_eth_dense_against": ("socnavbench_eth", 627.8, 66, 60.0),
    "t_zara1_basic_cross": ("socnavbench_zara", 0.0, 27, 60.0),
    "t_zara1_wrap": ("socnavbench_zara", 0.0, 26, 50.0),
    "t_zara1_dense": ("socnavbench_zara", 202.44, 38, 60.0),
    "t_zara1_dense_rev": ("socnavbench_zara", 202.44, 38, 60.0),
    "t_zara1_dense_cross": ("socnavbench_zara", 202.44, 38, 60.0),
    "t_zara1_dense_crosswrap_rev": ("socnavbench_zara", 202.44, 38, 60.0),
    "t_zara2": ("socnavbench_zara", 0.0, 33, 60.0),
    "t_zara2_basic_against": ("socnavbench_zara", 15.04, 34, 60.0),
    "t_zara2_wrap": ("socnavbench_zara", 48.04, 34, 60.0),
    "t_zara2_wrap2": ("socnavbench_zara", 48.04, 34, 60.0),
    "t_zara2_against_seq": ("socnavbench_zara", 82.48, 23, 60.0),
}


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")
    pytest.importorskip("arena_simulation_setup.shared.track")


def episodes_ini() -> str:
    sections = ["[episodes_params]\n# names of which tests to include in the final benchmark\ntests=['t_all']\n\nwithout_robot=False ; lite version without the robot\n"]
    for name, datasets, ranges, start_goal, max_time in EPISODES:
        sections.append(
            f"[{name}]\nmap_name=Synthetic\npedestrian_datasets={datasets!r} ; prerecorded agents\ndatasets_start_t = {[0] * len(datasets)}\n"
            f"ped_ranges={ranges!r} ; [start, end] range\n## NOTE: all pos_3's are (x, y, theta)\nagents_start=[]\nagents_end=[]\n"
            f"robot_start_goal={start_goal!r} ; start and goal pos_3's of the robot\nmax_time={max_time} ; maximum time provided for this test\nwrite_episode_log=True\n"
        )
    return "\n".join(sections)


def seed(cache: Path) -> Path:
    from task_generator.adapters.eth_ucy import SEQUENCES
    from task_generator.adapters.socnavbench import DATASETS, EPISODES_FILE, PLACEMENTS_FILE

    for item, walkers in WALKERS.items():
        write_sequence(cache / DATASETS / SEQUENCES[item].file, walkers)
    (cache / EPISODES_FILE).parent.mkdir(parents=True, exist_ok=True)
    (cache / EPISODES_FILE).write_text(episodes_ini())
    (cache / PLACEMENTS_FILE).write_text(PLACEMENTS)
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from task_generator.adapters.socnavbench import SocNavBench

    return SocNavBench(cache=seed(tmp_path / "cache"))


def gather_posn(placement, x: float, y: float) -> tuple[float, float]:
    """Map position of a raw sample, the arithmetic of SocNavBench's gather_posn_data."""
    scale_x = -1 if placement.flipxn else 1
    scale_y = -1 if placement.flipyn else 1
    first, second = (scale_y * y, scale_x * x) if placement.swapxy else (scale_x * x, scale_y * y)
    s, c = math.sin(placement.offset[2]), math.cos(placement.offset[2])
    return first * c - second * s + placement.offset[0], first * s + second * c + placement.offset[1]


def _yaw(pose) -> float:
    return 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)


def _turn(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _view(out: Path, world: str):
    from arena_simulation_setup.tree.World import WorldIdentifier

    return WorldIdentifier(world).load(out / world)


def _level(view):
    return view.load().levels["0"]


def _floor(level):
    import shapely

    floor = shapely.union_all([shapely.Polygon([(corner.x, corner.y) for corner in zone.polygon]) for zone in level.zones])
    walls = shapely.MultiLineString([((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)) for wall in level.all_walls])
    return floor, walls


def _replay_stage():
    spec = importlib.util.spec_from_file_location("replay_stage", Path(__file__).parent / "ros" / "test_replay_backend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_provider_is_pinned_to_the_published_commit() -> None:
    from task_generator.adapters.socnavbench import BASE_URL, EPISODES_FILE, SocNavBench

    assert SocNavBench.default_provider().url(EPISODES_FILE) == f"{BASE_URL}/params/episode_params_val.ini"
    assert BASE_URL == "https://raw.githubusercontent.com/CMU-TBD/SocNavBench/2724ea85ff22ca61a88ee95285dfc3fa056656c6"


def test_items_are_the_episodes_on_sequences_played_as_recorded(importer) -> None:
    assert list(importer.items()) == IMPORTED
    assert list(importer.episodes) == [name for name, *_ in EPISODES]
    assert list(importer.placements) == ["eth", "hotel", "dhotel-top", "dhotel-bot", "zara01", "zara02", "univ"]
    assert (importer.placements["zara02"].file, importer.placements["zara02"].fps, importer.placements["zara02"].offset) == ("zara02/world_coordinate_inter.csv", 25.0, (9.5, 1.85, 0.0))
    late = importer.episode("t_late")
    assert (late.datasets, late.ped_ranges, late.max_time) == (("eth",), ((1, -1),), 8.0)
    assert (late.start, late.goal) == ((5.0, 1.0, 1.5), (12.5, 15.5, 0.0))


def test_unknown_item_names_the_items(importer) -> None:
    with pytest.raises(KeyError, match="no SocNavBench episode 't_nowhere', items: t_all, t_late, t_cross, t_shifted"):
        importer.fetch("t_nowhere")


def test_left_out_item_says_why(importer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'t_univ' is left out: SocNavBench plays univ at 15 fps although univ was recorded at 25 fps. Items: t_all"):
        importer.fetch("t_univ")
    with pytest.raises(ValueError, match="plays hotel at 15 fps although hotel was recorded at 25 fps, SocNavBench places hotel as a mirror image of hotel"):
        importer.import_item("t_hotel", tmp_path / "worlds")
    with pytest.raises(ValueError, match="it overlays 2 datasets, SocNavBench plays dhotel-top at 15 fps"):
        importer.attribution("t_dhotel")
    assert not (tmp_path / "worlds").exists()


def test_placement_inverts_gather_posn_data_with_heading(importer) -> None:
    import attrs

    from task_generator.adapters.socnavbench import Placement

    turned = Placement("turned.csv", 25.0, (1.0, -2.0, 0.7), swapxy=True, flipxn=False, flipyn=True)
    assert not turned.mirrored and attrs.evolve(turned, swapxy=False).mirrored
    for placement in (*(importer.placements[name] for name in ("eth", "zara01", "zara02")), turned, attrs.evolve(turned, swapxy=False)):
        for x, y, yaw in ((0.0, 0.0, 0.0), (3.2, -1.7, 2.1), (-4.0, 6.5, -2.9)):
            here = gather_posn(placement, x, y)
            ahead = gather_posn(placement, x + math.cos(yaw), y + math.sin(yaw))
            back = placement.from_map((*here, math.atan2(ahead[1] - here[1], ahead[0] - here[0])))
            assert back[:2] == pytest.approx((x, y), abs=1e-9), placement
            assert _turn(back[2], yaw) < 1e-9, placement
    assert not any(importer.placements[name].mirrored for name in ("eth", "zara01", "zara02", "univ"))
    assert all(importer.placements[name].mirrored for name in ("hotel", "dhotel-top", "dhotel-bot"))


def test_start_and_goal_land_in_the_frame_of_the_scene_world(importer) -> None:
    from task_generator.adapters.eth_ucy import SEQUENCES, pose_in_meters
    from task_generator.adapters.socnavbench import scene_pose

    eth = importer.placements["eth"]
    assert scene_pose(eth, SEQUENCES["eth"], (5.0, 1.0, 1.5)) == pytest.approx((13.4, 3.6, 1.5 + math.pi / 2), abs=1e-9)
    assert scene_pose(eth, SEQUENCES["eth"], (12.5, 15.5, 0.0)) == pytest.approx((-1.1, 11.1, math.pi / 2), abs=1e-9)
    assert gather_posn(eth, 13.4, 3.6) == pytest.approx((5.0, 1.0))
    zara = SEQUENCES["zara01"].to_meters
    zara01 = scene_pose(importer.placements["zara01"], SEQUENCES["zara01"], (8.0, 8.0, 0.0))
    expected = pose_in_meters(zara, (1.5, -5.0, math.pi))
    assert zara01[:2] == pytest.approx(expected[:2])
    assert _turn(zara01[2], expected[2]) < 1e-9
    zara02 = scene_pose(importer.placements["zara02"], SEQUENCES["zara02"], (20.0, 13.0, -1.5))
    expected = pose_in_meters(zara, (-10.5, -10.0, math.pi - 1.5))
    assert zara02[:2] == pytest.approx(expected[:2])
    assert _turn(zara02[2], expected[2]) < 1e-9
    raw = importer.placements["zara02"].from_map((20.0, 13.0, -1.5))
    assert pose_in_meters(zara, (raw[0] + SEQUENCES["zara02"].shift[0], raw[1] + SEQUENCES["zara02"].shift[1], raw[2]))[:2] == pytest.approx(zara02[:2])
    assert gather_posn(importer.placements["zara02"], raw[0], raw[1]) == pytest.approx((20.0, 13.0))


def test_selection_follows_generate_humans() -> None:
    from task_generator.adapters.socnavbench import select

    spans = {1: (100, 130), 2: (110, 450), 3: (400, 440), 5: (420, 470), 6: (1000, 1050), 7: (990, 1040)}
    rows = sorted((frame, ident) for ident, (first, last) in spans.items() for frame in range(first, last + 1, 10))
    frames, idents = (np.array(column) for column in zip(*rows, strict=True))
    assert select(frames, idents, (0, -1), 10.0, 40.0) == (100, [1, 2, 3, 5])
    assert select(frames, idents, (0, -1), 10.0, 1000.0) == (100, [1, 2, 3, 5, 6])
    assert select(frames, idents, (2, -1), 10.0, 5.0) == (400, [3, 5])
    assert select(frames, idents, (2, -1), 10.0, 1.9) == (400, [3])
    assert select(frames, idents, (3, -1), 10.0, 5.0) == (100, [])
    assert select(frames, idents, (3, -1), 10.0, 40.0) == (100, [5])
    assert select(frames, idents, (5, -1), 10.0, 5.0) == (1000, [6, 7])
    assert select(frames, idents, (0, 2), 10.0, 1000.0) == (100, [1, 2])
    assert select(frames, idents, (1, 3), 10.0, 1000.0) == (110, [2, 3])
    assert select(frames, idents, (2, 2), 10.0, 1000.0) == (100, [])
    assert select(frames, idents, (4, 3), 10.0, 1000.0) == (420, [5, 6, 7])


def test_window_cuts_a_track_onto_the_episode_clock(tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import Track

    from task_generator.adapters.socnavbench import window

    track = Track.from_segments(
        [
            (np.array([10.0, 11.0, 12.0]), np.array([0.0, 1.0, 2.0]), np.array([5.0, 5.0, 5.0]), np.array([0.0, 0.0, 0.0])),
            (np.array([14.0, 15.0, 16.0]), np.array([8.0, 8.0, 8.0]), np.array([0.0, 2.0, 4.0]), np.array([1.5, 1.5, 1.5])),
        ]
    )
    cut = window(track, 10.5, 15.25)
    assert cut.spans == [pytest.approx((0.0, 1.5)), pytest.approx((3.5, 4.75))]
    assert cut.t.tolist() == pytest.approx([0.0, 0.5, 1.5, 3.5, 4.5, 4.75])
    assert cut.x.tolist() == pytest.approx([0.5, 1.0, 2.0, 8.0, 8.0, 8.0])
    assert cut.y.tolist() == pytest.approx([5.0, 5.0, 5.0, 0.0, 2.0, 2.5])
    assert cut.yaw.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.5, 1.5, 1.5])
    cut.save(tmp_path / "cut.csv")
    assert Track.load(tmp_path / "cut.csv").spans == cut.spans
    assert window(track, 11.0, 12.0).t.tolist() == pytest.approx([0.0, 1.0])
    assert window(track, 11.0 - 1e-12, 12.0 + 1e-12).t.tolist() == pytest.approx([0.0, 1.0])
    assert len(window(track, 12.0, 14.0).t) == 0
    assert len(window(track, 20.0, 30.0).t) == 0
    assert window(track, 0.0, 100.0).t.tolist() == pytest.approx(track.t.tolist())


def test_episode_plays_the_selected_pedestrians_from_the_window_start(importer, tmp_path: Path) -> None:
    import yaml
    from arena_simulation_setup.shared.track import Track

    from task_generator.adapters.eth_ucy import ETHUCY
    from task_generator.adapters.socnavbench import DATASETS

    out = tmp_path / "worlds"
    target = importer.import_item("t_late", out)
    assert target == out / "socnavbench_eth" / "scenarios" / "t_late"
    assert _files(out / "socnavbench_eth") == [
        "ATTRIBUTION.yaml",
        "map/0.png",
        "map/0.yaml",
        "scenarios/t_late/ATTRIBUTION.yaml",
        "scenarios/t_late/ped_2.csv",
        "scenarios/t_late/ped_3.csv",
        "scenarios/t_late/ped_4.csv",
        "scenarios/t_late/ped_6.csv",
        "scenarios/t_late/scenario.yaml",
        "world.yaml",
    ]
    scenario = yaml.safe_load((target / "scenario.yaml").read_text())
    assert [entry["name"] for entry in scenario["humans"]] == ["ped_2", "ped_3", "ped_4", "ped_6"]
    played = {entry["name"]: Track.load(target / f"{entry['name']}.csv") for entry in scenario["humans"]}
    assert played["ped_3"].start == 0.0
    assert played["ped_2"].start == pytest.approx(10.1 - LATE_START_S, abs=1e-3)
    assert played["ped_4"].start == pytest.approx(12.1 - LATE_START_S, abs=1e-3)
    assert played["ped_6"].start == pytest.approx(16.1 - LATE_START_S, abs=1e-3)
    assert played["ped_6"].end == pytest.approx(8.0, abs=1e-3)
    assert max(track.end for track in played.values()) <= 8.0
    entered = played["ped_3"].state(0.0)
    assert (entered.x, entered.y) == pytest.approx(((LATE_START_S - 9.0), 4.0), abs=1e-3)
    assert next(entry for entry in scenario["humans"] if entry["name"] == "ped_3")["pose"][:2] == pytest.approx([LATE_START_S - 9.0, 4.0], abs=1e-3)
    left = played["ped_6"].state(played["ped_6"].end)
    assert (left.x, left.y) == pytest.approx((LATE_START_S + 8.0 - PED_6_ENTERS_S, 6.0), abs=1e-3)

    plain = ETHUCY(cache=importer.cache / DATASETS).import_item("eth", tmp_path / "plain")
    for name, track in played.items():
        recorded = Track.load(plain / f"{name}.csv")
        for t in np.linspace(track.start, track.end, 23):
            here, there = track.state(float(t)), recorded.state(min(max(LATE_START_S + float(t), recorded.start), recorded.end))
            assert (here.x, here.y) == pytest.approx((there.x, there.y), abs=2e-3), (name, t)
            assert _turn(here.yaw, there.yaw) < 1e-3, (name, t)
            assert (here.vx, here.vy) == pytest.approx((there.vx, there.vy), abs=0.05), (name, t)


def test_range_from_zero_starts_with_the_sequence_and_drops_its_last_pedestrian(importer, tmp_path: Path) -> None:
    import yaml
    from arena_simulation_setup.shared.track import Track

    target = importer.import_item("t_all", tmp_path / "worlds")
    scenario = yaml.safe_load((target / "scenario.yaml").read_text())
    assert [entry["name"] for entry in scenario["humans"]] == ["ped_1", "ped_2", "ped_3", "ped_4", "ped_6", "ped_7"]
    assert Track.load(target / "ped_1.csv").start == 0.0
    assert Track.load(target / "ped_3.csv").start == pytest.approx((915 - ETH_FIRST_FRAME) / 15)


def test_scenario_loads_natively_with_the_robot_start_and_goal(importer, tmp_path: Path) -> None:
    import yaml
    from arena_simulation_setup.tree.World import WorldIdentifier
    from arena_simulation_setup.shared.task import GoToPhase

    out = tmp_path / "worlds"
    target = importer.import_item("t_late", out)
    view = _view(out, "socnavbench_eth")
    assert list(view.load(validate=True).levels) == ["0"]
    assert [scenario.name for scenario in view.scenario.listall()] == ["t_late"]
    scenario = view.scenario("t_late").resolve_sync().load()
    assert [obstacle.name for obstacle in scenario.humans] == ["ped_2", "ped_3", "ped_4", "ped_6"]
    (robot,) = scenario.robots
    (phase,) = robot.phase_list()
    assert isinstance(phase, GoToPhase)
    assert (robot.start.position.x, robot.start.position.y) == pytest.approx((13.4, 3.6))
    assert _turn(_yaw(robot.start), 1.5 + math.pi / 2) < 1e-3
    assert (phase.pose.position.x, phase.pose.position.y) == pytest.approx((-1.1, 11.1))
    assert _turn(_yaw(phase.pose), math.pi / 2) < 1e-3

    native_view = WorldIdentifier(NATIVE_WORLD).resolve_sync().scenario(NATIVE_SCENARIO).resolve_sync()
    native = yaml.safe_load(Path(native_view.scenario_path).read_text())
    imported = yaml.safe_load((target / "scenario.yaml").read_text())
    assert set(imported) == {"format", "humans", "robots"}
    assert [set(robot) for robot in imported["robots"]] == [set(native["robots"][0])]
    assert set(imported["robots"][0]["phases"][0]) == set(native["robots"][0]["phases"][0])
    for entry in imported["humans"]:
        assert set(entry) == set(native["humans"][0]) | {"velocity", "waypoint_mode", "stays", "track"}


def test_floor_holds_every_start_and_goal_with_room_for_the_robot(importer, tmp_path: Path) -> None:
    import shapely

    from task_generator.adapters.eth_ucy import FOOTPRINT_M, SEQUENCES, in_meters, parse_sequence, scene_items, walked_area
    from task_generator.adapters.socnavbench import DATASETS

    out = tmp_path / "worlds"
    importer.import_item("t_cross", out)
    importer.import_item("t_late", out)
    zara_x, zara_y = in_meters(SEQUENCES["zara01"].to_meters, np.array([1.5, -5.5, -10.5, 8.0]), np.array([-5.0, 0.0, -10.0, -5.0]))
    for scene, expected in (("zara", list(zip(zara_x, zara_y, strict=True))), ("eth", [(9.4, 6.6), (-0.6, 10.6), (13.4, 3.6), (-1.1, 11.1)])):
        stops = importer.stops(scene)
        assert stops == [pytest.approx(stop) for stop in expected]
        floor, walls = _floor(_level(_view(out, f"socnavbench_{scene}")))
        bodies = [body for item in scene_items(scene) for body in parse_sequence(importer.cache / DATASETS / SEQUENCES[item].file, SEQUENCES[item].fps, SEQUENCES[item].shift, SEQUENCES[item].to_meters)]
        walked = walked_area(bodies)
        assert not all(walked.contains(shapely.Point(stop)) for stop in stops)
        for stop in stops:
            assert floor.contains(shapely.Point(stop)), (scene, stop)
            assert walls.distance(shapely.Point(stop)) >= FOOTPRINT_M - 0.06, (scene, stop)
        for body in bodies:
            path = shapely.LineString(np.column_stack([body.x, body.y]))
            assert floor.buffer(1e-6).covers(path), (scene, body.name)
            assert walls.distance(path) >= FOOTPRINT_M - 0.06, (scene, body.name)


def test_floor_of_a_scene_is_the_same_whichever_episode_comes_first(tmp_path: Path) -> None:
    from task_generator.adapters.socnavbench import SocNavBench

    worlds = []
    for order in (("t_cross", "t_shifted"), ("t_shifted", "t_cross")):
        importer = SocNavBench(cache=seed(tmp_path / "-".join(order) / "cache"))
        out = tmp_path / "-".join(order) / "worlds"
        for item in order:
            importer.import_item(item, out)
            worlds.append((out / "socnavbench_zara" / "world.yaml").read_bytes())
        assert sorted(scenario.name for scenario in _view(out, "socnavbench_zara").scenario.listall()) == ["t_cross", "t_shifted"]
        assert not (out / "eth_ucy_zara").exists()
    assert len(set(worlds)) == 1


def test_attribution_names_the_dataset_authors_and_the_episode_authors(importer, tmp_path: Path) -> None:
    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    from task_generator.adapters.eth_ucy import REVISION

    out = tmp_path / "worlds"
    for item, world, authors, paper in (("t_late", "socnavbench_eth", "Pellegrini", "ICCV 2009"), ("t_shifted", "socnavbench_zara", "Lerner", "Crowds by Example")):
        target = importer.import_item(item, out)
        for path in (target, out / world):
            attribution = read_attribution(path / ATTRIBUTION_FILE)
            assert attribution.url == f"https://github.com/CMU-TBD/SocNavBench/blob/{REVISION}/params/episode_params_val.ini"
            assert attribution.license == "MIT for the episodes, unspecified for the trajectories"
            assert authors in attribution.authors and "Gustavo Silvera" in attribution.authors
            assert paper in attribution.citation and "SocNavBench: A Grounded Simulation Testing Framework" in attribution.citation


def test_cli_lists_and_imports_an_episode_where_the_resolver_looks(tmp_path: Path) -> None:
    cache = seed(tmp_path / "cache")
    env = {**os.environ, "ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "imported")}
    command = [sys.executable, "-m", "arena_assets", "import", "scenario", "socnavbench"]
    listed = subprocess.run([*command, "--cache", str(cache)], env=env, capture_output=True, text=True, check=True)
    assert listed.stdout.splitlines()[: len(IMPORTED)] == IMPORTED
    target = tmp_path / "imported" / "worlds" / "socnavbench_eth"
    imported = subprocess.run([*command, "t_late", "--cache", str(cache)], env=env, capture_output=True, text=True, check=True)
    assert imported.stdout.splitlines()[-1] == str(target / "scenarios" / "t_late")
    code = "from arena_simulation_setup.tree.World import WorldIdentifier\nview = WorldIdentifier('socnavbench_eth').resolve_sync()\nprint(view.path)\nrobot = view.scenario('t_late').resolve_sync().load().robots[0]\nprint(robot.start.position.x, robot.phase_list()[0].pose.position.y)"
    resolved = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert resolved.stdout.splitlines() == [str(target), "13.4 11.1"]
    refused = subprocess.run([*command, "t_hotel", "--cache", str(cache)], env=env, capture_output=True, text=True)
    assert refused.returncode != 0
    assert "SocNavBench episode 't_hotel' is left out: SocNavBench plays hotel at 15 fps although hotel was recorded at 25 fps" in " ".join(re.sub(r"\x1b\[[0-9;]*m", "", refused.stderr).replace("\u2502", " ").split())


def test_imported_episode_plays_on_the_bus_from_the_window_start(rclpy_context, tmp_path: Path) -> None:
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")
    from arena_simulation_setup.shared.track import Track
    from arena_simulation_setup.tree import SimplePathResolver
    from arena_simulation_setup.tree.World import WorldIdentifier

    from task_generator.adapters.socnavbench import SocNavBench

    replay = _replay_stage()
    out = tmp_path / "worlds"
    target = SocNavBench(cache=seed(tmp_path / "cache")).import_item("t_late", out)
    tracks = {name: Track.load(target / f"{name}.csv") for name in ("ped_2", "ped_3", "ped_4", "ped_6")}

    def expected(t: float) -> dict[str, tuple[float, float]]:
        states = {name: track.state(t) for name, track in tracks.items()}
        return {name: (state.x + replay.OFFSET[0], state.y + replay.OFFSET[1]) for name, state in states.items() if state is not None}

    async def body(stage) -> None:
        identifier = WorldIdentifier("socnavbench_eth")
        view = identifier.load(await SimplePathResolver(WorldIdentifier, out).resolve(identifier))
        scenario = view.scenario("t_late").resolve_sync().load()
        assert len(scenario.robots) == 1
        await stage.play(scenario.humans)
        assert sorted(expected(1.03)) == ["ped_2", "ped_3"]
        peds = await stage.at(1.03, expected(1.03))
        assert peds["ped_3"].pose.position.x == pytest.approx(LATE_START_S + 1.03 - 9.0 + replay.OFFSET[0], abs=5e-3)
        assert sorted(expected(3.02)) == ["ped_2", "ped_3", "ped_4"]
        await stage.at(3.02, expected(3.02))
        assert sorted(expected(6.52)) == ["ped_4", "ped_6"]
        peds = await stage.at(6.52, expected(6.52))
        assert peds["ped_6"].pose.position.x == pytest.approx(LATE_START_S + 6.52 - PED_6_ENTERS_S + replay.OFFSET[0], abs=5e-3)
        await stage.at(8.5, {})
        seen = {ped.name for batch in stage.bus for ped in batch.pedestrians}
        assert seen == {stage.realizer.prefix(name) for name in tracks}

    replay._on_stage(body)


@pytest.mark.network
def test_live_episodes_select_the_published_windows_on_floors_that_hold_every_start_and_goal(tmp_path: Path) -> None:
    import shapely

    from task_generator.adapters.eth_ucy import FOOTPRINT_M, SEQUENCES, parse_sequence, read_rows, scene_items, walked_area
    from task_generator.adapters.socnavbench import DATASETS, SocNavBench, left_out, select, world_name

    importer = SocNavBench(cache=tmp_path / "cache")
    assert len(importer.episodes) == 33
    assert list(importer.items()) == list(LIVE)
    reasons = {name: left_out(episode, importer.placements) for name, episode in importer.episodes.items() if name not in LIVE}
    assert len(reasons) == 16
    assert all("at 15 fps although" in reason and "recorded at 25 fps" in reason for reason in reasons.values())
    assert sorted(name for name, reason in reasons.items() if "mirror image of hotel" in reason) == sorted(name for name in reasons if "hotel" in name)
    for item, (world, start, count, max_time) in LIVE.items():
        src = importer.fetch(item)
        episode, placement, sequence = importer.episode(item), importer.placement(item), importer.sequence(item)
        frames, idents, _, _ = read_rows(src / DATASETS / sequence.file)
        start_frame, chosen = select(frames.astype(int), idents.astype(int), episode.ped_ranges[0], placement.fps, episode.max_time)
        assert world_name(sequence.scene) == world, item
        assert (start_frame - frames.min()) / sequence.fps == pytest.approx(start, abs=1e-6), item
        assert len(chosen) == count, item
        assert episode.max_time == max_time, item
    assert importer.placements["eth"].from_map(importer.episodes["t_eth_dense_against"].start) == pytest.approx((13.4, 3.6, 1.5 + math.pi / 2))
    for scene in ("eth", "zara"):
        bodies = [body for other in scene_items(scene) for body in parse_sequence(importer.cache / DATASETS / SEQUENCES[other].file, SEQUENCES[other].fps, SEQUENCES[other].shift, SEQUENCES[other].to_meters)]
        stops = importer.stops(scene)
        assert len(stops) == 2 * sum(1 for world, *_ in LIVE.values() if world == world_name(scene))
        floor = walked_area(bodies, stops)
        assert not all(walked_area(bodies).contains(shapely.Point(stop)) for stop in stops), scene
        for stop in stops:
            assert floor.contains(shapely.Point(stop)), (scene, stop)
            assert floor.exterior.distance(shapely.Point(stop)) >= FOOTPRINT_M - 1e-9, (scene, stop)
        walked = shapely.union_all([shapely.LineString(np.column_stack([body.x, body.y])) for body in bodies])
        assert floor.covers(walked), scene
        assert floor.exterior.distance(walked) >= FOOTPRINT_M - 1e-9, scene


@pytest.mark.network
def test_live_episode_imports_with_every_selected_pedestrian(tmp_path: Path) -> None:
    import shapely
    from arena_simulation_setup.shared.track import Track

    from task_generator.adapters.eth_ucy import SEQUENCES, in_meters
    from task_generator.adapters.socnavbench import SocNavBench

    out = tmp_path / "worlds"
    target = SocNavBench(cache=tmp_path / "cache").import_item("t_zara2_wrap", out)
    start_x, start_y = in_meters(SEQUENCES["zara02"].to_meters, np.array([-10.5]), np.array([-10.0]))
    view = _view(out, "socnavbench_zara")
    scenario = view.scenario("t_zara2_wrap").resolve_sync().load()
    assert len(scenario.humans) == LIVE["t_zara2_wrap"][2]
    assert (scenario.robots[0].start.position.x, scenario.robots[0].start.position.y) == pytest.approx((start_x[0], start_y[0]), abs=1e-3)
    played = [Track.load(target / f"{obstacle.name}.csv") for obstacle in scenario.humans]
    assert sum(1 for track in played if track.start == 0.0) == 3
    assert max(track.end for track in played) == pytest.approx(60.0)
    floor, _ = _floor(_level(view))
    for track in played:
        assert floor.buffer(1e-6).covers(shapely.LineString(np.column_stack([track.x, track.y])))
