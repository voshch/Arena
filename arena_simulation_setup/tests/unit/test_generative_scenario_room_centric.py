from __future__ import annotations

import math
from pathlib import Path

import pytest
from arena_simulation_setup.generative.scenario.populate import build, MIN_ROUTE_CLEARANCE, robot_zones, route_points, ROUTE_STOPS, TALKERS
from arena_simulation_setup.generative.scenario.zone_screen import ref_safe_zones, screen_zones
from arena_simulation_setup.generative.scenario.populate import MIN_PED_ROUTE_M
from arena_simulation_setup.generative.core.bodies import RUNTIME_SAFE_DIST


def zone(name, x0, y0, x1, y1):
    return {
        "name": name,
        "corners": [
            {"x": x0, "y": y0},
            {"x": x1, "y": y0},
            {"x": x1, "y": y1},
            {"x": x0, "y": y1},
        ],
    }


@pytest.fixture
def world(tmp_path: Path) -> Path:
    import yaml

    root = tmp_path / "w"
    root.mkdir(parents=True)
    (root / "world.yaml").write_text(yaml.safe_dump({"format": "arena-world/2", "levels": [{"id": "0", "zones": [{"name": "box", "polygon": [[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]], "floor": False}]}]}))
    return root


ZONES = [
    zone("open_work_area", 2.0, 2.0, 12.0, 12.0),
    zone("kitchen", 16.0, 2.0, 26.0, 12.0),
    zone("main_hallway", 2.0, 14.0, 26.0, 18.0),
]

ROOMY = [
    zone("open_work_area", 2.0, 2.0, 12.0, 12.0),
    zone("meeting_room_1", 16.0, 2.0, 22.0, 8.0),
    zone("meeting_room_2", 16.0, 10.0, 22.0, 16.0),
    zone("reception", 2.0, 20.0, 10.0, 26.0),
    zone("kitchen", 16.0, 20.0, 22.0, 26.0),
    zone("main_hallway", 12.5, 2.0, 15.5, 26.0),
]
ROOMY_CENTROIDS = {z["name"]: ((z["corners"][0]["x"] + z["corners"][2]["x"]) / 2, (z["corners"][0]["y"] + z["corners"][2]["y"]) / 2) for z in ROOMY}

ROBOT = {"start": [4.0, 16.0, 0.0], "goal": [24.0, 16.0, 0.0]}
ROUTE = [(4.0, 16.0), (24.0, 16.0)]
RESOLVABLE = {"open_work_area": 5000, "kitchen": 5000, "main_hallway": 4000}


def test_a_zone_clear_of_the_route_is_a_home(world) -> None:
    verdicts = screen_zones(
        ZONES,
        world_root=world,
        route=[(28.0, 1.0), (29.0, 1.0)],
        min_route_clearance=MIN_ROUTE_CLEARANCE,
    )
    assert ref_safe_zones(verdicts) == {"open_work_area", "kitchen", "main_hallway"}


def test_a_zone_the_route_runs_through_is_not(world) -> None:
    verdicts = screen_zones(
        ZONES,
        world_root=world,
        route=[(4.0, 7.0), (10.0, 7.0)],
        min_route_clearance=MIN_ROUTE_CLEARANCE,
    )
    assert "open_work_area" not in ref_safe_zones(verdicts)
    assert "robot's route" in verdicts["open_work_area"].reason
    assert verdicts["open_work_area"].rejected > 0
    assert "kitchen" in ref_safe_zones(verdicts)


def test_furniture_the_map_does_not_show_still_counts(world) -> None:
    bare = screen_zones(
        ZONES,
        world_root=world,
        route=[(28.0, 1.0), (29.0, 1.0)],
        min_route_clearance=MIN_ROUTE_CLEARANCE,
    )
    furnished = screen_zones(
        ZONES,
        world_root=world,
        route=[(28.0, 1.0), (29.0, 1.0)],
        statics=[{"name": "desk", "pose": [7.0, 7.0], "radius": 1.5}],
        min_route_clearance=MIN_ROUTE_CLEARANCE,
    )
    assert furnished["open_work_area"].admissible < bare["open_work_area"].admissible
    assert furnished["kitchen"].admissible == bare["kitchen"].admissible


def test_a_zone_smaller_than_the_runtime_disc_resolves_nowhere(world) -> None:
    tiny = [zone("cupboard", 5.0, 5.0, 5.0 + RUNTIME_SAFE_DIST, 5.0 + RUNTIME_SAFE_DIST)]
    verdicts = screen_zones(
        tiny,
        world_root=world,
        route=[(28.0, 1.0), (29.0, 1.0)],
        statics=[{"name": "crate", "pose": [5.1, 5.1], "radius": 1.0}],
        min_route_clearance=MIN_ROUTE_CLEARANCE,
    )
    assert not verdicts["cupboard"].ref_safe
    assert verdicts["cupboard"].admissible == 0
    assert "SAFE_DIST" in verdicts["cupboard"].reason


def test_an_unscreenable_world_refuses_every_zone(tmp_path) -> None:
    verdicts = screen_zones(ZONES, world_root=tmp_path, route=[], min_route_clearance=MIN_ROUTE_CLEARANCE)
    assert ref_safe_zones(verdicts) == frozenset()
    assert all("no occupancy grid" in v.reason for v in verdicts.values())


def test_a_zone_the_robot_is_named_into_is_never_a_home(world) -> None:
    verdicts = screen_zones(
        ZONES,
        world_root=world,
        route=[(28.0, 1.0), (29.0, 1.0)],
        min_route_clearance=MIN_ROUTE_CLEARANCE,
        robot_zones=frozenset({"kitchen"}),
    )
    assert "kitchen" not in ref_safe_zones(verdicts)
    assert "endpoint" in verdicts["kitchen"].reason
    assert verdicts["kitchen"].admissible > 0


def test_a_zone_named_robot_still_produces_a_route() -> None:
    named = {"start": "reception", "goal": "open_work_area"}
    zones = [*ZONES, zone("reception", 2.0, 20.0, 8.0, 24.0)]
    assert route_points(named, zones) == [(5.0, 22.0), (7.0, 7.0)]
    assert route_points(named) == []
    assert robot_zones(named) == {"reception", "open_work_area"}


def _population(world="w", profile="normal", seed=0, zones=None, homes=None, robot=ROBOT):
    zones = ROOMY if zones is None else zones
    if homes is None:
        homes = frozenset(z["name"] for z in zones if z["name"] != "main_hallway")
    return build(profile, zones=zones, robot=robot, seed=seed, homes=homes, world=world)["humans"]


def test_every_pose_and_waypoint_is_a_room() -> None:
    for profile in ("normal", "break", "night"):
        for agent in _population(profile=profile):
            assert isinstance(agent["pose"], str)
            assert agent["waypoints"] and all(isinstance(w, str) for w in agent["waypoints"])
            assert agent["pose"] in ROOMY_CENTROIDS


def test_no_agent_starts_in_a_zone_the_robot_is_named_into() -> None:
    named = {"start": "reception", "goal": "open_work_area"}
    for agent in _population(robot=named, homes=frozenset()):
        assert agent["pose"] not in ("reception", "open_work_area")


def test_homes_come_from_the_screen_when_it_found_any() -> None:
    only = frozenset({"meeting_room_2"})
    assert {a["pose"] for a in _population(profile="normal", homes=only)} == only


def test_walkers_go_somewhere_else_and_never_repeat_a_room() -> None:
    for agent in _population(profile="normal", seed=3):
        if agent["agent"]["agent_type"] == "adult":
            waypoints = agent["waypoints"]
            assert len(waypoints) >= 2
            assert waypoints[1] != waypoints[0]
            for before, after in zip(waypoints, waypoints[1:], strict=False):
                assert before != after


def test_walkers_cross_the_corridor() -> None:
    routes = [a["waypoints"] for a in _population(profile="normal", seed=5) if a["agent"]["agent_type"] == "adult"]
    assert any("main_hallway" in r for r in routes)
    assert not any(r[0] == "main_hallway" for r in routes)


def test_walks_reach_the_route_floor_where_the_world_allows() -> None:
    def length(agent):
        pts = [ROOMY_CENTROIDS[w] for w in agent["waypoints"]]
        return sum(math.dist(a, b) for a, b in zip(pts, pts[1:], strict=False))

    walks = [length(a) for a in _population(profile="normal", seed=2) if a["agent"]["agent_type"] == "adult"]
    assert walks
    assert min(walks) >= MIN_PED_ROUTE_M * 0.8


def test_a_single_room_world_walks_between_two_points_of_the_room() -> None:
    one = [zone("open_work_area", 0.0, 0.0, 10.0, 10.0)]
    doc = build("normal", zones=one, robot={"start": [-5.0, -5.0, 0.0], "goal": [-4.0, -5.0, 0.0]}, seed=2, world="w")
    assert doc["humans"]
    for agent in doc["humans"]:
        assert agent["waypoints"] == ["open_work_area", "open_work_area"]
        assert agent["waypoint_mode"] == "reverse"


def test_talkers_come_in_pairs_per_room_and_everyone_else_walks() -> None:
    agents = _population(profile="normal", seed=1)
    talkers = [a for a in agents if a["agent"]["agent_type"].startswith("./")]
    walkers = [a for a in agents if a["agent"]["agent_type"] == "adult"]
    assert talkers and len(talkers) % TALKERS == 0
    rooms: dict[str, int] = {}
    for t in talkers:
        rooms[t["pose"]] = rooms.get(t["pose"], 0) + 1
    assert all(n == TALKERS for n in rooms.values())
    assert walkers
    for w in walkers:
        assert len(w["waypoints"]) >= 2 and w["waypoints"][-1] != w["waypoints"][0]


def test_a_tiny_world_still_gets_a_walking_cast() -> None:
    from arena_simulation_setup.generative.scenario.populate import MIN_CAST, MIN_WALKERS

    two = [zone("open_work_area", 0.0, 0.0, 6.0, 6.0), zone("kitchen", 8.0, 0.0, 12.0, 4.0)]
    doc = build("night", zones=two, robot={"start": [20.0, 20.0, 0.0], "goal": [21.0, 20.0, 0.0]}, seed=0, world="tiny")
    agents = doc["humans"]
    assert len(agents) == MIN_CAST
    walkers = [a for a in agents if not a["agent"]["agent_type"].startswith("./")]
    assert len(walkers) >= MIN_WALKERS


def test_once_is_never_emitted() -> None:
    for profile in ("normal", "break", "night"):
        for agent in _population(profile=profile, seed=4):
            assert agent["waypoint_mode"] != "once"


def test_the_mode_suits_the_route() -> None:
    for agent in _population(profile="normal", seed=8):
        mode, legs = agent["waypoint_mode"], len(agent["waypoints"])
        assert mode in {"reverse", "repeat", "random"}
        if legs < 3:
            assert mode == "reverse"


def test_routes_are_longer_than_a_single_out_and_back() -> None:
    counts = {len(a["waypoints"]) for a in _population(profile="normal", seed=5)}
    assert counts - {2}
    assert max(counts) <= 1 + max(s for s, _ in ROUTE_STOPS) + 2


def test_a_populated_profile_is_never_left_empty() -> None:
    tight = [
        zone("living_room", 0.0, 0.0, 6.0, 6.0),
        zone("kitchen", 0.0, 7.0, 2.0, 9.0),
        zone("bathroom", 3.0, 7.0, 4.2, 8.2),
    ]
    for profile in ("normal", "break", "night"):
        doc = build(profile, zones=tight, robot={"start": [20.0, 20.0, 0.0], "goal": [21.0, 20.0, 0.0]}, seed=1, world="flat")
        assert doc["humans"], f"{profile} came out empty"


def test_generation_is_byte_identical_for_the_same_inputs() -> None:
    assert _population(seed=7) == _population(seed=7)


def test_two_worlds_at_the_same_seed_are_not_the_same_crowd() -> None:
    a, b = _population(world="hospital_1"), _population(world="office_1")
    assert [x["agent"]["desired_velocity"] for x in a] != [x["agent"]["desired_velocity"] for x in b]


def test_a_worlds_character_survives_regenerating_one_profile() -> None:
    fast = _population(world="hospital_1", profile="normal")
    slow = _population(world="office_1", profile="normal")
    assert max(a["agent"]["desired_velocity"] for a in fast) != max(a["agent"]["desired_velocity"] for a in slow)
