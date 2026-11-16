"""Unit tests for procedural concourse world generator and dock socket scenarios."""

from __future__ import annotations

import math
import random
from pathlib import Path
from types import ModuleType

import numpy as np
import pydantic
import pytest
import shapely
import yaml
from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.concourse import (
    ConcourseConfiguration,
    ElevatorBayFeature,
    FlowStreamSpec,
    SideBranchFeature,
    WorldGeneratorConcourse,
)
from arena_simulation_setup.tree.World import World
from arena_simulation_setup.tree.World.Scenario import ScenarioView


def test_concourse_registered_in_world_generator_registry() -> None:
    """Verify WorldGeneratorType.CONCOURSE is registered and constructs via WorldGenerator."""
    assert WorldGeneratorType.CONCOURSE in WorldGenerator.available()
    gen = WorldGenerator(WorldGeneratorType.CONCOURSE, {"length": 24.0, "width": 4.0}, seed=42)
    world = gen.compute()
    assert len(world.all_levels) == 1
    assert len(world.all_levels[0].zones) >= 1


def test_concourse_params_contract() -> None:
    """Verify concourse params() binds robots and obstacles to the generated scenario file."""
    gen = WorldGeneratorConcourse({"length": 24.0, "width": 4.0}, random.Random(42))
    assert gen.params() == {
        "modes": {"robots": "scenario", "obstacles": "scenario"},
        "params": {
            "robots": {"file": "default"},
            "obstacles": {"file": "default"},
        },
    }


def test_concourse_straight_default_synthesizes_docks_and_scenario() -> None:
    """Verify default straight concourse synthesis and scenario emission with dock sockets."""
    rng = random.Random(42)
    gen = WorldGeneratorConcourse({"length": 30.0, "width": 4.0, "shape": "straight"}, rng)
    world = gen.compute()

    level = world.all_levels[0]
    zone_names = {z.name for z in level.zones}
    assert "corridor_seg_0" in zone_names
    assert "dock_start" in zone_names
    assert "dock_end" in zone_names

    files = gen.files()
    assert "scenarios/default/scenario.yaml" in files
    scenario = yaml.safe_load(files["scenarios/default/scenario.yaml"])
    assert scenario["format"] == "arena-scenario/2"
    assert len(scenario["robots"]) == 1
    assert "dock_start_sink" in scenario["regions"]
    assert "dock_end_source_main_opposing" in scenario["regions"]
    assert scenario["regions"]["dock_start_sink"]["ref"] == "dock_start"
    assert scenario["regions"]["dock_end_source_main_opposing"]["ref"] == "dock_end"


def test_concourse_smooth_s_bend_and_l_bend_geometries() -> None:
    """Verify C2 smooth S-bend and quadratic Bezier smooth L-bend spine geometries."""
    rng = random.Random(42)
    s_gen = WorldGeneratorConcourse({"length": 36.0, "width": 4.5, "shape": "smooth_s_bend"}, rng)
    s_world = s_gen.compute()
    s_corr = next(z for z in s_world.all_levels[0].zones if z.name == "corridor_seg_0")
    assert len(s_corr.polygon) > 10

    l_gen = WorldGeneratorConcourse(
        {"length": 30.0, "width": 4.0, "shape": "smooth_l_bend", "bend_angle_deg": 60.0},
        rng,
    )
    l_world = l_gen.compute()
    l_corr = next(z for z in l_world.all_levels[0].zones if z.name == "corridor_seg_0")
    assert len(l_corr.polygon) > 10


def test_concourse_angled_cross_branches_create_dock_sockets() -> None:
    """Verify 60-degree bilateral branch synthesis partitions spine and attaches dock sockets."""
    rng = random.Random(42)
    config = {
        "length": 35.0,
        "width": 4.0,
        "branches": [
            {
                "at_s": 17.5,
                "side": "both",
                "angle_deg": 60.0,
                "width": 3.0,
                "length": 6.0,
            }
        ],
    }
    gen = WorldGeneratorConcourse(config, rng)
    world = gen.compute()
    zone_names = {z.name for z in world.all_levels[0].zones}
    assert "intersection_0" in zone_names
    assert "dock_branch_0_left" in zone_names
    assert "dock_branch_0_right" in zone_names

    scenario = yaml.safe_load(gen.files()["scenarios/default/scenario.yaml"])
    assert "dock_branch_0_right_sink" in scenario["regions"]
    assert "dock_branch_0_left_source_cross_flow" in scenario["regions"]


def test_concourse_custom_flow_distribution_with_blind_and_aware_peds() -> None:
    """Verify custom flow distribution with diverse agent types, speeds, and behavior trees."""
    rng = random.Random(42)
    config = {
        "length": 30.0,
        "width": 4.0,
        "flows": [
            FlowStreamSpec(
                name="aware_commuters",
                source_dock="dock_start",
                sink_dock="dock_end",
                rate=0.8,
                agent_type="adult",
                velocity_min=1.2,
                velocity_max=1.6,
                behavior_tree="social_aware",
                active_window=(10.0, 50.0),
            ).model_dump(),
            FlowStreamSpec(
                name="blind_distracted",
                source_dock="dock_end",
                sink_dock="dock_start",
                rate=0.3,
                agent_type="blind_pedestrian",
                velocity_min=0.8,
                velocity_max=1.1,
                behavior_tree="distracted_inattentive",
            ).model_dump(),
        ],
    }
    gen = WorldGeneratorConcourse(config, rng)
    _ = gen.compute()
    scenario = yaml.safe_load(gen.files()["scenarios/default/scenario.yaml"])

    aware_src = scenario["regions"]["dock_start_source_aware_commuters"]
    assert aware_src["config"]["agent"]["agent_type"] == "adult"
    assert aware_src["config"]["agent"]["behavior_tree"] == "social_aware"
    assert aware_src["config"]["agent"]["desired_velocity"]["min"] == 1.2
    assert aware_src["config"]["rate_profile"] == [
        {"t": 0.0, "rate": 0.0},
        {"t": 10.0, "rate": 0.0},
        {"t": 10.0, "rate": 0.8},
        {"t": 50.0, "rate": 0.8},
        {"t": 50.0, "rate": 0.0},
    ]

    blind_src = scenario["regions"]["dock_end_source_blind_distracted"]
    assert blind_src["config"]["agent"]["agent_type"] == "blind_pedestrian"
    assert blind_src["config"]["agent"]["behavior_tree"] == "distracted_inattentive"
    assert blind_src["config"]["agent"]["desired_velocity"]["max"] == 1.1


def _interpolate_scheduler_rate(profile: list[dict[str, float]], t: float) -> float:
    """Evaluate spawn rate at time t according to Arena spawn_scheduler interpolation rules.

    Mirrors spawn_scheduler.py:117-135 (arena_humansim.core.spawn_scheduler.SpawnScheduler._interpolate_rate).
    """
    if not profile:
        return 0.0
    if len(profile) == 1:
        return profile[0]["rate"]
    if t <= profile[0]["t"]:
        return profile[0]["rate"]
    if t >= profile[-1]["t"]:
        return profile[-1]["rate"]

    for i in range(len(profile) - 1):
        if profile[i]["t"] <= t <= profile[i + 1]["t"]:
            dt = profile[i + 1]["t"] - profile[i]["t"]
            if dt <= 0:
                return profile[i + 1]["rate"]
            alpha = (t - profile[i]["t"]) / dt
            return profile[i]["rate"] + alpha * (profile[i + 1]["rate"] - profile[i]["rate"])

    return profile[-1]["rate"]


def test_concourse_active_window_rate_profile_interpolation() -> None:
    """Verify active_window keyframe step construction accurately models discrete windows."""
    gen = WorldGeneratorConcourse({"length": 24.0}, random.Random(42))

    # Case 1: Window from t=10.0 to t=50.0
    spec_window = FlowStreamSpec(name="f1", rate=0.8, active_window=(10.0, 50.0))
    p1 = gen._compile_rate_profile(spec_window)
    assert _interpolate_scheduler_rate(p1, 0.0) == 0.0
    assert _interpolate_scheduler_rate(p1, 5.0) == 0.0
    assert _interpolate_scheduler_rate(p1, 9.99) == 0.0
    assert _interpolate_scheduler_rate(p1, 10.01) == 0.8
    assert _interpolate_scheduler_rate(p1, 30.0) == 0.8
    assert _interpolate_scheduler_rate(p1, 49.99) == 0.8
    assert _interpolate_scheduler_rate(p1, 50.0) == 0.0
    assert _interpolate_scheduler_rate(p1, 60.0) == 0.0

    # Case 2: Window from t=0.0 to t=40.0
    spec_from_zero = FlowStreamSpec(name="f2", rate=0.5, active_window=(0.0, 40.0))
    p2 = gen._compile_rate_profile(spec_from_zero)
    assert _interpolate_scheduler_rate(p2, 0.0) == 0.5
    assert _interpolate_scheduler_rate(p2, 20.0) == 0.5
    assert _interpolate_scheduler_rate(p2, 39.99) == 0.5
    assert _interpolate_scheduler_rate(p2, 40.0) == 0.0
    assert _interpolate_scheduler_rate(p2, 50.0) == 0.0

    # Case 3: Window starting at t=15.0 extending to episode end
    spec_delayed_start = FlowStreamSpec(name="f3", rate=0.6, active_window=(15.0, 999999.0))
    p3 = gen._compile_rate_profile(spec_delayed_start)
    assert _interpolate_scheduler_rate(p3, 0.0) == 0.0
    assert _interpolate_scheduler_rate(p3, 10.0) == 0.0
    assert _interpolate_scheduler_rate(p3, 15.01) == 0.6
    assert _interpolate_scheduler_rate(p3, 100.0) == 0.6


def test_concourse_attached_room_with_door_and_enter_room_mission() -> None:
    """Verify enclosed room branch creates sliding Door and routes robot inside."""
    rng = random.Random(42)
    config = {
        "length": 30.0,
        "width": 4.0,
        "branches": [
            {
                "at_s": 15.0,
                "side": "left",
                "width": 5.0,
                "length": 6.0,
                "is_room": True,
                "door_width": 1.2,
            }
        ],
        "robot_task_mode": "enter_room",
    }
    gen = WorldGeneratorConcourse(config, rng)
    world = gen.compute()
    level = world.all_levels[0]

    assert len(level.doors) == 1
    door = level.doors[0]
    assert door.name == "door_branch_0_left"
    door_len = math.hypot(door.end.x - door.start.x, door.end.y - door.start.y)
    assert round(door_len, 2) == 1.2

    # Verify door endpoints lie on wall boundary
    walls_lines = [shapely.LineString([(p.x, p.y) for p in w.path]) for w in level.walls]
    walls_union = shapely.ops.unary_union(walls_lines)
    p_start = shapely.Point(door.start.x, door.start.y)
    p_end = shapely.Point(door.end.x, door.end.y)
    assert walls_union.distance(p_start) < 0.2
    assert walls_union.distance(p_end) < 0.2

    scenario = yaml.safe_load(gen.files()["scenarios/default/scenario.yaml"])
    robot = scenario["robots"][0]
    assert robot["phases"][0]["goto"][0] == 15.0
    assert robot["phases"][0]["goto"][1] > 4.0


def test_concourse_runtime_scenario_loading_with_zone_converter(tmp_path: Path) -> None:
    """Verify generated scenario loads successfully via ScenarioView without mocks."""
    rng = random.Random(42)
    gen = WorldGeneratorConcourse(
        {
            "length": 30.0,
            "width": 4.0,
            "branches": [{"at_s": 15.0, "side": "left", "width": 3.0, "length": 5.0}],
        },
        rng,
    )
    world = gen.compute()
    c = world.compact_world(dict.fromkeys(world.level_ids, (0.0, 0.0))).zone_converter(np.random.default_rng(0))

    scenario_dir = tmp_path / "scenarios" / "default"
    scenario_dir.mkdir(parents=True)
    scenario_file = scenario_dir / "scenario.yaml"
    scenario_file.write_bytes(gen.files()["scenarios/default/scenario.yaml"])

    scenario = ScenarioView(str(scenario_dir)).load(converter=c)
    assert len(scenario.robots) == 1
    assert len(scenario.regions) >= 2


def test_concourse_validation_errors() -> None:
    """Verify boundary checks fail with descriptive ValueErrors."""
    # Door width exceeds branch width
    with pytest.raises(ValueError, match="cannot exceed branch width"):
        SideBranchFeature(at_s=10.0, width=3.0, length=5.0, door_width=4.0)

    # Door width positive lower bound (0.0 rejected)
    with pytest.raises(pydantic.ValidationError):
        SideBranchFeature(at_s=10.0, width=3.0, length=5.0, door_width=0.0)

    # Velocity min exceeds velocity max
    with pytest.raises(ValueError, match="velocity_min .* cannot exceed velocity_max"):
        FlowStreamSpec(velocity_min=2.0, velocity_max=1.0)

    # Equal door width is permitted
    feat = SideBranchFeature(at_s=10.0, width=3.0, length=5.0, door_width=3.0)
    assert feat.door_width == 3.0

    # Branch at_s exceeds nominal length
    with pytest.raises(ValueError, match="exceeds concourse length"):
        ConcourseConfiguration(length=20.0, branches=[SideBranchFeature(at_s=25.0)])

    # Elevator at_s exceeds nominal length
    with pytest.raises(ValueError, match="exceeds concourse length"):
        ConcourseConfiguration(length=20.0, elevators=[ElevatorBayFeature(at_s=25.0)])

    # Elevator away from both spine ends fails on compute()
    with pytest.raises(ValueError, match="must be 0 .spine start. or 24.00 m .spine end."):
        _ = WorldGeneratorConcourse({"length": 24.0, "elevators": [{"at_s": 12.0}]}, random.Random(42)).compute()

    # Elevator door_width exceeds cabin_width
    with pytest.raises(ValueError, match="cannot exceed cabin_width"):
        ElevatorBayFeature(cabin_width=2.0, door_width=2.5)

    # Duplicate flow stream names
    with pytest.raises(ValueError, match="Flow stream names must be unique"):
        ConcourseConfiguration(
            length=20.0,
            flows=[FlowStreamSpec(name="f1"), FlowStreamSpec(name="f1")],
        )

    # Curved-spine actual arc length boundary check on compute()
    with pytest.raises(ValueError, match="exceeds actual spine length"):
        WorldGeneratorConcourse(
            {
                "length": 30.0,
                "shape": "smooth_l_bend",
                "bend_angle_deg": 60.0,
                "branches": [{"at_s": 28.0, "width": 3.0, "length": 4.0}],
            },
            random.Random(42),
        ).compute()

    # Invalid dock reference in flows fails early on compute()
    rng = random.Random(42)
    invalid_flow_gen = WorldGeneratorConcourse(
        {
            "length": 20.0,
            "flows": [FlowStreamSpec(name="f1", source_dock="dock_nonexistent").model_dump()],
        },
        rng,
    )
    with pytest.raises(ValueError, match="not found in dock sockets"):
        _ = invalid_flow_gen.compute()

    # Flow whose source and sink are the same dock fails early on compute()
    same_dock_gen = WorldGeneratorConcourse(
        {
            "length": 20.0,
            "flows": [FlowStreamSpec(name="loop", source_dock="dock_start", sink_dock="dock_start").model_dump()],
        },
        rng,
    )
    with pytest.raises(ValueError, match="source_dock and sink_dock are both 'dock_start'"):
        _ = same_dock_gen.compute()

    # Invalid robot_target_branch fails early on compute()
    invalid_target_gen = WorldGeneratorConcourse(
        {
            "length": 20.0,
            "robot_task_mode": "turn_into_branch",
            "robot_target_branch": "dock_fake_branch",
        },
        rng,
    )
    with pytest.raises(ValueError, match="robot_target_branch 'dock_fake_branch' not found"):
        _ = invalid_target_gen.compute()


def test_concourse_scenario_output_determinism() -> None:
    """Verify files() produces identical byte output and deterministic region order across runs."""
    config = {
        "length": 35.0,
        "width": 4.0,
        "branches": [{"at_s": 15.0, "side": "both", "width": 3.0, "length": 5.0}],
    }
    gen1 = WorldGeneratorConcourse(config, random.Random(1))
    gen2 = WorldGeneratorConcourse(config, random.Random(2))
    files1 = gen1.files()
    files2 = gen2.files()
    assert files1 == files2

    # Verify region keys are sorted deterministically (sinks first, then sources)
    scenario = yaml.safe_load(files1["scenarios/default/scenario.yaml"])
    region_keys = list(scenario["regions"].keys())
    assert region_keys == [
        "dock_branch_0_right_sink",
        "dock_start_sink",
        "dock_end_source_main_opposing",
        "dock_branch_0_left_source_cross_flow",
    ]


def test_concourse_empty_robot_target_branch_normalizes_to_none() -> None:
    """Verify that an empty string for robot_target_branch normalizes to None directly in model."""
    config = ConcourseConfiguration.model_validate({"robot_target_branch": ""})
    assert config.robot_target_branch is None

    # Verify ROS panel configuration computes successfully without dock reference error
    gen = WorldGeneratorConcourse({"robot_target_branch": ""}, random.Random(42))
    world = gen.compute()
    assert len(world.all_levels) == 1
    assert "dock_start" in {z.name for z in world.all_levels[0].zones}


def test_concourse_with_elevator_creates_proper_elevator_entity() -> None:
    """Verify that elevator features instantiate first-class Elevator entities with valid door_side and bounds."""
    config = {
        "length": 24.0,
        "width": 4.0,
        "elevators": [
            {"at_s": 0.0, "cabin_width": 2.5, "cabin_depth": 2.5, "door_width": 1.4},
            {"at_s": 24.0, "cabin_width": 2.5, "cabin_depth": 2.5, "door_width": 1.4},
        ],
    }
    gen = WorldGeneratorConcourse(config, random.Random(42))
    world = gen.compute()
    level = world.all_levels[0]

    assert len(level.elevators) == 2
    elev_start, elev_end = level.elevators[0], level.elevators[1]

    assert elev_start.name == "elevator_0"
    assert elev_start.size == [2.5, 2.5, 2.5]
    # At start (s=0, tangent=(1,0), extrude_dir=(-1,0)), door faces back into concourse (+x)
    assert elev_start.door_side == "+x"

    assert elev_end.name == "elevator_1"
    assert elev_end.size == [2.5, 2.5, 2.5]
    # At end (s=24, tangent=(1,0), extrude_dir=(1,0)), door faces back into concourse (-x)
    assert elev_end.door_side == "-x"

    # Both elevator dock sockets must exist in zones
    zones = {z.name for z in level.zones}
    assert "dock_elevator_0" in zones
    assert "dock_elevator_1" in zones

    # World must pass full validation
    assert world.validate() == []


def _zone_polygon(world: World, name: str) -> shapely.Polygon:
    pieces = [shapely.Polygon([(p.x, p.y) for p in z.polygon]) for z in world.all_levels[0].zones if z.name == name or (z.name.startswith(f"{name}_") and z.name[len(name) + 1 :].isdigit())]
    return shapely.union_all(pieces)


def test_concourse_elevator_cabin_extends_behind_spine_end_with_width_across_corridor() -> None:
    """Verify a non-square cabin sits beyond the spine end, depth along the spine and width across it."""
    config = {
        "length": 24.0,
        "width": 4.0,
        "elevators": [
            {"at_s": 0.0, "cabin_width": 3.0, "cabin_depth": 2.0, "door_width": 1.4},
            {"at_s": 24.0, "cabin_width": 3.0, "cabin_depth": 2.0, "door_width": 1.4},
        ],
    }
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    corridor = shapely.box(0.0, -2.0, 24.0, 2.0)
    for idx, x_range in enumerate(((-2.0, 0.0), (24.0, 26.0))):
        cabin = _zone_polygon(world, f"dock_elevator_{idx}")
        min_x, min_y, max_x, max_y = cabin.bounds
        assert (round(min_x, 3), round(max_x, 3)) == x_range
        assert (round(min_y, 3), round(max_y, 3)) == (-1.5, 1.5)
        assert cabin.intersection(corridor).area < 1e-6
        elevator = world.all_levels[0].elevators[idx]
        assert elevator.size == [2.0, 3.0, 2.5]
        assert (elevator.position.x, elevator.position.y) == (round(cabin.centroid.x, 2), round(cabin.centroid.y, 2))


def test_concourse_elevator_on_vertical_spine_end_swaps_cabin_extents() -> None:
    """Verify a cabin at the end of a spine heading along y takes depth on y and width on x."""
    config = {
        "shape": "l_bend",
        "bend_angle_deg": 90.0,
        "length": 24.0,
        "width": 4.0,
        "elevators": [{"at_s": 24.0, "cabin_width": 3.0, "cabin_depth": 2.0, "door_width": 1.4}],
    }
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    cabin = _zone_polygon(world, "dock_elevator_0")
    min_x, min_y, max_x, max_y = cabin.bounds
    assert round(max_x - min_x, 3) == 3.0
    assert round(max_y - min_y, 3) == 2.0
    elevator = world.all_levels[0].elevators[0]
    assert elevator.size == [3.0, 2.0, 2.5]
    assert elevator.door_side == "-y"


def test_concourse_rejects_elevator_on_skewed_spine_end() -> None:
    """Verify an elevator at a spine end far from axis alignment fails on compute."""
    gen = WorldGeneratorConcourse(
        {"shape": "l_bend", "bend_angle_deg": 45.0, "length": 24.0, "elevators": [{"at_s": 24.0}]},
        random.Random(42),
    )
    with pytest.raises(ValueError, match="off the nearest axis"):
        _ = gen.compute()


def test_concourse_angled_branch_door_and_jambs_sit_outside_the_corridor() -> None:
    """Verify a 45-degree branch door lies clear of the corridor and its jambs cut the branch zone."""
    config = {
        "length": 35.0,
        "width": 4.0,
        "branches": [{"at_s": 17.5, "side": "left", "angle_deg": 45.0, "width": 3.0, "length": 6.0, "door_width": 1.2}],
    }
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    door = world.all_levels[0].doors[0]
    assert door.start.y >= 2.0 - 1e-3
    assert door.end.y >= 2.0 - 1e-3

    branch = _zone_polygon(world, "branch_0_left")
    dir_x = dir_y = math.sqrt(0.5)
    mid_x = (door.start.x + door.end.x) / 2.0
    mid_y = (door.start.y + door.end.y) / 2.0
    assert branch.contains(shapely.Point(mid_x + dir_x * 0.5, mid_y + dir_y * 0.5))
    for sign in (-1.0, 1.0):
        jamb_point = shapely.Point(mid_x - dir_y * 1.05 * sign, mid_y + dir_x * 1.05 * sign)
        assert not branch.contains(jamb_point)

    dock = _zone_polygon(world, "dock_branch_0_left")
    assert dock.intersection(shapely.box(0.0, -2.0, 35.0, 2.0)).area < 1e-6


def test_concourse_corner_fillet_rounds_the_bend_without_widening_straights() -> None:
    """Verify the fillet cuts the outer corner of a bend, keeps straight sections at the nominal width and keeps branch doors on the wall."""
    base = {
        "shape": "l_bend",
        "bend_angle_deg": 45.0,
        "length": 24.0,
        "width": 4.0,
        "branches": [{"at_s": 4.0, "side": "left", "width": 3.0, "length": 6.0, "door_width": 1.2}],
    }
    sharp = WorldGeneratorConcourse(base, random.Random(42)).compute()
    rounded = WorldGeneratorConcourse({**base, "corner_fillet_radius": 6.0}, random.Random(42)).compute()

    def free_space(world: World) -> shapely.Geometry:
        return shapely.union_all([shapely.Polygon([(p.x, p.y) for p in z.polygon]) for z in world.all_levels[0].zones])

    assert not free_space(rounded).contains(shapely.Point(2.0, 2.05))
    assert free_space(rounded).contains(shapely.Point(2.0, -1.95))
    assert free_space(rounded).contains(shapely.Point(2.0, 1.95))

    tip_distance = 2.0 / math.cos(math.radians(22.5))
    bisector = math.radians(-67.5)
    tip = shapely.Point(12.0 + 0.95 * tip_distance * math.cos(bisector), 0.95 * tip_distance * math.sin(bisector))
    assert free_space(sharp).contains(tip)
    assert not free_space(rounded).contains(tip)

    door = rounded.all_levels[0].doors[0]
    assert round(door.start.y, 3) == 2.0
    assert round(door.end.y, 3) == 2.0


def test_concourse_room_default_door_fits_a_narrow_room_opening() -> None:
    """Verify the default room door never exceeds the room width."""
    config = {
        "length": 24.0,
        "width": 4.0,
        "branches": [{"at_s": 12.0, "side": "left", "width": 0.9, "length": 4.0, "is_room": True}],
    }
    level = WorldGeneratorConcourse(config, random.Random(42)).compute().all_levels[0]
    door = level.doors[0]
    assert math.hypot(door.end.x - door.start.x, door.end.y - door.start.y) <= 0.9 + 1e-3


def test_concourse_flow_field_descriptions_cite_existing_agent_types_and_behavior_trees() -> None:
    """Verify FlowStreamSpec descriptions cite only humansim agent types and hunav behavior trees that exist."""
    builtin_agents = pytest.importorskip("arena_humansim.core.agents").BUILTIN_AGENTS

    agent_description = FlowStreamSpec.model_fields["agent_type"].description or ""
    tree_description = FlowStreamSpec.model_fields["behavior_tree"].description or ""
    for agent in ("adult", "elder", "child", "hurried", "distracted"):
        assert agent in builtin_agents
        assert agent in agent_description
    trees_dir = Path(__file__).resolve().parents[2] / "configs" / "hunav" / "behavior_trees"
    for tree in ("default.xml", "BTRegularNav.xml", "BTScaredNav.xml"):
        assert (trees_dir / tree).is_file()
        assert tree in tree_description or tree.removesuffix(".xml") in tree_description
    assert "blind_pedestrian" not in agent_description
    assert "social_aware" not in tree_description


def test_concourse_rejects_pillar_that_splits_a_dock_socket() -> None:
    """Verify a pillar spanning a dock socket so it falls into pieces fails on compute."""
    config = {
        "length": 24.0,
        "width": 4.0,
        "pillars": [{"at_s": 0.8, "shape": "rectangle", "width": 0.2, "depth": 3.9}],
    }
    with pytest.raises(ValueError, match="split dock socket 'dock_start'"):
        _ = WorldGeneratorConcourse(config, random.Random(42)).compute()


def test_concourse_rejects_pillar_covering_a_dock_nav_pose() -> None:
    """Verify a pillar standing on a dock nav pose fails on compute."""
    config = {"length": 24.0, "width": 4.0, "pillars": [{"at_s": 0.75, "radius": 0.35}]}
    with pytest.raises(ValueError, match="covers the nav pose of dock socket 'dock_start'"):
        _ = WorldGeneratorConcourse(config, random.Random(42)).compute()


def test_concourse_keeps_dock_whole_under_its_name_when_a_pillar_clips_it() -> None:
    """Verify a pillar clipping a dock edge leaves one zone carrying the dock name."""
    config = {"length": 24.0, "width": 4.0, "pillars": [{"at_s": 1.0, "lateral_offset": 1.5, "radius": 0.3}]}
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    dock_zones = [z.name for z in world.all_levels[0].zones if z.name.startswith("dock_start")]
    assert dock_zones == ["dock_start"]


def test_schema_bridge_roundtrip_optional_str_and_ros_panel_config() -> None:
    """Verify config_from_params maps empty strings to None and ROS panel configs compute."""
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_rclpy_mixins")
    from arena_simulation_setup.generative.world.schema import config_from_params

    ros_params = {
        "length": 30.0,
        "width": 4.0,
        "robot_target_branch": "",
        "shape": "straight",
    }
    mapped = config_from_params(ConcourseConfiguration, ros_params)
    assert mapped["robot_target_branch"] is None


def test_schema_bridge_enum_default_list_skip_and_optional_str(rclpy_context: ModuleType) -> None:
    """Verify declare_config_params handles enums, skips lists, and handles optional strings."""
    import rclpy.node
    from arena_simulation_setup.generative.world.schema import config_from_params, declare_config_params

    assert config_from_params(ConcourseConfiguration, {"robot_target_branch": ""})["robot_target_branch"] is None

    node = rclpy.node.Node("schema_bridge")
    try:
        declare_config_params(node, "test_prefix", ConcourseConfiguration)

        assert node.get_parameter("test_prefix.shape").value == "straight"
        assert "straight" in node.describe_parameter("test_prefix.shape").additional_constraints

        for skipped in ("branches", "elevators", "pillars", "colonnades", "flows"):
            assert not node.has_parameter(f"test_prefix.{skipped}")

        assert node.has_parameter("test_prefix.robot_target_branch")
        assert node.get_parameter("test_prefix.robot_target_branch").value == ""
    finally:
        node.destroy_node()


def test_concourse_dock_zones_rotate_with_angled_paths_and_branches() -> None:
    """Verify that dock zones are oriented along local path tangents and branch directions."""
    # 1. 60-degree angled branch on straight spine
    cfg_branch = {
        "length": 36.0,
        "width": 4.0,
        "branches": [{"at_s": 18.0, "width": 3.5, "length": 10.0, "side": "left", "angle_deg": 60.0}],
    }
    gen_branch = WorldGeneratorConcourse(cfg_branch, random.Random(42))
    world_branch = gen_branch.compute()
    dock_branch = next(z for z in world_branch.all_levels[0].zones if z.name == "dock_branch_0_left")
    # Verify polygon edges are not axis-aligned (i.e. edges are angled at 60 / 150 deg)
    xs = [p.x for p in dock_branch.polygon]
    ys = [p.y for p in dock_branch.polygon]
    # No two consecutive vertices have identical X or identical Y
    for i in range(len(dock_branch.polygon)):
        p1 = dock_branch.polygon[i]
        p2 = dock_branch.polygon[(i + 1) % len(dock_branch.polygon)]
        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        if math.hypot(dx, dy) > 0.01:
            assert dx > 0.05, "Edge should not be purely vertical (aligned with world Y)"
            assert dy > 0.05, "Edge should not be purely horizontal (aligned with world X)"

    # 2. 75-degree smooth L-bend trunk terminal
    cfg_bend = {
        "shape": "smooth_l_bend",
        "length": 40.0,
        "width": 4.5,
        "bend_angle_deg": 75.0,
    }
    gen_bend = WorldGeneratorConcourse(cfg_bend, random.Random(42))
    world_bend = gen_bend.compute()
    dock_end = next(z for z in world_bend.all_levels[0].zones if z.name == "dock_end")
    # End terminal is oriented at 75 degrees, so edges are non-axis-aligned
    for i in range(len(dock_end.polygon)):
        p1 = dock_end.polygon[i]
        p2 = dock_end.polygon[(i + 1) % len(dock_end.polygon)]
        dx = abs(p2.x - p1.x)
        dy = abs(p2.y - p1.y)
        if math.hypot(dx, dy) > 0.01:
            assert dx > 0.05, "End dock edge should be rotated with bend angle"
            assert dy > 0.05, "End dock edge should be rotated with bend angle"
