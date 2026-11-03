"""Unit tests for procedural concourse world generator and dock socket scenarios."""

from __future__ import annotations

import math
import random

import yaml

from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
from arena_simulation_setup.generative.world.concourse import FlowStreamSpec, WorldGeneratorConcourse


def test_concourse_registered_in_world_generator_registry() -> None:
    """Verify WorldGeneratorType.CONCOURSE is registered and constructs via WorldGenerator."""
    assert WorldGeneratorType.CONCOURSE in WorldGenerator.available()
    gen = WorldGenerator(WorldGeneratorType.CONCOURSE, {"length": 24.0, "width": 4.0}, seed=42)
    world = gen.compute()
    assert len(world.all_levels) == 1
    assert len(world.all_levels[0].zones) >= 1


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

    blind_src = scenario["regions"]["dock_end_source_blind_distracted"]
    assert blind_src["config"]["agent"]["agent_type"] == "blind_pedestrian"
    assert blind_src["config"]["agent"]["behavior_tree"] == "distracted_inattentive"
    assert blind_src["config"]["agent"]["desired_velocity"]["max"] == 1.1


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

    scenario = yaml.safe_load(gen.files()["scenarios/default/scenario.yaml"])
    robot = scenario["robots"][0]
    assert robot["phases"][0]["goto"][0] == 15.0
    assert robot["phases"][0]["goto"][1] > 4.0
