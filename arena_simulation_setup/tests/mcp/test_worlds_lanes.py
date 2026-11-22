"""Lane checks of worlds and scenarios: the offline verdict on a lane block against a world's walls."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("arena_mcp")

import yaml
from arena_simulation_setup.generative.world.concourse import WorldGeneratorConcourse
from arena_simulation_setup.tree.World.lane_check import lane_config_from_block, lane_summary, scenario_lane_problems
from arena_simulation_setup.tree.World.pedestrian_walls import wall_segments_of_world

CORRIDOR = [((0.0, -2.0), (20.0, -2.0)), ((0.0, 2.0), (20.0, 2.0))]


def _block(side: str = "right") -> dict[str, object]:
    return {"side": side, "segments": [{"centerline": [[0.0, 0.0], [20.0, 0.0]]}]}


def test_valid_lane_reports_its_settings_band_and_track_points() -> None:
    result = lane_summary("east", _block(), CORRIDOR, agent_radius=0.25, source_extent=1.5)
    assert result["valid"] is True
    settings = result["settings"]
    assert settings["track_spacing"] == pytest.approx(3.0)
    assert settings["arrival_radius"] == pytest.approx(1.0)
    band = result["band"]
    assert band["min"] <= band["median"] <= band["max"]
    assert result["track_points"] >= 2


def test_lane_through_a_full_width_wall_names_its_flow_and_station() -> None:
    walls = CORRIDOR + [((10.0, -2.0), (10.0, 2.0))]
    result = lane_summary("cross", _block(), walls, agent_radius=0.25, source_extent=1.5)
    assert result["valid"] is False
    assert "flow 'cross'" in result["reason"]
    assert "no walkable space at station 8.10 m" in result["reason"]


def test_lane_block_without_a_side_is_rejected_with_its_flow_name() -> None:
    block = {"segments": [{"centerline": [[0.0, 0.0], [20.0, 0.0]]}]}
    with pytest.raises(ValueError, match="flow 'nosided'"):
        lane_config_from_block("nosided", block)


def test_scenario_flow_with_a_lane_passes_the_lane_check_on_its_concourse() -> None:
    config = {
        "length": 24.0,
        "width": 4.0,
        "flows": [{"name": "through", "source_dock": "dock_start", "sink_dock": "dock_end", "lane": {"side": "right"}}],
    }
    gen = WorldGeneratorConcourse(config, random.Random(42))
    world = gen.compute()
    raw = yaml.safe_load(gen.files()["scenarios/default/scenario.yaml"])
    assert scenario_lane_problems(raw["regions"], world) == []


def test_pedestrian_walls_of_a_concourse_are_what_the_lane_check_uses() -> None:
    config = {"length": 24.0, "width": 4.0, "pillars": [{"at_s": 12.0, "radius": 0.35}]}
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    assert len(wall_segments_of_world(world)) > 4
