"""Lane diagnostics of generated worlds and the concourse configuration schema."""

from __future__ import annotations

import json
import random

import pytest

pytest.importorskip("arena_mcp")

from arena_simulation_setup.generative.world.concourse import ConcourseConfiguration, WorldGeneratorConcourse
from arena_simulation_setup.tree.World.lane_check import scenario_lane_results


def test_lane_results_summarise_each_lane_flow_and_skip_plain_flows() -> None:
    config = {
        "length": 24.0,
        "width": 4.0,
        "flows": [
            {"name": "through", "source_dock": "dock_start", "sink_dock": "dock_end", "lane": {"side": "right"}},
            {"name": "plain", "source_dock": "dock_end", "sink_dock": "dock_start"},
        ],
    }
    gen = WorldGeneratorConcourse(config, random.Random(42))
    world = gen.compute()
    regions = gen.scenarios()["default"].scenario["regions"]
    results = scenario_lane_results(regions, world)
    assert set(results) == {"dock_start_source_through"}
    assert results["dock_start_source_through"]["valid"] is True
    assert results["dock_start_source_through"]["track_points"] >= 2


def test_lane_results_name_a_source_whose_zone_is_missing() -> None:
    config = {"length": 24.0, "width": 4.0}
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    regions = {
        "ghost_source": {
            "type": "source",
            "ref": "nowhere",
            "config": {"lane": {"side": "right", "segments": [{"centreline": [[0.0, 0.0], [5.0, 0.0]]}]}},
        }
    }
    results = scenario_lane_results(regions, world)
    assert results["ghost_source"]["valid"] is False
    assert "no zone 'nowhere'" in results["ghost_source"]["reason"]


def test_concourse_configuration_schema_lists_the_lane_fields() -> None:
    schema = json.dumps(ConcourseConfiguration.model_json_schema())
    assert "FlowLaneSpec" in schema
    assert '"side"' in schema
    assert '"sweep_length"' in schema
