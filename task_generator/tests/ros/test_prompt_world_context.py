from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

try:
    from arena_simulation_setup.tree.World import WorldView
    from arena_simulation_setup.tree.World.Scenario import parse_scenario

    from arena_runtime.shared import WORLD_ENTITY_PREFIX
    from task_generator.tasks.obstacles.prompt.arena import TM_Prompt
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 or google-genai not available")

WORLD = """\
format: arena-world/2
levels:
- id: '0'
  zones:
  - name: office
    polygon: [[0, 0], [4, 0], [4, 4], [0, 4]]
    walls: outline
  - name: hall
    polygon: [[4, 0], [8, 0], [8, 4], [4, 4]]
  objects:
  - {name: desk_1, model: desk, pose: [1.0, 1.0, 0.0]}
  - {name: crate, model: crate, pose: [20.0, 20.0, 0.0]}
"""

SEMANTICS = """\
format: arena-kg/1
uses: [core@2]
entities:
- {id: 'room:office', props: {description: Corner Office}}
facts: []
constraints: []
"""


def _context(tmp_path: Path) -> dict:
    (tmp_path / "world.yaml").write_text(WORLD)
    (tmp_path / "kg.yaml").write_text(SEMANTICS)
    world = WorldView(tmp_path).load()
    assert world.kg is not None
    level = world.levels["0"]
    for entity in level.objects:
        entity.name = f"{WORLD_ENTITY_PREFIX}{entity.name}"
    return json.loads(TM_Prompt.preprocess_world_description(TM_Prompt.__new__(TM_Prompt), level, world.kg))


def test_entities_are_grouped_by_the_zones_that_contain_them(tmp_path: Path) -> None:
    context = _context(tmp_path)
    zones = {zone["name"]: zone for zone in context["zones"]}
    assert [entity["name"] for entity in zones["office"]["entities"]] == [f"{WORLD_ENTITY_PREFIX}desk_1"]
    assert zones["hall"]["entities"] == []
    assert [entity["name"] for entity in context["entities"]] == [f"{WORLD_ENTITY_PREFIX}crate"]


def test_zones_carry_their_corners_bordering_walls_and_description(tmp_path: Path) -> None:
    zones = {zone["name"]: zone for zone in _context(tmp_path)["zones"]}
    assert zones["office"]["corners"] == [[0, 0], [4, 0], [4, 4], [0, 4]]
    assert len(zones["office"]["walls"]) == 4
    assert [sorted(map(tuple, wall)) for wall in zones["hall"]["walls"]] == [[(4, 0), (4, 4)]]
    assert zones["office"]["description"] == "Corner Office"
    assert "description" not in zones["hall"]


def test_llm_agents_become_format_2_humans() -> None:
    llm_output = {
        "obstacles": {
            "static": [],
            "dynamic": [
                {
                    "name": "agent_1",
                    "pos": [1.0, 2.0, 90.0],
                    "agent_type": "elder",
                    "model": "female_adult_business_02",
                    "desired_velocity": 0.8,
                    "waypoints": [[3.0, 4.0, 0.0]],
                    "waypoint_mode": "Reverse",
                }
            ],
        }
    }
    scenario = parse_scenario(TM_Prompt.llm_output_to_config(TM_Prompt.__new__(TM_Prompt), llm_output, "arena"))
    (human,) = scenario.humans
    assert human.name == "agent_1"
    assert (human.pose.position.x, human.pose.position.y) == (1.0, 2.0)
    assert human.pose.orientation.to_yaw() == pytest.approx(math.pi / 2)
    assert [(waypoint.x, waypoint.y) for waypoint in human.waypoints] == [(3.0, 4.0)]
    assert human.waypoint_mode == "reverse"
    assert human.agent == {"agent_type": "elder", "desired_velocity": 0.8}
    assert scenario.objects == []
