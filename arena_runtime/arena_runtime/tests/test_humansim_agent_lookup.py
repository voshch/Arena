"""Engine agent ids of spawned pedestrians by name or sim path."""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim_msgs")

from arena_runtime.shared import DynamicObstacle, Pose, Position
from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.humans.utils import KnownObstacles


def _adapter() -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._known_obstacles = KnownObstacles()
    edge = DynamicObstacle(name="edge_0", model="arenian", pose=Pose(Position(1.0, 1.0)), waypoints=[])
    edge.sim_path = "env_0/edge_0"
    adapter._known_obstacles.create_or_get(name=edge.name, obstacle=edge)
    adapter._agent_names = {4: "env_0/edge_0", 9: "env_0/flow_9"}
    return adapter


def test_agent_id_resolves_a_spawned_name_through_its_sim_path() -> None:
    assert _adapter()._agent_id("edge_0") == 4


def test_agent_id_resolves_a_sim_path_directly() -> None:
    adapter = _adapter()
    assert adapter._agent_id("env_0/edge_0") == 4
    assert adapter._agent_id("env_0/flow_9") == 9


def test_agent_id_of_an_unknown_agent_is_none() -> None:
    assert _adapter()._agent_id("edge_1") is None


def test_agent_id_reads_a_given_id_table_instead_of_the_spawned_agents() -> None:
    adapter = _adapter()
    ids = adapter._agent_ids()
    assert ids == {"env_0/edge_0": 4, "env_0/flow_9": 9}
    assert adapter._agent_id("edge_0", ids) == 4
    assert adapter._agent_id("env_0/flow_9", ids) == 9
    assert adapter._agent_id("edge_0", {"env_0/edge_0": 7}) == 7
    assert adapter._agent_id("edge_1", ids) is None
