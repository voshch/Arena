"""A speed and radius the scenario declares override the agent type's, and every builtin type resolves by name."""

from __future__ import annotations

import attrs
import numpy as np
import pytest

pytestmark = pytest.mark.ros

pytest.importorskip("arena_humansim_msgs", reason="needs a sourced environment")

from arena_runtime.shared import DynamicObstacle, Pose, Position
from arena_runtime.humans.arena_humansim.agent import ArenaHumanDynamicObstacle


def _obstacle(agent: dict | str) -> DynamicObstacle:
    return DynamicObstacle(
        name="p",
        model="arenian",
        pose=Pose(Position(1.0, 1.0)),
        waypoints=[Position(2.0, 2.0)],
        agent=agent,
    )


def test_declared_scalars_collapse_the_speed_range_and_keep_the_type() -> None:
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(
        _obstacle({"agent_type": "elder", "desired_velocity": 1.2, "radius": 0.3}),
    )
    assert (parsed.agent_type, parsed.desired_velocity_min, parsed.desired_velocity_max, parsed.agent_radius) == ("elder", 1.2, 1.2, 0.3)


def test_declared_scalars_replace_only_speed_and_radius_of_the_sampled_type() -> None:
    declared = ArenaHumanDynamicObstacle.from_dynamic_obstacle(
        _obstacle({"agent_type": "elder", "desired_velocity": 1.4, "radius": 0.5}),
    ).sample_params(np.random.default_rng(0))
    undeclared = ArenaHumanDynamicObstacle(agent_type="elder").sample_params(np.random.default_rng(0))
    assert declared.name == "elder"
    assert (declared.desired_velocity, declared.agent_radius) == (1.4, 0.5)
    assert attrs.evolve(declared, desired_velocity=0.0, agent_radius=0.0) == attrs.evolve(undeclared, desired_velocity=0.0, agent_radius=0.0)


@pytest.mark.parametrize("kind", ["child", "hurried", "distracted"])
def test_new_builtin_types_resolve_by_bare_name(kind: str) -> None:
    rng = np.random.default_rng(1)
    params = ArenaHumanDynamicObstacle.from_dynamic_obstacle(_obstacle({"agent_type": kind})).sample_params(rng)
    assert params is not None and params.name == kind
