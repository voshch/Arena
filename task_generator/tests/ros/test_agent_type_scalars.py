"""A type's speed and radius apply unless the scenario overrides them - and a knob that rewrites
either field is not undone by a stale scalar on the agent block.

Before this, `ArenaHumanDynamicObstacle` defaulted to 1.0-1.5 m/s and 0.35 m whether or not the
scenario declared anything: `elder` walked like an adult, `child` had an adult's body, and the
Level A `desired_velocity` / `agent_radius` knobs on a perturbed agent were overwritten at spawn.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.ros

pytest.importorskip("arena_humansim_msgs", reason="needs a sourced environment")

from task_generator.shared import DynamicObstacle, Pose, Position  # noqa: E402
from task_generator.simulators.human.arena_humansim import ArenaHumanDynamicObstacle  # noqa: E402


def _obstacle(agent: dict | str) -> DynamicObstacle:
    return DynamicObstacle(
        name="p", model="arenian", pose=Pose(Position(1.0, 1.0)),
        waypoints=[Position(2.0, 2.0)], extra={"agent": agent},
    )


def test_undeclared_speed_and_radius_are_none_not_defaults():
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(_obstacle({"agent_type": "elder"}))
    assert parsed is not None
    assert parsed.desired_velocity_min is None and parsed.desired_velocity_max is None
    assert parsed.agent_radius is None


def test_declared_scalars_are_kept():
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(
        _obstacle({"agent_type": "adult", "desired_velocity": 1.2, "radius": 0.3}),
    )
    assert (parsed.desired_velocity_min, parsed.desired_velocity_max, parsed.agent_radius) == (1.2, 1.2, 0.3)


def test_the_types_own_distribution_applies_when_nothing_is_declared():
    rng = np.random.default_rng(0)
    elder = ArenaHumanDynamicObstacle.from_dynamic_obstacle(_obstacle({"agent_type": "elder"})).sample_params(rng)
    assert elder is not None
    # elder: desired_velocity clip 0.3-1.1, radius clip 0.15-0.35. The old defaults were
    # uniform(1.0, 1.5) and exactly 0.35.
    assert 0.3 <= elder.desired_velocity <= 1.1
    assert elder.agent_radius < 0.35


def test_a_declared_scalar_still_wins():
    rng = np.random.default_rng(0)
    parsed = ArenaHumanDynamicObstacle.from_dynamic_obstacle(
        _obstacle({"agent_type": "elder", "desired_velocity": 1.4, "radius": 0.5}),
    )
    params = parsed.sample_params(rng)
    assert params.desired_velocity == 1.4 and params.agent_radius == 0.5


@pytest.mark.parametrize("kind", ["child", "hurried", "distracted"])
def test_new_builtin_types_resolve_by_bare_name(kind: str) -> None:
    rng = np.random.default_rng(1)
    params = ArenaHumanDynamicObstacle.from_dynamic_obstacle(_obstacle({"agent_type": kind})).sample_params(rng)
    if params is None:
        pytest.skip("humansim's share dir does not carry the new types yet - rebuild arena_humansim")
    assert params.name == kind
