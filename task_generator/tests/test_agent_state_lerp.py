"""Interpolating two engine snapshots must carry every field it does not interpolate."""

from __future__ import annotations

import pytest

pytest.importorskip("rclpy")

from arena_humansim_msgs.msg import AgentState, AgentStates
from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator

INTERPOLATED = {"pose", "velocity"}


def _agent(x: float, **kw: object) -> AgentState:
    agent = AgentState()
    agent.agent_id = 7
    agent.pose.x = x
    agent.velocity.x = 1.0
    for key, value in kw.items():
        setattr(agent, key, value)
    return agent


def _snapshot(agent: AgentState, sec: int) -> AgentStates:
    msg = AgentStates()
    msg.header.stamp.sec = sec
    msg.agents.append(agent)
    return msg


def test_lerp_preserves_every_uninterpolated_field() -> None:
    filled = {
        "name": "ped_1",
        "agent_type": "adult",
        "policy": "sfm",
        "policy_params": "{}",
        "handedness": "l",
        "kind": AgentState.KIND_HUMAN,
        "animation_state": 2,
        "desired_velocity": 1.4,
        "radius": 0.37,
        "vision_range": 6.5,
        "vision_fov": 170.0,
        "relaxation_time": 0.4,
        "repulsion_strength": 2.5,
        "repulsion_range": 0.8,
    }
    curr = _agent(2.0, **filled)
    lerped = ArenaHumanSimulator._interpolate_agent_states(
        _lerper(_snapshot(_agent(0.0, **filled), 0), _snapshot(curr, 1)), int(0.5e9)
    ).agents[0]

    for field in AgentState.get_fields_and_field_types():
        if field in INTERPOLATED:
            continue
        assert getattr(lerped, field) == getattr(curr, field), f"{field} was dropped by the lerp"
    assert lerped.pose.x == pytest.approx(1.0)


class _lerper:
    """The interpolator only reads the two cached snapshots."""

    def __init__(self, prev: AgentStates, curr: AgentStates) -> None:
        self._prev_agent_states = prev
        self._curr_agent_states = curr
