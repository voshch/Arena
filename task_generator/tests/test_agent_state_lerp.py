"""Interpolation and pedestrian conversion of the engine's AgentFrame."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("rclpy")

from arena_humansim_msgs.msg import AgentFrame, AgentGestures, AgentMeta, AgentState, Gesture
from geometry_msgs.msg import Point
from task_generator.manager.realizer import Realizer
from task_generator.simulators.human.arena_humansim.arena_humansim import ArenaHumanSimulator

INTERPOLATED = {"header", "x", "y", "theta", "vx", "vy", "gait_phase"}


def _frame(sec: int, nanosec: int = 0, **columns: list) -> AgentFrame:
    msg = AgentFrame(**columns)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nanosec
    return msg


def _adapter(prev: AgentFrame | None = None, curr: AgentFrame | None = None, origin: tuple[float, float] = (0.0, 0.0)) -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._prev_agent_states = prev
    adapter._curr_agent_states = curr
    adapter._agent_gestures = {}
    adapter._agent_names = {}
    adapter._agent_types = {}
    adapter._realizer = Realizer(Realizer._Configuration(x=origin[0], y=origin[1], prefix="env_0"))
    return adapter


def test_lerp_preserves_every_uninterpolated_field() -> None:
    filled = {
        "agent_id": [7],
        "desired_velocity": [1.4],
        "radius": [0.37],
        "kind": [AgentState.KIND_HUMAN],
        "animation_state": [2],
        "policy_idx": [3],
        "gait_cadence": [1.3],
    }
    prev = _frame(0, x=[0.0], y=[0.0], theta=[0.0], vx=[1.0], vy=[0.0], gait_phase=[1.0], **filled)
    curr = _frame(1, x=[2.0], y=[0.0], theta=[0.0], vx=[1.0], vy=[0.0], gait_phase=[3.0], **filled)
    lerped = _adapter(prev, curr)._interpolate_agent_states(int(0.5e9))

    for field in AgentFrame.get_fields_and_field_types():
        if field in INTERPOLATED:
            continue
        assert getattr(lerped, field) == getattr(curr, field), f"{field} was dropped by the lerp"
    assert lerped.x[0] == pytest.approx(1.0)
    assert lerped.gait_phase[0] == pytest.approx(2.0)


def test_lerp_gait_phase_by_id_and_passes_cadence_from_curr() -> None:
    prev = _frame(0, agent_id=[3, 1], x=[0.0, 0.0], y=[0.0, 0.0], theta=[0.0, 0.0], vx=[0.0, 0.0], vy=[0.0, 0.0], gait_phase=[6.0, 1.0], gait_cadence=[0.9, 0.8])
    curr = _frame(1, agent_id=[1, 2, 3], x=[0.0, 0.0, 0.0], y=[0.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], vx=[0.0, 0.0, 0.0], vy=[0.0, 0.0, 0.0], gait_phase=[2.0, 5.0, 8.0], gait_cadence=[1.1, 1.2, 1.3])
    lerped = _adapter(prev, curr)._interpolate_agent_states(int(0.25e9))

    assert list(lerped.gait_phase) == pytest.approx([1.25, 5.0, 6.5])
    assert list(lerped.gait_cadence) == pytest.approx([1.1, 1.2, 1.3])


def test_lerp_leaves_gait_phase_alone_when_a_frame_lacks_it() -> None:
    prev = _frame(0, agent_id=[1], x=[0.0], y=[0.0], theta=[0.0], vx=[0.0], vy=[0.0])
    curr = _frame(1, agent_id=[1], x=[2.0], y=[0.0], theta=[0.0], vx=[0.0], vy=[0.0], gait_phase=[4.0], gait_cadence=[1.0])
    lerped = _adapter(prev, curr)._interpolate_agent_states(int(0.5e9))

    assert list(lerped.gait_phase) == [4.0]
    assert list(lerped.gait_cadence) == [1.0]
    assert lerped.x[0] == pytest.approx(1.0)


def test_lerp_aligns_by_id_and_passes_new_agents_through() -> None:
    prev = _frame(0, agent_id=[3, 1], x=[30.0, 10.0], y=[-3.0, -1.0], theta=[0.3, 0.1], vx=[3.0, 1.0], vy=[0.0, 0.0])
    curr = _frame(1, agent_id=[1, 2, 3], x=[12.0, 20.0, 34.0], y=[-1.0, -2.0, -5.0], theta=[0.1, 0.2, 0.3], vx=[2.0, 5.0, 4.0], vy=[1.0, 6.0, 2.0])
    lerped = _adapter(prev, curr)._interpolate_agent_states(int(0.25e9))

    assert list(lerped.agent_id) == [1, 2, 3]
    assert list(lerped.x) == pytest.approx([10.5, 20.0, 31.0])
    assert list(lerped.y) == pytest.approx([-1.0, -2.0, -3.5])
    assert list(lerped.vx) == pytest.approx([1.25, 5.0, 3.25])
    assert list(lerped.vy) == pytest.approx([0.25, 6.0, 0.5])
    assert list(lerped.theta) == pytest.approx([0.1, 0.2, 0.3])


def test_lerp_takes_the_short_way_across_pi() -> None:
    prev = _frame(0, agent_id=[1, 2], x=[0.0, 0.0], y=[0.0, 0.0], theta=[3.0, -3.0], vx=[0.0, 0.0], vy=[0.0, 0.0])
    curr = _frame(1, agent_id=[1, 2], x=[0.0, 0.0], y=[0.0, 0.0], theta=[-3.0, 3.0], vx=[0.0, 0.0], vy=[0.0, 0.0])
    lerped = _adapter(prev, curr)._interpolate_agent_states(int(0.5e9))

    step = 2.0 * math.pi - 6.0
    assert list(lerped.theta) == pytest.approx([3.0 + 0.5 * step, -3.0 - 0.5 * step])


def test_lerp_stamps_the_lerped_pose_time() -> None:
    columns = {"agent_id": [1], "x": [0.0], "y": [0.0], "theta": [0.0], "vx": [0.0], "vy": [0.0]}
    adapter = _adapter(_frame(4, 950_000_000, **columns), _frame(5, 50_000_000, **columns))

    lerped = adapter._interpolate_agent_states(5 * 10**9)
    assert (lerped.header.stamp.sec, lerped.header.stamp.nanosec) == (5, 0)
    assert lerped.header.frame_id == "map"

    clamped = adapter._interpolate_agent_states(6 * 10**9)
    assert (clamped.header.stamp.sec, clamped.header.stamp.nanosec) == (5, 50_000_000)


def test_lerp_without_prev_returns_curr() -> None:
    curr = _frame(1, agent_id=[1], x=[1.0], y=[0.0], theta=[0.0], vx=[0.0], vy=[0.0])
    assert _adapter(None, curr)._interpolate_agent_states(0) is curr


def test_gestures_attach_to_their_owner_in_env_frame() -> None:
    adapter = _adapter(origin=(100.0, 50.0))
    adapter._agent_names = {2: "env_0/ped_2"}
    adapter._agent_gestures_callback(
        AgentGestures(
            agent_id=[2, 1, 2],
            gestures=[
                Gesture(slot="head", at=Point(x=1.0, y=2.0, z=1.6)),
                Gesture(slot="arm", at=Point(x=3.0, y=4.0, z=1.0), hand="l"),
                Gesture(slot="body", at=Point(x=5.0, y=6.0), clip="hug", render_pose_override=True),
            ],
        )
    )
    frame = _frame(1, agent_id=[1, 2, 3], x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0], theta=[0.0, math.pi / 2, 0.0], vx=[0.5, 0.0, 0.0], vy=[0.0, 0.25, 0.0], animation_state=[1, 2, 0])
    peds = {ped.id: ped for ped in adapter._agent_states_to_pedestrians(frame).pedestrians}

    assert [(g.slot, g.at.x, g.at.y, g.hand) for g in peds[1].gestures] == [("arm", 103.0, 54.0, "l")]
    assert [(g.slot, g.at.x, g.at.y, g.at.z, g.clip) for g in peds[2].gestures] == [("head", 101.0, 52.0, 1.6, ""), ("body", 105.0, 56.0, 0.0, "hug")]
    assert peds[3].gestures == []

    assert (peds[1].pose.position.x, peds[1].pose.position.y) == (100.0, 50.0)
    assert (peds[2].pose.position.x, peds[2].pose.position.y) == (105.0, 56.0)
    assert (peds[3].pose.position.x, peds[3].pose.position.y) == (102.0, 52.0)
    assert peds[2].pose.orientation.z == pytest.approx(math.sin(math.pi / 4))
    assert (peds[1].twist.linear.x, peds[2].twist.linear.y) == (0.5, 0.25)
    assert [peds[i].animation_state for i in (1, 2, 3)] == [1, 2, 0]
    assert [peds[i].name for i in (1, 2, 3)] == ["1", "env_0/ped_2", "3"]


def test_gestures_replace_on_each_update() -> None:
    adapter = _adapter()
    adapter._agent_gestures_callback(AgentGestures(agent_id=[1], gestures=[Gesture(slot="head")]))
    adapter._agent_gestures_callback(AgentGestures())
    frame = _frame(1, agent_id=[1], x=[0.0], y=[0.0], theta=[0.0], vx=[0.0], vy=[0.0], animation_state=[0])
    assert adapter._agent_states_to_pedestrians(frame).pedestrians[0].gestures == []


def test_flow_obstacle_reads_pose_and_desired_velocity() -> None:
    adapter = _adapter(origin=(10.0, -5.0))
    frame = _frame(1, agent_id=[4, 9], x=[1.0, 2.0], y=[3.0, 4.0], desired_velocity=[1.1, 1.3])
    obs = adapter._make_flow_dynamic_obstacle(frame, 1, "Hospital/nurse_female_caucasian_young")

    assert obs.name == "flow_9"
    assert (obs.model.domain, obs.model.name) == ("Hospital", "nurse_female_caucasian_young")
    assert obs.sim_path == "env_0/flow_9"
    assert (obs.pose.position.x, obs.pose.position.y) == (12.0, -1.0)
    assert obs.velocity == 1.3


def test_pedestrians_carry_gait_phase_cadence_and_agent_type() -> None:
    adapter = _adapter()
    adapter._agent_meta_callback(AgentMeta(agent_id=[1, 2], name=["a", "b"], handedness=["", "l"], agent_type=["elder", ""]))
    frame = _frame(1, agent_id=[1, 2, 3], x=[0.0, 0.0, 0.0], y=[0.0, 0.0, 0.0], theta=[0.0, 0.0, 0.0], vx=[0.0, 0.0, 0.0], vy=[0.0, 0.0, 0.0], animation_state=[1, 1, 0], gait_phase=[2.5, 0.0, 7.0], gait_cadence=[1.2, 0.0, 1.9])
    peds = {ped.id: ped for ped in adapter._agent_states_to_pedestrians(frame).pedestrians}

    assert [peds[i].gait_phase for i in (1, 2, 3)] == pytest.approx([2.5, 0.0, 7.0])
    assert [peds[i].gait_cadence for i in (1, 2, 3)] == pytest.approx([1.2, 0.0, 1.9])
    assert [peds[i].agent_type for i in (1, 2, 3)] == ["elder", "", ""]

    adapter._agent_meta_callback(AgentMeta(agent_id=[3], name=["c"], handedness=[""], agent_type=["robot"]))
    peds = {ped.id: ped for ped in adapter._agent_states_to_pedestrians(frame).pedestrians}
    assert [peds[i].agent_type for i in (1, 2, 3)] == ["", "", "robot"]

    adapter._agent_meta_callback(AgentMeta(agent_id=[1, 2], name=["a", "b"], handedness=["", ""]))
    assert adapter._agent_types == {1: "", 2: ""}


def test_pedestrians_default_gait_fields_when_the_frame_has_none() -> None:
    frame = _frame(1, agent_id=[1, 2], x=[0.0, 0.0], y=[0.0, 0.0], theta=[0.0, 0.0], vx=[0.0, 0.0], vy=[0.0, 0.0], animation_state=[1, 1])
    peds = _adapter()._agent_states_to_pedestrians(frame).pedestrians
    assert [(p.gait_phase, p.gait_cadence, p.agent_type) for p in peds] == [(0.0, 0.0, ""), (0.0, 0.0, "")]
