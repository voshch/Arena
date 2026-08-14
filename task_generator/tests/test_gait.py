from __future__ import annotations

import math

import pytest

from task_generator.simulators.human.gait import GaitGenerator

# Animation state constants matching Pedestrian.msg
_IDLE = 0
_WALKING = 1
_RUNNING = 2
_PANIC = 3
_SURPRISED = 4
_CURIOUS = 5
_THREATENING = 6

_JOINT_COUNT = 36


@pytest.fixture()
def gen() -> GaitGenerator:
    return GaitGenerator()


def test_joint_names_count() -> None:
    assert len(GaitGenerator.JOINT_NAMES) == _JOINT_COUNT


def test_joint_names_unique() -> None:
    assert len(set(GaitGenerator.JOINT_NAMES)) == _JOINT_COUNT


def test_all_joints_present_walking(gen: GaitGenerator) -> None:
    result = gen.compute(agent_id=1, animation_state=_WALKING, speed=1.0, dt=0.05)
    assert set(result.keys()) == set(GaitGenerator.JOINT_NAMES)


def test_all_joints_present_idle(gen: GaitGenerator) -> None:
    result = gen.compute(agent_id=1, animation_state=_IDLE, speed=0.0, dt=0.05)
    assert set(result.keys()) == set(GaitGenerator.JOINT_NAMES)


def test_all_joints_present_running(gen: GaitGenerator) -> None:
    result = gen.compute(agent_id=1, animation_state=_RUNNING, speed=3.0, dt=0.05)
    assert set(result.keys()) == set(GaitGenerator.JOINT_NAMES)


@pytest.mark.parametrize("state", [_IDLE, _WALKING, _RUNNING, _PANIC, _SURPRISED, _CURIOUS, _THREATENING])
def test_all_values_finite(gen: GaitGenerator, state: int) -> None:
    result = gen.compute(agent_id=2, animation_state=state, speed=1.0, dt=0.05)
    for name, val in result.items():
        assert math.isfinite(val), f"{name} is not finite: {val}"


_LIMITS: tuple[tuple[float, float], ...] = (
    (-0.6, 0.6),
    (-0.8, 0.8),
    (-0.2, 1.0),
    (-0.3, 0.3),
    (-0.4, 0.4),
    (-0.1, 0.5),
    (-0.3, 0.3),
    (-0.4, 0.4),
    (-0.1, 0.5),
    (-1.0, 1.0),
    (-1.4, 1.4),
    (-1.5, 1.5),
    (-0.5, 0.5),
    (-0.2, 0.6),
    (-3.1, 3.1),
    (-1.0, 3.3),
    (-1.6, 1.6),
    (0.0, 2.5),
    (-0.5, 0.5),
    (-0.2, 0.6),
    (-3.1, 3.1),
    (-1.0, 3.3),
    (-1.6, 1.6),
    (0.0, 2.5),
    (-0.1, 0.6),
    (-0.4, 3.3),
    (-0.4, 0.7),
    (-2.5, 0.0),
    (-0.1, 0.6),
    (-0.4, 3.3),
    (-0.4, 0.7),
    (-2.5, 0.0),
    (-0.6, 0.6),
    (-0.9, 0.6),
    (-0.6, 0.6),
    (-0.9, 0.6),
)


@pytest.mark.parametrize("state", [_IDLE, _WALKING, _RUNNING])
def test_values_within_limits(gen: GaitGenerator, state: int) -> None:
    result = gen.compute(agent_id=3, animation_state=state, speed=1.5, dt=0.05)
    for i, name in enumerate(GaitGenerator.JOINT_NAMES):
        lo, hi = _LIMITS[i]
        val = result[name]
        assert lo <= val <= hi, f"{name}={val} outside [{lo}, {hi}]"


def test_phase_monotonic_across_calls(gen: GaitGenerator) -> None:
    """A gait-driving joint should vary across successive WALKING calls."""
    values: list[float] = []
    for _ in range(30):
        result = gen.compute(agent_id=10, animation_state=_WALKING, speed=1.2, dt=0.05)
        values.append(result["l_r_hip"])
    # With 30 steps at cadence ~1.06 Hz and dt=0.05, phase advances ~1.6 rad total.
    # The driven joint cannot be constant for that arc.
    assert max(values) - min(values) > 0.01, "l_r_hip did not change over 30 steps"


def test_phase_completes_cycle(gen: GaitGenerator) -> None:
    """After a full gait cycle the hip joint should return near its start value."""
    g = GaitGenerator()
    cadence = 0.4 + 0.55 * 1.2  # ~1.06 Hz
    period = 1.0 / cadence
    steps = 200
    dt = period / steps
    first = g.compute(agent_id=5, animation_state=_WALKING, speed=1.2, dt=0.0)["l_r_hip"]
    for _ in range(steps):
        last = g.compute(agent_id=5, animation_state=_WALKING, speed=1.2, dt=dt)["l_r_hip"]
    assert math.isclose(first, last, abs_tol=0.05), f"cycle not closed: {first} vs {last}"


def test_different_agents_not_in_lockstep() -> None:
    """Two agents with different ids must start at different phases."""
    g = GaitGenerator()
    r1 = g.compute(agent_id=1, animation_state=_WALKING, speed=1.2, dt=0.0)
    r2 = g.compute(agent_id=37, animation_state=_WALKING, speed=1.2, dt=0.0)
    assert r1["l_r_hip"] != r2["l_r_hip"], "agents 1 and 37 are in lockstep at t=0"


def test_idle_limb_joints_near_zero(gen: GaitGenerator) -> None:
    result = gen.compute(agent_id=1, animation_state=_IDLE, speed=0.0, dt=0.0)
    limb_joints = [
        "l_r_hip",
        "r_r_hip",
        "l_knee",
        "r_knee",
        "l_p_shoulder",
        "r_p_shoulder",
        "l_elbow",
        "r_elbow",
    ]
    for name in limb_joints:
        assert abs(result[name]) < 1e-9, f"idle {name}={result[name]} not near zero"


def test_forget_resets_phase(gen: GaitGenerator) -> None:
    gen.compute(agent_id=99, animation_state=_WALKING, speed=1.0, dt=0.1)
    gen.forget(99)
    # After forget, next call re-seeds from id, giving the same value as a fresh generator.
    fresh = GaitGenerator()
    r_fresh = fresh.compute(agent_id=99, animation_state=_WALKING, speed=1.0, dt=0.0)
    r_reset = gen.compute(agent_id=99, animation_state=_WALKING, speed=1.0, dt=0.0)
    assert math.isclose(r_fresh["l_r_hip"], r_reset["l_r_hip"], abs_tol=1e-12)


def test_behavior_states_treated_as_idle(gen: GaitGenerator) -> None:
    """PANIC/SURPRISED/CURIOUS/THREATENING should produce idle-like limb angles."""
    for state in (_PANIC, _SURPRISED, _CURIOUS, _THREATENING):
        result = gen.compute(agent_id=1, animation_state=state, speed=0.0, dt=0.0)
        assert abs(result["l_r_hip"]) < 1e-9, f"state {state}: l_r_hip not idle"
        assert abs(result["r_r_hip"]) < 1e-9, f"state {state}: r_r_hip not idle"


def test_limb_pairs_exact_antiphase() -> None:
    """L/R limb joints share one profile half a cycle apart (JOINTS.md antiphase).
    Ids 0 and 180 seed phases exactly pi apart, so left at one equals right at the
    other. A plain sign-mirror does not hold because the baked profiles have
    nonzero means (hips average forward-flexed)."""
    g = GaitGenerator()
    r0 = g.compute(agent_id=0, animation_state=_WALKING, speed=1.0, dt=0.0)
    r180 = g.compute(agent_id=180, animation_state=_WALKING, speed=1.0, dt=0.0)
    for left, right in (("l_r_hip", "r_r_hip"), ("l_knee", "r_knee"), ("l_p_shoulder", "r_p_shoulder"), ("l_elbow", "r_elbow")):
        assert math.isclose(r0[left], r180[right], abs_tol=1e-9), f"{left} vs {right} not antiphase"
    assert abs(r0["l_p_shoulder"]) > 1e-6, "l_p_shoulder should be clearly nonzero"
    assert abs(r0["l_r_hip"] - r0["r_r_hip"]) > 1e-6, "hips should differ within one agent"
