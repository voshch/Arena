"""Device tracks fitted for humansim's agent types (arena_animation_retarget.devices): harmonics of the warped gait phase."""

from __future__ import annotations

import math

import numpy as np
import pytest
from arena_simulation_setup.adapters.arena_animation_retarget import devices
from arena_simulation_setup.shared.actor_pose import TemplatePose


def _phase(samples: int = 360) -> np.ndarray:
    return np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)


def test_a_track_is_fitted_with_the_fewest_harmonics_that_reproduce_it() -> None:
    w = _phase()
    v = 0.3 + 0.05 * np.sin(w + 0.2) + 0.02 * np.sin(3 * w - 1.0)
    mean, harmonics, err = devices.fit_track(w, v)
    assert mean == pytest.approx(0.3, abs=1e-4)
    assert len(harmonics) == 3
    assert harmonics[0] == pytest.approx([0.05, 0.2], abs=1e-4)
    assert harmonics[1][0] == pytest.approx(0.0, abs=1e-4)
    assert harmonics[2] == pytest.approx([0.02, -1.0], abs=1e-4)
    assert err < devices.TOLERANCE
    assert float(np.max(np.abs(devices.evaluate(w, mean, harmonics) - v))) == pytest.approx(err)


def test_a_harmonic_within_the_tolerance_is_left_out() -> None:
    w = _phase()
    mean, harmonics, _ = devices.fit_track(w, 0.1 + 0.03 * np.sin(w) + 0.0004 * np.sin(5 * w))
    assert mean == pytest.approx(0.1, abs=1e-4)
    assert len(harmonics) == 1  # the 0.4 mm fifth harmonic is within the tolerance


def test_a_constant_track_is_a_plain_number() -> None:
    w = _phase(90)
    block, err = devices.track_block(w, np.stack([np.full_like(w, 0.36), 0.1 * np.sin(w), np.zeros_like(w)], axis=1))
    assert block == {"x": pytest.approx(0.36), "y": {"mean": 0.0, "harmonics": [[0.1, 0.0]]}, "z": 0.0}
    assert err < devices.TOLERANCE


@pytest.mark.parametrize(
    ("face", "expected"),
    [
        ("-y", {"length": 0.6, "width": 0.4, "height": 0.9, "ahead": -0.05, "left": 0.1}),  # Arena's mobility aids
        ("+x", {"length": 0.4, "width": 0.6, "height": 0.9, "ahead": 0.1, "left": 0.05}),
        ("+y", {"length": 0.6, "width": 0.4, "height": 0.9, "ahead": 0.05, "left": -0.1}),
        ("-x", {"length": 0.4, "width": 0.6, "height": 0.9, "ahead": -0.1, "left": -0.05}),
    ],
)
def test_a_box_footprint_turns_the_models_front_onto_the_heading(face: str, expected: dict[str, float]) -> None:
    box = [[-0.1, 0.3], [-0.25, 0.35], [0.0, 0.9]]  # centre (0.1, 0.05)
    assert devices.box_footprint(box, face) == pytest.approx(expected)


def test_a_diagonal_front_is_rejected_for_a_box() -> None:
    with pytest.raises(ValueError, match="along a model axis"):
        devices.box_footprint([[-0.1, 0.1], [-0.1, 0.1], [0.0, 1.0]], "xy")


@pytest.mark.parametrize(("lifted", "low"), [(True, 0.0), (False, 0.0)])
def test_a_box_origin_lifts_only_a_carried_device(lifted: bool, low: float) -> None:
    body = TemplatePose()
    poses = [body.positions({"l_p_shoulder": 0.3 * math.sin(p), "r_p_shoulder": -0.3 * math.sin(p), "l_elbow": 0.5, "r_elbow": 0.5}) for p in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)]
    grip = {"side": "lr", "radius": 0.01, "angle": 0.0}
    origin = devices.box_origin(poses, grip, lifted)
    assert origin[:, 2].min() == pytest.approx(low)
    assert (origin[:, 2].max() > 0.0) == lifted
    one = devices.box_origin(poses, {**grip, "side": "l"}, lifted=False)
    assert not np.allclose(one[:, 1], origin[:, 1])  # one hand: under that hand, not between both


def test_every_source_has_a_hold_task_and_a_kind() -> None:
    from arena_simulation_setup.adapters.arena_animation_retarget.hold import TASKS

    for name, src in devices.SOURCES.items():
        assert src.grip in TASKS, name
        assert src.kind in ("cane", "rollator", "walking_frame", "iv_stand"), name
