from __future__ import annotations

import math

import pytest


def _view(model: str):
    from arena_robots.Robot import RobotIdentifier

    return RobotIdentifier.parse(model).resolve_sync()


def _write_model(root, mobile: str | None):
    from arena_robots.Robot import RobotView

    (root / "model_params.yaml").write_text("base_frame: base_link\n")
    if mobile is not None:
        (root / "caps").mkdir()
        (root / "caps" / "mobile.yaml").write_text(mobile)
    return RobotView(root)


@pytest.mark.parametrize(("model", "radius"), [("jackal", 0.267), ("rbsummit", 0.61)])
def test_each_robot_model_yields_its_own_mobile_cap_radius(model, radius):
    from arena_runtime.env.robot_manager.robot_manager import footprint_radius

    assert footprint_radius(_view(model)) == pytest.approx(radius)


def test_a_model_without_radius_uses_its_padded_footprint_circumradius(tmp_path):
    from arena_runtime.env.robot_manager.robot_manager import footprint_radius

    view = _write_model(tmp_path, "footprint: [[0.3, 0.4], [0.3, -0.4], [-0.3, -0.4], [-0.3, 0.4]]\nfootprint_padding: 0.05\n")
    assert footprint_radius(view) == pytest.approx(math.hypot(0.3, 0.4) + 0.05)


def test_a_model_without_radius_or_footprint_fails(tmp_path):
    from arena_runtime.env.robot_manager.robot_manager import footprint_radius

    view = _write_model(tmp_path, "odom_frame: odom\n")
    with pytest.raises(ValueError, match="neither radius nor footprint"):
        footprint_radius(view)


def test_a_model_without_mobile_cap_fails(tmp_path):
    from arena_runtime.env.robot_manager.robot_manager import footprint_radius

    view = _write_model(tmp_path, None)
    with pytest.raises(ValueError, match="no mobile cap"):
        footprint_radius(view)
