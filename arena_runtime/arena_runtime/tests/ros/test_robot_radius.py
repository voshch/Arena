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
    from arena_runtime.shared import footprint_radius

    assert footprint_radius(_view(model).caps, model) == pytest.approx(radius)


def test_a_model_without_radius_uses_its_padded_footprint_circumradius(tmp_path):
    from arena_runtime.shared import footprint_radius

    view = _write_model(tmp_path, "footprint: [[0.3, 0.4], [0.3, -0.4], [-0.3, -0.4], [-0.3, 0.4]]\nfootprint_padding: 0.05\n")
    assert footprint_radius(view.caps, "model") == pytest.approx(math.hypot(0.3, 0.4) + 0.05)


def test_a_model_without_radius_or_footprint_fails(tmp_path):
    from arena_runtime.shared import footprint_radius

    view = _write_model(tmp_path, "odom_frame: odom\n")
    with pytest.raises(ValueError, match="neither radius nor footprint"):
        footprint_radius(view.caps, "model")


def test_a_model_without_mobile_cap_fails(tmp_path):
    from arena_runtime.shared import footprint_radius

    view = _write_model(tmp_path, None)
    with pytest.raises(ValueError, match="no mobile cap"):
        footprint_radius(view.caps, "model")


def test_footprint_radius_follows_the_instance_cap_overrides(tmp_path):
    from arena_robots.request import InstanceRequest
    from arena_runtime.shared import footprint_radius

    view = _write_model(tmp_path, "radius: 0.3\nfootprint: [[0.3, 0.4], [-0.3, -0.4]]\n")
    scaled = InstanceRequest(caps={"mobile.radius": {"scale": 2.0}}).caps_of(view)
    assert footprint_radius(scaled, "model") == pytest.approx(0.6)
    assert footprint_radius(view.caps, "model") == pytest.approx(0.3)


def test_footprint_radius_uses_an_overridden_footprint_when_radius_is_unset(tmp_path):
    from arena_robots.request import InstanceRequest
    from arena_runtime.shared import footprint_radius

    view = _write_model(tmp_path, "footprint: [[0.3, 0.4], [-0.3, -0.4]]\n")
    wide = InstanceRequest(caps={"mobile.footprint": [[1.0, 0.0], [-1.0, 0.0]]}).caps_of(view)
    assert footprint_radius(wide, "model") == pytest.approx(1.0)
