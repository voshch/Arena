from __future__ import annotations

import pytest
from arena_runtime.humans.arena_humansim.lane_config import lane_msg_from_config

CENTERLINE = [[0.0, 0.0], [5.0, 0.0]]


def _translate(x: float, y: float) -> tuple[float, float]:
    return x + 1.0, y - 2.0


def test_lane_centerline_points_go_through_the_engine_frame() -> None:
    cfg = {"lane": {"side": "right", "segments": [{"centerline": CENTERLINE}]}}
    msg = lane_msg_from_config("lane_a", cfg, _translate)
    assert msg is not None
    points = msg.segments[0].centerline
    assert (points[0].x, points[0].y) == (1.0, -2.0)
    assert (points[1].x, points[1].y) == (6.0, -2.0)


def test_offset_key_sets_has_offset_and_the_offset() -> None:
    cfg = {"lane": {"side": "left", "offset": 0.6, "segments": [{"centerline": CENTERLINE}]}}
    msg = lane_msg_from_config("lane_b", cfg, _translate)
    assert msg is not None
    assert msg.has_offset is True
    assert msg.offset == pytest.approx(0.6)


def test_source_without_a_lane_block_has_no_lane() -> None:
    assert lane_msg_from_config("plain", {"rate_profile": []}, _translate) is None


def test_lane_without_a_side_names_the_source() -> None:
    cfg = {"lane": {"segments": [{"centerline": CENTERLINE}]}}
    with pytest.raises(ValueError, match="lane_c"):
        lane_msg_from_config("lane_c", cfg, _translate)


def test_null_optional_fields_are_treated_as_unset() -> None:
    cfg = {"lane": {"side": "right", "offset": None, "sigma_q": None, "segments": [{"centerline": CENTERLINE}]}}
    msg = lane_msg_from_config("lane_d", cfg, _translate)
    assert msg is not None
    assert msg.has_offset is False
    assert msg.sigma_q == 0.0
