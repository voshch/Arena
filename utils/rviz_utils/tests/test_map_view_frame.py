from __future__ import annotations

import pytest

pytest.importorskip("rclpy")


def _grid(width: int, height: int, res: float, ox: float, oy: float):
    from nav_msgs.msg import MapMetaData

    info = MapMetaData()
    info.width, info.height, info.resolution = width, height, res
    info.origin.position.x, info.origin.position.y = ox, oy
    return info


def test_a_map_is_framed_on_its_walls_not_its_image() -> None:
    from rviz_utils.scripts.rviz_config import _MAP_FRAME_MARGIN_M, map_view_frame

    info = _grid(300, 300, 0.1, -15.0, -15.0)  # a 30 m image ...
    data = [0] * (300 * 300)
    for r in range(50, 100):  # ... holding a 5 m x 5 m room in one corner
        for c in range(50, 100):
            data[r * 300 + c] = 100
    whole = map_view_frame(info)
    walls = map_view_frame(info, data=data)
    assert whole["Focal Point"] == {"X": 0.0, "Y": 0.0, "Z": 0.0}
    fp = walls["Focal Point"]
    assert fp["X"] == pytest.approx(-15.0 + 7.5, abs=0.1) and fp["Y"] == pytest.approx(-15.0 + 7.5, abs=0.1)
    assert walls["Distance"] < whole["Distance"] / 3
    assert walls["Distance"] == pytest.approx((5.0 + 2 * _MAP_FRAME_MARGIN_M) * (whole["Distance"] / 30.0), rel=0.05)


def test_an_empty_map_keeps_the_image_framing() -> None:
    from rviz_utils.scripts.rviz_config import map_view_frame

    info = _grid(100, 100, 0.1, 0.0, 0.0)
    assert map_view_frame(info, data=[0] * 10000) == map_view_frame(info)
