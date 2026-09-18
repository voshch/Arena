from __future__ import annotations

import pytest
from nav_msgs.msg import MapMetaData

from rviz_utils.scripts.rviz_config import _FALLBACK_MAP_VIEW, _MAP_VIEW_DISTANCE_PER_M, map_view_frame


def _info(width: int, height: int, resolution: float, ox: float, oy: float) -> MapMetaData:
    info = MapMetaData(width=width, height=height, resolution=resolution)
    info.origin.position.x = ox
    info.origin.position.y = oy
    return info


def test_frames_the_map_centre():
    view = map_view_frame(_info(410, 250, 0.05, -11.25, -8.25))  # arena_arena_002
    assert view["Focal Point"] == {"X": -1.0, "Y": -2.0, "Z": 0.0}
    assert view["Distance"] == pytest.approx(20.5 * _MAP_VIEW_DISTANCE_PER_M, abs=0.05)


def test_hospital_sized_map_keeps_the_legacy_distance():
    view = map_view_frame(_info(510, 694, 0.05, -0.25, -0.25))  # hospital_1
    assert view["Distance"] == pytest.approx(_FALLBACK_MAP_VIEW["Distance"], abs=1.0)
    assert view["Focal Point"]["X"] == pytest.approx(12.5)


def test_degenerate_map_never_collapses_the_camera():
    assert map_view_frame(_info(1, 1, 0.05, 0.0, 0.0))["Distance"] == pytest.approx(_MAP_VIEW_DISTANCE_PER_M, abs=0.06)


def test_grid_frame_offset_moves_the_focal_point_into_the_fixed_frame():
    # arena_arena_002's grid is published at origin (4.75, 4.75) in `map`, which sits at
    # (-16, -13) in the env's fixed frame: the focal point must land on the world centre.
    view = map_view_frame(_info(410, 250, 0.05, 4.75, 4.75), offset=(-16.0, -13.0, 0.0))
    assert view["Focal Point"] == {"X": -1.0, "Y": -2.0, "Z": 0.0}


def test_grid_frame_yaw_rotates_the_centre():
    import math

    view = map_view_frame(_info(20, 20, 1.0, 0.0, 0.0), offset=(0.0, 0.0, math.pi / 2))
    assert view["Focal Point"]["X"] == pytest.approx(-10.0)
    assert view["Focal Point"]["Y"] == pytest.approx(10.0)
