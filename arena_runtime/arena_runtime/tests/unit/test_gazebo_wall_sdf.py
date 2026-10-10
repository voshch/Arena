"""Wall SDF generation honors the solid and shadows segment flags."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")

from arena_simulation_setup.tree.Wall import WallSegment  # noqa: E402
from arena_simulation_setup.utils.geometry import Position  # noqa: E402

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _generate_wall_sdf  # noqa: E402


def _links(*segments: WallSegment) -> list[ET.Element]:
    sdf = _generate_wall_sdf("wall", [(segment, {}) for segment in segments])
    return ET.fromstring(sdf).findall("./model/link")


def _segment(**flags: bool) -> WallSegment:
    return WallSegment(start=Position(0.0, 0.0, 0.0), end=Position(4.0, 0.0, 0.0), height=2.5, width=0.05, **flags)


def test_default_segment_has_visual_collision_and_shadows() -> None:
    (link,) = _links(_segment())
    assert link.find("visual") is not None
    assert link.find("collision") is not None
    assert link.findtext("visual/cast_shadows") == "true"


def test_non_solid_segment_has_no_collision() -> None:
    (link,) = _links(_segment(solid=False))
    assert link.find("visual") is not None
    assert link.find("collision") is None


def test_shadowless_segment_does_not_cast_shadows() -> None:
    (link,) = _links(_segment(shadows=False))
    assert link.findtext("visual/cast_shadows") == "false"


def test_camera_hidden_segment_stays_drawn_and_solid() -> None:
    (link,) = _links(_segment(visible=False))
    assert link.find("visual") is not None
    assert link.find("collision") is not None


def test_segments_keep_their_own_flags_in_one_model() -> None:
    line, backdrop = _links(_segment(visible=False, shadows=False), _segment(solid=False, shadows=False))
    assert line.find("collision") is not None
    assert backdrop.find("collision") is None
    assert [link.findtext("visual/cast_shadows") for link in (line, backdrop)] == ["false", "false"]
