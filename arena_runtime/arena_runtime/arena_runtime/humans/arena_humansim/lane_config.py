"""Lane block of a scenario source region as the arena_humansim LaneConfig message."""

from __future__ import annotations

from collections.abc import Mapping

from arena_humansim_msgs.msg import LaneConfig as LaneConfigMsg
from arena_humansim_msgs.msg import LaneSegment as LaneSegmentMsg
from arena_simulation_setup.shared.lane import LaneBlock
from geometry_msgs.msg import Pose2D as Pose2DMsg


def lane_msg_from_config(name: str, cfg: Mapping[str, object]) -> LaneConfigMsg | None:
    raw = cfg.get("lane")
    if raw is None:
        return None
    try:
        block = LaneBlock.parse(raw)
    except ValueError as exc:
        raise ValueError(f"source {name!r}: {exc}") from None
    msg = LaneConfigMsg()
    msg.side = block.side
    msg.two_way = block.two_way
    msg.has_offset = block.offset is not None
    msg.offset = block.offset or 0.0
    msg.sigma_q = block.sigma_q or 0.0
    msg.chord_tolerance = block.chord_tolerance or 0.0
    msg.track_spacing = block.track_spacing or 0.0
    msg.arrival_radius = block.arrival_radius or 0.0
    msg.merge_length = block.merge_length or 0.0
    msg.spawn_length = block.spawn_length or 0.0
    msg.sweep_length = block.sweep_length or 0.0
    msg.segments = [_segment(segment.centerline) for segment in block.segments]
    return msg


def _segment(centerline: tuple[tuple[float, float], ...]) -> LaneSegmentMsg:
    msg = LaneSegmentMsg()
    msg.centerline = [Pose2DMsg(x=x, y=y, theta=0.0) for x, y in centerline]
    return msg
