"""Lane block of a scenario source region as the arena_humansim LaneConfig message."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from arena_humansim_msgs.msg import LaneConfig as LaneConfigMsg
from arena_humansim_msgs.msg import LaneSegment as LaneSegmentMsg
from geometry_msgs.msg import Pose2D as Pose2DMsg

LANE_SIDES = ("right", "left", "center")


def lane_msg_from_config(
    name: str,
    cfg: Mapping[str, object],
    to_engine: Callable[[float, float], tuple[float, float]],
) -> LaneConfigMsg | None:
    block = cfg.get("lane")
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"source {name!r}: the lane block must be a mapping")
    side = block.get("side", "")
    if not isinstance(side, str) or side not in LANE_SIDES:
        raise ValueError(f"source {name!r}: lane side must be one of {', '.join(LANE_SIDES)}, got {side!r}")
    msg = LaneConfigMsg()
    msg.side = side
    msg.two_way = bool(block.get("two_way", False))
    if (offset := block.get("offset")) is not None:
        msg.has_offset = True
        msg.offset = _number(name, "offset", offset)
    msg.sigma_q = _optional_number(name, block, "sigma_q")
    msg.chord_tolerance = _optional_number(name, block, "chord_tolerance")
    msg.track_spacing = _optional_number(name, block, "track_spacing")
    msg.arrival_radius = _optional_number(name, block, "arrival_radius")
    msg.merge_length = _optional_number(name, block, "merge_length")
    msg.spawn_length = _optional_number(name, block, "spawn_length")
    msg.sweep_length = _optional_number(name, block, "sweep_length")
    segments = block.get("segments", [])
    if not isinstance(segments, Sequence) or isinstance(segments, str):
        raise ValueError(f"source {name!r}: lane segments must be a list")
    msg.segments = [_segment(name, segment, to_engine) for segment in segments]
    return msg


def _optional_number(name: str, block: Mapping[str, object], key: str) -> float:
    value = block.get(key)
    return 0.0 if value is None else _number(name, key, value)


def _number(name: str, key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"source {name!r}: lane {key} must be a number")
    return float(value)


def _segment(name: str, segment: object, to_engine: Callable[[float, float], tuple[float, float]]) -> LaneSegmentMsg:
    if not isinstance(segment, Mapping):
        raise ValueError(f"source {name!r}: each lane segment is a mapping with a centerline")
    points = segment.get("centerline")
    if not isinstance(points, Sequence) or isinstance(points, str):
        raise ValueError(f"source {name!r}: a lane segment needs a centerline list")
    msg = LaneSegmentMsg()
    msg.centerline = [_pose(name, point, to_engine) for point in points]
    return msg


def _pose(name: str, point: object, to_engine: Callable[[float, float], tuple[float, float]]) -> Pose2DMsg:
    if not isinstance(point, Sequence) or isinstance(point, str) or len(point) != 2:
        raise ValueError(f"source {name!r}: a lane point is [x, y]")
    x, y = to_engine(_number(name, "point x", point[0]), _number(name, "point y", point[1]))
    return Pose2DMsg(x=x, y=y, theta=0.0)
