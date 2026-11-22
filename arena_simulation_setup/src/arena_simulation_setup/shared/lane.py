"""Lane block of a scenario source region."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import attrs

LANE_SIDES = ('right', 'left', 'center')
_OPTIONAL_NUMBERS = (
    'sigma_q',
    'chord_tolerance',
    'track_spacing',
    'arrival_radius',
    'merge_length',
    'spawn_length',
    'sweep_length',
)


def _number(key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f'lane {key} must be a number, got {value!r}')
    return float(value)


def _positive(key: str, value: object) -> float | None:
    if value is None:
        return None
    number = _number(key, value)
    if number <= 0.0:
        raise ValueError(f'lane {key} must be positive, got {number}')
    return number


def _point(value: object) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f'a lane point is [x, y], got {value!r}')
    return _number('point x', value[0]), _number('point y', value[1])


@attrs.frozen
class LaneSegment:
    """One centerline of at least two [x, y] points."""

    centerline: tuple[tuple[float, float], ...]

    @classmethod
    def parse(cls, value: object) -> LaneSegment:
        points = value.get('centerline') if isinstance(value, Mapping) else None
        if not isinstance(points, Sequence) or isinstance(points, str) or len(points) < 2:
            raise ValueError('each lane segment needs a centerline of at least two [x, y] points')
        return cls(tuple(_point(point) for point in points))


@attrs.frozen
class LaneBlock:
    """The `lane` block of a source region: unset optional numbers are None."""

    side: str
    segments: tuple[LaneSegment, ...]
    offset: float | None = None
    two_way: bool = False
    sigma_q: float | None = None
    chord_tolerance: float | None = None
    track_spacing: float | None = None
    arrival_radius: float | None = None
    merge_length: float | None = None
    spawn_length: float | None = None
    sweep_length: float | None = None

    @classmethod
    def parse(cls, block: object) -> LaneBlock:
        if not isinstance(block, Mapping):
            raise ValueError('the lane block must be a mapping')
        side = block.get('side')
        if side is None or side == '':
            raise ValueError(f'a lane needs a side ({", ".join(LANE_SIDES)})')
        if not isinstance(side, str) or side not in LANE_SIDES:
            raise ValueError(f'lane side must be one of {", ".join(LANE_SIDES)}, got {side!r}')
        segments = block.get('segments')
        if not isinstance(segments, Sequence) or isinstance(segments, str) or not segments:
            raise ValueError('a lane needs at least one segment')
        offset = block.get('offset')
        return cls(
            side=side,
            segments=tuple(LaneSegment.parse(segment) for segment in segments),
            offset=None if offset is None else _number('offset', offset),
            two_way=bool(block.get('two_way', False)),
            **{key: _positive(key, block.get(key)) for key in _OPTIONAL_NUMBERS},
        )


def shift_lane_points(config: Mapping[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """Copy of a region config with every lane centerline point moved by (dx, dy)."""
    shifted = dict(config)
    lane = config.get('lane')
    if lane is None:
        return shifted
    block = LaneBlock.parse(lane)
    shifted['lane'] = {
        **lane,
        'segments': [{**raw, 'centerline': [[x + dx, y + dy] for x, y in segment.centerline]} for raw, segment in zip(lane['segments'], block.segments, strict=True)],
    }
    return shifted
