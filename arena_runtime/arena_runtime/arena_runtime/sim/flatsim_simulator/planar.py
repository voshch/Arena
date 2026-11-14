"""Drive model of a robot from the `planar` block of its mobile cap."""

from __future__ import annotations

import math
import typing

from flatsim_msgs.msg import Axis, DriveModel

from flatsim.drive import DEFAULT_TIMEOUT

if typing.TYPE_CHECKING:
    from arena_robots.caps import PlanarSpec


def _axis(low: float = -math.inf, high: float = math.inf, accel: float = math.inf) -> Axis:
    return Axis(low=low, high=high, accel=accel)


def drive_model(spec: PlanarSpec | None, *, holonomic: bool, min_turning_radius: float = 0.0) -> DriveModel:
    """Drive model of a planar spec, or an unbounded one that moves sideways only when `holonomic`, turning no tighter than `min_turning_radius`."""
    if spec is None:
        return DriveModel(linear=_axis(), lateral=_axis() if holonomic else _axis(0.0, 0.0), angular=_axis(), gain=[1.0, 1.0, 1.0], traction=math.inf, timeout=DEFAULT_TIMEOUT, min_turning_radius=min_turning_radius)

    def axis(name: str) -> Axis:
        bounds = spec.velocity.get(name)
        accel = spec.acceleration.get(name, math.inf)
        return _axis(accel=accel) if bounds is None else _axis(bounds.min, bounds.max, accel)

    return DriveModel(
        linear=axis('linear'),
        lateral=axis('lateral'),
        angular=axis('angular'),
        gain=[spec.gain.get('linear', 1.0), spec.gain.get('lateral', 1.0), spec.gain.get('angular', 1.0)],
        lag=spec.lag,
        traction=spec.traction if spec.traction is not None else math.inf,
        pivot=spec.pivot,
        timeout=spec.timeout if spec.timeout is not None else DEFAULT_TIMEOUT,
        min_turning_radius=min_turning_radius,
    )
