"""Assistive devices pedestrians hold (a long cane, a rollator, a walking frame, an IV stand), posed from the rendered gait.

An agent type's `device:` block (humansim passes it through untouched, like `pose:`) gives the device in the body frame (x
forward, y left, z up from the soles, origin under the pedestrian's position, metres) as tracks of the warped gait phase w,
the gait profiles' signal convention: value = mean + sum_k amp_k * sin(k * w + phase_k), a plain number for a constant.
The arm is drawn from the same profile at the same phase, so the device posed here is the one in the hand.

  cane                            grip, tip: the shaft from the fist's grip centre to the tip; radius: the shaft's
  rollator, walking_frame,        origin: on the floor under the middle of the grips (z: lift); size: the footprint's
  iv_stand                        length along the heading, width and height, its centre `ahead` and `left` of the origin

and optionally `model`, the object model renderers draw for it (an Arena asset name). The blocks are fitted by
`python3 -m arena_simulation_setup.adapters.arena_animation_retarget.devices`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import attrs
import shapely
from arena_people_msgs.msg import HeldDevice

CANE = HeldDevice.CANE
BOXES = (HeldDevice.ROLLATOR, HeldDevice.WALKING_FRAME, HeldDevice.IV_STAND)
KINDS = (CANE, *BOXES)


@attrs.frozen
class Track:
    """mean + sum_k amp_k * sin(k * w + phase_k) of the warped gait phase w."""

    mean: float
    harmonics: tuple[tuple[float, float], ...] = ()

    @classmethod
    def parse(cls, raw: object, where: str) -> Track:
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            return cls(float(raw))
        if not isinstance(raw, Mapping) or set(raw) - {"mean", "harmonics"} or "mean" not in raw:
            raise ValueError(f"{where}: a number or {{mean, harmonics: [[amp, phase], ...]}}, got {raw!r}")
        return cls(float(raw["mean"]), tuple((float(amp), float(ph)) for amp, ph in raw.get("harmonics") or ()))

    def at(self, w: float) -> float:
        return self.mean + sum(amp * math.sin(k * w + ph) for k, (amp, ph) in enumerate(self.harmonics, start=1))


@attrs.frozen
class Point3:
    """A body-frame point whose coordinates follow the gait phase."""

    x: Track
    y: Track
    z: Track

    @classmethod
    def parse(cls, raw: object, where: str) -> Point3:
        if not isinstance(raw, Mapping) or set(raw) != {"x", "y", "z"}:
            raise ValueError(f"{where}: needs exactly x, y and z, got {raw!r}")
        return cls(*(Track.parse(raw[a], f"{where}.{a}") for a in "xyz"))

    def at(self, w: float) -> tuple[float, float, float]:
        return self.x.at(w), self.y.at(w), self.z.at(w)


_CANE_KEYS = {"kind", "grip", "tip", "radius"}
_BOX_KEYS = {"kind", "origin", "size"}
_SIZE_KEYS = {"length", "width", "height", "ahead", "left"}


@attrs.frozen
class DeviceSpec:
    """The device an agent type holds, in the body frame (module docstring)."""

    kind: str
    model: str = ""
    grip: Point3 | None = None
    tip: Point3 | None = None
    radius: float = 0.0
    origin: Point3 | None = None
    length: float = 0.0
    width: float = 0.0
    height: float = 0.0
    ahead: float = 0.0
    left: float = 0.0

    @classmethod
    def parse(cls, raw: object) -> DeviceSpec:
        if not isinstance(raw, Mapping) or raw.get("kind") not in KINDS:
            raise ValueError(f"device: kind must be one of {', '.join(KINDS)}, got {raw!r}")
        kind, model, keys = str(raw["kind"]), str(raw.get("model", "")), set(raw) - {"model"}
        if kind == CANE:
            if keys != _CANE_KEYS:
                raise ValueError(f"device ({kind}): needs exactly {', '.join(sorted(_CANE_KEYS))}[, model], got {', '.join(sorted(raw))}")
            radius = float(raw["radius"])
            if radius < 0.0:
                raise ValueError(f"device ({kind}): radius must not be negative, got {radius}")
            return cls(kind, model, grip=Point3.parse(raw["grip"], "device.grip"), tip=Point3.parse(raw["tip"], "device.tip"), radius=radius)
        size = raw.get("size")
        if keys != _BOX_KEYS or not isinstance(size, Mapping) or not {"length", "width", "height"} <= set(size) or set(size) - _SIZE_KEYS:
            raise ValueError(f"device ({kind}): needs exactly kind, origin and size {{length, width, height[, ahead, left]}}[, model], got {raw!r}")
        dims = {k: float(size.get(k, 0.0)) for k in _SIZE_KEYS}
        if min(dims["length"], dims["width"], dims["height"]) <= 0.0:
            raise ValueError(f"device ({kind}): length, width and height must be positive, got {dict(size)}")
        return cls(kind, model, origin=Point3.parse(raw["origin"], "device.origin"), **dims)

    def pose(self, x: float, y: float, z: float, theta: float, w: float, out: HeldDevice | None = None) -> HeldDevice:
        """This device held by a pedestrian standing at (x, y, z) heading `theta` at warped gait phase `w`, written into `out`."""
        msg = HeldDevice() if out is None else out
        c, s = math.cos(theta), math.sin(theta)

        def place(point: object, bx: float, by: float, bz: float) -> None:
            point.x, point.y, point.z = x + c * bx - s * by, y + s * bx + c * by, z + bz  # type: ignore[attr-defined]

        msg.kind, msg.model = self.kind, self.model
        if self.kind == CANE:
            assert self.grip is not None and self.tip is not None
            place(msg.start, *self.grip.at(w))
            place(msg.end, *self.tip.at(w))
            msg.centre.x, msg.centre.y, msg.centre.z = msg.end.x, msg.end.y, msg.end.z
            msg.yaw = math.atan2(msg.end.y - msg.start.y, msg.end.x - msg.start.x)
            msg.radius = self.radius
            msg.size.x = msg.size.y = msg.size.z = 0.0
            return msg
        assert self.origin is not None
        ox, oy, oz = self.origin.at(w)
        place(msg.start, ox, oy, oz)
        msg.end.x, msg.end.y, msg.end.z = msg.start.x, msg.start.y, msg.start.z
        place(msg.centre, ox + self.ahead, oy + self.left, 0.0)
        msg.yaw, msg.radius = theta, 0.0
        msg.size.x, msg.size.y, msg.size.z = self.length, self.width, self.height
        return msg


def footprint(device: HeldDevice) -> shapely.Geometry | None:
    """Plan view of a posed device: a cane's shaft as wide as it is thick, a box's footprint; None when it holds none or is lost (not finite)."""
    if not all(math.isfinite(v) for v in (device.start.x, device.start.y, device.end.x, device.end.y, device.centre.x, device.centre.y, device.yaw)):
        return None  # GEOS aborts the process on a NaN coordinate
    if device.kind == CANE:
        return shapely.LineString([(device.start.x, device.start.y), (device.end.x, device.end.y)]).buffer(max(device.radius, 1e-3))
    if device.kind in BOXES:
        c, s = math.cos(device.yaw), math.sin(device.yaw)
        hl, hw = device.size.x / 2.0, device.size.y / 2.0
        return shapely.Polygon([(device.centre.x + c * u - s * v, device.centre.y + s * u + c * v) for u, v in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))])
    return None


def shift(device: HeldDevice, dx: float, dy: float, dz: float) -> None:
    """Move a posed device by (dx, dy, dz), as its pedestrian moves into another frame."""
    if not device.kind:
        return
    for point in (device.start, device.end, device.centre):
        point.x, point.y, point.z = point.x + dx, point.y + dy, point.z + dz
