"""Cameras fixed on a TF frame, recorded next to a take's cameras at the same sim time."""

from __future__ import annotations

from dataclasses import dataclass

from . import curves
from .curves import Quat, Vec3

SPEC_FORM = "FRAME[@X,Y,Z[,PITCH]]"


@dataclass(frozen=True)
class Mount:
    """A camera at `at` in `frame` (m, x forward, z up), looking along the frame's x axis tilted down by `pitch` rad."""

    frame: str
    fov: float
    at: Vec3 = (0.0, 0.0, 0.0)
    pitch: float = 0.0

    @property
    def tag(self) -> str:
        return self.frame.strip("/").replace("/", "_")

    def local_pose(self) -> tuple[Vec3, Quat]:
        return self.at, curves.quat_from_euler(0.0, self.pitch, 0.0)


def parse_mount(spec: str, fov: float) -> Mount:
    """`FRAME`, `FRAME@X,Y,Z` or `FRAME@X,Y,Z,PITCH`."""
    frame, sep, rest = spec.partition("@")
    if not frame:
        raise ValueError(f"mount {spec!r} names no frame, expected {SPEC_FORM}")
    if not sep:
        return Mount(frame, fov)
    try:
        values = [float(part) for part in rest.split(",")]
    except ValueError:
        raise ValueError(f"mount {spec!r} has a non-number after @, expected {SPEC_FORM}") from None
    if len(values) not in (3, 4):
        raise ValueError(f"mount {spec!r} takes 3 or 4 numbers after @, expected {SPEC_FORM}")
    x, y, z = values[:3]
    return Mount(frame, fov, (x, y, z), values[3] if len(values) == 4 else 0.0)
