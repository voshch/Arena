"""Encounters in an open world: on the longest straight chords through its free space, at any crossing angle."""

from __future__ import annotations

import math

import shapely

from arena_simulation_setup.generative.world.layout import Layout

from .encounter import CLEARANCE, CROSSING, ENCOUNTER_AT, LEAD_OUT_M, ROBOT_RADIUS, Generated, Leg, Rejected, build, crossed, crossing_direction, fits, parallel, summary
from .sampler import Sample

HEADINGS = (0.0, 90.0, 45.0, 135.0, 180.0, 270.0, 225.0, 315.0)  # degrees tried for the robot's leg
CHORD_STEP_M = 1.0  # spacing of parallel chords across the world
MIN_CHORD_M = 8.0


def chords(free: shapely.geometry.base.BaseGeometry, margin: float) -> list[Leg]:
    """Straight legs through `free` eroded by `margin`, longest and most central first."""
    room = free.buffer(-margin)
    if room.is_empty:
        return []
    minx, miny, maxx, maxy = free.bounds
    center = ((minx + maxx) / 2, (miny + maxy) / 2)
    reach = math.hypot(maxx - minx, maxy - miny)
    found: list[tuple[float, float, Leg]] = []
    for degrees in HEADINGS:
        theta = math.radians(degrees)
        u, n = (math.cos(theta), math.sin(theta)), (-math.sin(theta), math.cos(theta))
        steps = int(reach / 2 / CHORD_STEP_M)
        for k in range(-steps, steps + 1):
            c = (center[0] + n[0] * k * CHORD_STEP_M, center[1] + n[1] * k * CHORD_STEP_M)
            line = shapely.LineString([(c[0] - u[0] * reach, c[1] - u[1] * reach), (c[0] + u[0] * reach, c[1] + u[1] * reach)])
            pieces = room.intersection(line)
            for piece in pieces.geoms if isinstance(pieces, shapely.MultiLineString | shapely.GeometryCollection) else [pieces]:
                if not isinstance(piece, shapely.LineString) or piece.length < MIN_CHORD_M:
                    continue
                a, b = piece.coords[0], piece.coords[-1]
                if (b[0] - a[0]) * u[0] + (b[1] - a[1]) * u[1] < 0:
                    a, b = b, a
                found.append((-piece.length, abs(k), Leg(start=(a[0], a[1]), end=(b[0], b[1]))))
    return [leg for _, _, leg in sorted(found, key=lambda item: (item[0], item[1]))]


def generate(layout: Layout, sample: Sample) -> Generated | Rejected:
    """The first placement of `sample` on a chord of the open world that fits, else why none did."""
    free = layout.geometry()
    crossing = sample.encounter in CROSSING
    margin = ROBOT_RADIUS + CLEARANCE + (0.0 if crossing else abs(sample.lateral_offset) / 2)
    reasons: list[str] = []
    for leg in chords(free, margin):
        if crossing:
            arc = ENCOUNTER_AT * leg.length
            v = crossing_direction(sample, leg)
            meet = leg.at(arc)
            placement = crossed(sample, leg, arc, [(meet[0] + v[0] * LEAD_OUT_M, meet[1] + v[1] * LEAD_OUT_M)])
        else:
            placement = parallel(sample, leg)
        if isinstance(placement, Rejected):
            reasons.append(placement.reason)
            continue
        why = fits(placement, free)
        if why is None:
            return build(sample, placement, notes=[f"open chord of {leg.length:.2f} m at {math.degrees(leg.heading):.0f} degrees"])
        reasons.append(why)
    return Rejected(summary(reasons) or f"no straight chord of {MIN_CHORD_M:g} m fits the open world with {margin:.2f} m clearance")
