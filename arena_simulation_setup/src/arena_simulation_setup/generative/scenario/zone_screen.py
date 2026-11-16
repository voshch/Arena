"""Zones a pedestrian may start in by name: every point the runtime resolver could return there clears the route."""

from __future__ import annotations

import typing
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs

from arena_simulation_setup.generative.core.bodies import RUNTIME_SAFE_DIST

if typing.TYPE_CHECKING:
    import numpy as np

    from arena_simulation_setup.tree.World.model import World

from .occupancy import level_grid


@attrs.frozen(kw_only=True)
class Verdict:
    zone: str
    ref_safe: bool
    reason: str = ""
    admissible: int = 0
    rejected: int = 0

    def __bool__(self) -> bool:
        return self.ref_safe


def rasterize(poly: Sequence[tuple[float, float]], shape: tuple[int, int], res: float, ox: float, oy: float) -> np.ndarray:
    import numpy as np
    from PIL import Image, ImageDraw

    height, width = shape
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(
        [(int((x - ox) / res), height - 1 - int((y - oy) / res)) for x, y in poly],
        fill=1,
    )
    return np.asarray(canvas, dtype=bool)


def _cell_centers(mask: np.ndarray, res: float, ox: float, oy: float) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    rows, cols = np.nonzero(mask)
    height = mask.shape[0]
    return (cols + 0.5) * res + ox, (height - 1 - rows + 0.5) * res + oy


def route_clearance(xs: np.ndarray, ys: np.ndarray, route: Sequence[tuple[float, float]]) -> np.ndarray:
    import numpy as np

    if not route:
        return np.full(xs.shape, np.inf)
    if len(route) == 1:
        return np.hypot(xs - route[0][0], ys - route[0][1])

    best = np.full(xs.shape, np.inf)
    for (ax, ay), (bx, by) in zip(route, route[1:], strict=False):
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        if denom == 0:
            best = np.minimum(best, np.hypot(xs - ax, ys - ay))
            continue
        t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / denom, 0.0, 1.0)
        best = np.minimum(best, np.hypot(xs - (ax + t * dx), ys - (ay + t * dy)))
    return best


def screen_zones(
    zones: Sequence[Mapping[str, Any]],
    *,
    world_root: str | Path,
    route: Sequence[tuple[float, float]],
    statics: Sequence[Mapping[str, Any]] = (),
    min_route_clearance: float,
    robot_zones: frozenset[str] = frozenset(),
    clearance_map: tuple[np.ndarray, float, float, float] | None = None,
    world: World | None = None,
) -> dict[str, Verdict]:
    """A verdict per zone on whether every point the resolver could return there clears the route, `robot_zones` refused."""
    verdicts: dict[str, Verdict] = {}
    names = [str(z["name"]) for z in zones if z.get("name")]

    if clearance_map is None:
        grid = level_grid(world_root, statics, world)
        if grid is None:
            return {n: Verdict(zone=n, ref_safe=False, reason="no occupancy grid to screen against") for n in names}
        occupied, res, ox, oy = grid

        import numpy as np
        import scipy.ndimage

        edt = scipy.ndimage.distance_transform_edt(~occupied)
        assert isinstance(edt, np.ndarray)
        dist = edt * res
    else:
        dist, res, ox, oy = clearance_map

    for zone in zones:
        if not zone.get("name"):
            continue
        name = str(zone["name"])
        poly = [(float(c["x"]), float(c["y"])) for c in zone.get("corners", [])]
        if len(poly) < 3:
            verdicts[name] = Verdict(zone=name, ref_safe=False, reason="fewer than three corners")
            continue

        mask = rasterize(poly, dist.shape, res, ox, oy)
        if not mask.any():
            verdicts[name] = Verdict(zone=name, ref_safe=False, reason="polygon covers no grid cell")
            continue

        clearance = dist[mask]
        admissible = clearance > RUNTIME_SAFE_DIST
        n_admissible = int(admissible.sum())
        if n_admissible == 0:
            verdicts[name] = Verdict(
                zone=name,
                ref_safe=False,
                reason="no point in the zone clears SAFE_DIST, so the resolver would warn and return a blocked point",
            )
            continue

        if name in robot_zones:
            verdicts[name] = Verdict(
                zone=name,
                ref_safe=False,
                admissible=n_admissible,
                rejected=n_admissible,
                reason="the robot's own spec names this zone as an endpoint, so it may start or finish anywhere inside it",
            )
            continue

        xs, ys = _cell_centers(mask, res, ox, oy)
        xs, ys, clearance = xs[admissible], ys[admissible], clearance[admissible]

        route_bad = route_clearance(xs, ys, route) < min_route_clearance
        n_rejected = int(route_bad.sum())

        if n_rejected == 0:
            verdicts[name] = Verdict(zone=name, ref_safe=True, admissible=n_admissible)
            continue

        verdicts[name] = Verdict(
            zone=name,
            ref_safe=False,
            admissible=n_admissible,
            rejected=n_rejected,
            reason=f"{n_rejected}/{n_admissible} resolvable points sit within {min_route_clearance} m of the robot's route",
        )

    return verdicts


def ref_safe_zones(verdicts: Mapping[str, Verdict]) -> frozenset[str]:
    return frozenset(name for name, v in verdicts.items() if v.ref_safe)
