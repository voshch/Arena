"""Blind bends along a plan: where free space around the path ahead is hidden from the robot."""

from __future__ import annotations

import attrs
import numpy as np


@attrs.frozen
class BlindBend:
    bend_xy: tuple[float, float]
    direction: tuple[float, float]  # unit path direction at the bend
    bend_s: float  # arclength of the bend along the plan
    plan_s: np.ndarray  # arclength of every plan point
    approach_xy: np.ndarray  # (n, 2) plan points on the approach
    hold_xy: np.ndarray  # (n, 2) plan points of the hold band

    def dist_from(self, plan_xy: np.ndarray, robot_xy: tuple[float, float]) -> float:
        """Along-plan distance from the robot to the bend, negative once past it."""
        i = int(np.argmin(np.linalg.norm(plan_xy - np.asarray(robot_xy), axis=1)))
        return float(self.bend_s - self.plan_s[i])


def occupied_cells_on_line(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    """Occupied cells crossed by the Bresenham line between two cells, endpoints included."""
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 >= r0 else -1), (1 if c1 >= c0 else -1)
    err = dc - dr
    r, c = r0, c0
    n = 0
    h, w = occupied.shape
    while True:
        if 0 <= r < h and 0 <= c < w and occupied[r, c]:
            n += 1
        if r == r1 and c == c1:
            return n
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def hidden_fraction(occupied: np.ndarray, origin: tuple[float, float], resolution: float, src_xy: np.ndarray, targets_xy: np.ndarray) -> float:
    """Fraction of ``targets_xy`` whose straight line from ``src_xy`` crosses an occupied cell."""
    if len(targets_xy) == 0:
        return 0.0
    h, w = occupied.shape
    d = targets_xy - src_xy[None, :]
    steps = int(np.ceil(np.abs(d).max() / (0.5 * resolution))) + 1
    t = np.linspace(0.0, 1.0, steps)[None, :, None]
    pts = src_xy[None, None, :] + d[:, None, :] * t
    cols = np.clip(((pts[..., 0] - origin[0]) / resolution).astype(int), 0, w - 1)
    rows = np.clip(((pts[..., 1] - origin[1]) / resolution).astype(int), 0, h - 1)
    return float(np.mean(occupied[rows, cols].any(axis=1)))


def free_cells_near(occupied: np.ndarray, origin: tuple[float, float], resolution: float, centre_xy: np.ndarray, radius_m: float, limit: int = 64) -> np.ndarray:
    """Up to ``limit`` free cell centres within ``radius_m`` of ``centre_xy``, evenly subsampled."""
    h, w = occupied.shape
    k = int(np.ceil(radius_m / resolution))
    c0 = int((centre_xy[0] - origin[0]) / resolution)
    r0 = int((centre_xy[1] - origin[1]) / resolution)
    rr, cc = np.mgrid[max(r0 - k, 0) : min(r0 + k + 1, h), max(c0 - k, 0) : min(c0 + k + 1, w)]
    xs = origin[0] + (cc + 0.5) * resolution
    ys = origin[1] + (rr + 0.5) * resolution
    inside = ((xs - centre_xy[0]) ** 2 + (ys - centre_xy[1]) ** 2 <= radius_m * radius_m) & ~occupied[rr, cc]
    pts = np.stack([xs[inside], ys[inside]], axis=1)
    if len(pts) > limit:
        pts = pts[np.linspace(0, len(pts) - 1, limit).astype(int)]
    return pts


def find_blind_bend(
    plan_xy: np.ndarray,
    occupied: np.ndarray,
    origin: tuple[float, float],
    resolution: float,
    robot_xy: tuple[float, float],
    *,
    lookahead_m: float,
    approach_m: float,
    hold_len_m: float,
    hold_offset_m: float,
    probe_radius_m: float = 2.0,
    hidden_threshold: float = 0.25,
    step_m: float = 0.25,
) -> BlindBend | None:
    """The first bend ahead of the robot. A plan point is blind when more than ``hidden_threshold``
    of the free space within ``probe_radius_m`` of the point ``lookahead_m`` further along the plan
    (or of the plan end), on the far side of that point, is hidden from it. The bend is the first
    point from which that clears, or the plan end when the plan ends in the blind stretch (a goal
    sitting at a corner)."""
    plan = np.asarray(plan_xy, dtype=np.float64)
    if plan.ndim != 2 or len(plan) < 2:
        return None
    seg = np.linalg.norm(np.diff(plan, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    start = int(np.argmin(np.linalg.norm(plan - np.asarray(robot_xy), axis=1)))
    blind_seen = False
    bend = None
    next_s = s[start]
    for i in range(start, len(plan)):
        if s[i] < next_s:
            continue
        next_s = s[i] + step_m
        j = min(int(np.searchsorted(s, s[i] + lookahead_m)), len(plan) - 1)
        if j == i:
            break
        targets = free_cells_near(occupied, origin, resolution, plan[j], probe_radius_m)
        ahead = plan[j] - plan[i]
        targets = targets[(targets - plan[j]) @ ahead >= 0.0]
        blind = hidden_fraction(occupied, origin, resolution, plan[i], targets) >= hidden_threshold
        if blind:
            blind_seen = True
        elif blind_seen:
            bend = i
            break
    if bend is None:
        if not blind_seen:
            return None
        bend = len(plan) - 1
    s_b = s[bend]
    approach = plan[(s >= s_b - approach_m) & (s <= s_b)]
    lo, hi = s_b - hold_offset_m - hold_len_m, s_b - hold_offset_m
    if s[start] > hi:
        lo, hi = s[start], s_b  # already inside the offset: creep the rest of the way
    hold = plan[(s >= lo) & (s <= hi)]
    lo, hi = max(bend - 1, 0), min(bend + 1, len(plan) - 1)
    d = plan[hi] - plan[lo]
    norm = float(np.linalg.norm(d))
    direction = (float(d[0] / norm), float(d[1] / norm)) if norm > 1e-9 else (1.0, 0.0)
    return BlindBend(
        bend_xy=(float(plan[bend, 0]), float(plan[bend, 1])),
        direction=direction,
        bend_s=float(s_b),
        plan_s=s,
        approach_xy=approach,
        hold_xy=hold,
    )
