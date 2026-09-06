"""Decaying directional pedestrian-likelihood grid, in the map frame.

Pure numpy: no ROS imports, so the same update runs inside the belief node and
inside offline replay tooling. That is deliberate: analysis figures and the
live layer must be the same estimator.

Model
-----
The layer consumes only what a sound front-end can produce: an event tuple of
(type, bearing, received level, timestamp). It never reads emitter identity,
source position or true range, even when the simulator's ``HeardSoundEvent``
carries them.

Each event paints a wedge from the robot pose along the reported bearing:

* angular extent ``wedge_deg`` (default 10 deg), weighted by a Gaussian in
  lateral offset with sigma = half-width / 2, where the half-width at forward
  distance ``d`` is ``max(d * tan(wedge / 2), min_half_width_m)``: the wedge
  is never narrower than the array footprint (default 0.3 m), so the cells
  around the robot carry full weight instead of a one-cell sliver;
* radial extent out to ``R``.  With ``use_level_range`` the range is implied by
  inverting the simulator's spreading law
  ``received = emission - 20 log10(max(r, 1 m))`` against a per-type emission
  level, clamped to ``[min_range_m, max_range_m]``.  By default (off, the
  corpus finding: level is not a range cue, and a wall costs ~7 dB on the live
  bus so an occluded pedestrian sounds twice as far), or when the event carries
  no usable level, ``R = max_range_m`` (default 15 m, the median per-episode
  maximum source range in the recorded corpus) and the radial profile is flat;
* radial weighting when ``R`` is level-derived: a Gaussian bump at ``R`` with
  sigma = ``range_sigma_frac * R`` (the level-to-range inversion is coarse),
  plus a flat floor ``range_floor_weight`` over ``[0, R]`` so the whole wedge
  still carries mass.  Without the bump the argmax along the ray would be
  arbitrary and any attribution rate meaningless.

The wedge is painted **through walls on purpose**: an occluded pedestrian is
audible, and the belief must be able to put mass behind a corner where no
range sensor can see.  It is masked by ``free`` so mass never lands in an
occupied cell itself, only in the free cells the wedge crosses into.

Mass decays as ``exp(-dt / tau_sec)`` (default 2 s), so a source that stops
emitting fades rather than latching.

Outputs
-------
``normalized()``  belief in [0, 1] = mass / full scale, clipped.  Full scale is
the steady state of one source emitting at ``nominal_event_rate_hz``, i.e.
``event_mass * nominal_event_rate_hz * tau_sec``, so the normalisation follows
the front-end's event rate instead of being retuned by hand.
``belief_int8()``  that, times 100, as an ``OccupancyGrid`` payload for RViz.
``speed_mask_int8()``  the Nav2 SpeedFilter mask.  The filter reads the mask at
the robot's own cell, so the mask is the belief max-filtered over a disc of
``reaction_radius_m`` before thresholding: the robot is slowed while likely
pedestrian mass lies within that radius of it.  Nav2 semantics (see
``nav2_costmap_2d/costmap_filters/filter_values.hpp``): mask value 0 =
``SPEED_MASK_NO_LIMIT`` (no restriction), -1 = unknown, and any other value v
gives ``speed_limit = base + multiplier * v``, which for filter type 1 is a
*percentage of maximum speed*.  So a **high** mask value means **fast**.  Cells
below ``belief_threshold`` therefore get 0 (free), and cells above it get
``speed_free_pct`` linearly pulled down to ``speed_min_pct`` as the belief goes
from the threshold to 1.  The mask never emits a value below ``speed_min_pct``
(and never 0 for a restricted cell), so this layer slows the robot and never
commands a hard stop through the filter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import maximum_filter1d

if TYPE_CHECKING:
    from nav_msgs.msg import MapMetaData

# Nav2 costmap_filters/filter_values.hpp
SPEED_MASK_NO_LIMIT = 0
SPEED_MASK_UNKNOWN = -1

# Arena human_sound_node.py emission levels at the 1 m reference distance.
DEFAULT_EMISSION_DB: dict[str, float] = {
    "footstep": 45.0,
    "greeting": 60.0,
    "speech": 60.0,
}


@dataclass
class BeliefParams:
    """Every knob is a ROS parameter of the same name on the node."""

    wedge_deg: float = 10.0
    min_half_width_m: float = 0.3
    tau_sec: float = 2.0
    max_range_m: float = 15.0
    min_range_m: float = 0.5
    reference_distance_m: float = 1.0
    use_level_range: bool = False
    range_sigma_frac: float = 0.35
    range_floor_weight: float = 0.15
    event_mass: float = 1.0
    # belief 1.0 = one source at nominal_event_rate_hz, i.e. event_mass * f * tau; mass_full_scale > 0 overrides
    nominal_event_rate_hz: float = 2.0
    mass_full_scale: float = 0.0
    belief_threshold: float = 0.15
    speed_min_pct: int = 20
    speed_free_pct: int = 100
    reaction_radius_m: float = 4.0
    emission_db: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EMISSION_DB))
    default_emission_db: float = float("nan")  # nan -> unknown types use the flat fallback

    def full_scale(self) -> float:
        if self.mass_full_scale > 0.0:
            return float(self.mass_full_scale)
        return max(
            float(self.event_mass) * float(self.nominal_event_rate_hz) * float(self.tau_sec),
            1e-9,
        )

    def emission_for(self, sound_type: str) -> float:
        key = (sound_type or "").strip().lower()
        if key in self.emission_db:
            return float(self.emission_db[key])
        return float(self.default_emission_db)


def wrap_pi(a: np.ndarray | float) -> np.ndarray:
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class BeliefGrid:
    """Peak-normalised likelihood grid over a fixed map-frame rectangle."""

    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        resolution: float,
        width: int,
        height: int,
        params: BeliefParams | None = None,
        free: np.ndarray | None = None,
    ) -> None:
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.params = params or BeliefParams()
        self.mass = np.zeros((self.height, self.width), dtype=np.float32)
        self.free = np.ones((self.height, self.width), dtype=bool) if free is None else free
        self._xs = self.origin_x + (np.arange(self.width, dtype=np.float64) + 0.5) * self.resolution
        self._ys = self.origin_y + (np.arange(self.height, dtype=np.float64) + 0.5) * self.resolution
        self.n_events = 0

    @classmethod
    def from_occupancy_info(cls, info: MapMetaData, params: BeliefParams | None = None, free: np.ndarray | None = None) -> BeliefGrid:
        """``info`` is a ``nav_msgs/MapMetaData``."""
        return cls(
            origin_x=info.origin.position.x,
            origin_y=info.origin.position.y,
            resolution=info.resolution,
            width=info.width,
            height=info.height,
            params=params,
            free=free,
        )

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + (row + 0.5) * self.resolution,
        )

    def decay(self, dt: float) -> None:
        if dt <= 0.0:
            return
        tau = max(float(self.params.tau_sec), 1e-6)
        self.mass *= float(math.exp(-dt / tau))

    def clear(self) -> None:
        self.mass[:] = 0.0
        self.n_events = 0

    def range_estimate(self, sound_type: str, received_db: float | None) -> tuple[float, bool]:
        """Return (range_m, level_derived).

        Inverts the simulator's spreading law.  ``level_derived`` False means the
        fallback flat wedge out to ``max_range_m`` is in force.
        """
        p = self.params
        emission = p.emission_for(sound_type)
        if not p.use_level_range or received_db is None or not np.isfinite(received_db) or not np.isfinite(emission):
            return float(p.max_range_m), False
        r = float(p.reference_distance_m) * 10.0 ** ((emission - float(received_db)) / 20.0)
        if not np.isfinite(r):
            return float(p.max_range_m), False
        return float(np.clip(r, p.min_range_m, p.max_range_m)), True

    def add_event(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        bearing_rad: float,
        sound_type: str = "",
        received_db: float | None = None,
        weight: float = 1.0,
        bearing_frame: str = "robot",
    ) -> dict:
        """Paint one event.  ``bearing_rad`` is CCW-positive.

        ``bearing_frame='robot'`` (the front-end / label convention, and the
        ``bearing_robot_rad`` column) means the consumer rotates the bearing by
        the robot's yaw into the map frame; ``'map'`` takes the bearing as
        already absolute, which is what Arena's ``HeardSoundEvent.bearing_rad``
        carries by construction.
        """
        p = self.params
        theta = float(bearing_rad) + (float(robot_yaw) if bearing_frame == "robot" else 0.0)
        rng, level_derived = self.range_estimate(sound_type, received_db)

        half = math.radians(float(p.wedge_deg)) * 0.5
        tan_half = math.tan(min(half, math.radians(89.0)))
        min_hw = max(float(p.min_half_width_m), 0.0)
        sigma_rad = max(float(p.range_sigma_frac) * rng, self.resolution)
        r_paint = min(float(p.max_range_m), rng + 2.5 * sigma_rad) if level_derived else rng

        # bounding box in cells
        r0, c0 = self.world_to_cell(robot_x - r_paint, robot_y - r_paint)
        r1, c1 = self.world_to_cell(robot_x + r_paint, robot_y + r_paint)
        r0 = max(r0, 0)
        c0 = max(c0, 0)
        r1 = min(r1 + 1, self.height)
        c1 = min(c1 + 1, self.width)
        if r1 <= r0 or c1 <= c0:
            return {"painted": 0, "range_m": rng, "level_derived": level_derived, "theta": theta}

        dx = self._xs[c0:c1][None, :] - robot_x
        dy = self._ys[r0:r1][:, None] - robot_y
        rr = np.hypot(dx, dy)
        ct, st = math.cos(theta), math.sin(theta)
        fwd = dx * ct + dy * st
        lat = np.abs(-dx * st + dy * ct)
        hw = np.maximum(fwd * tan_half, min_hw)

        inside = (rr <= r_paint) & (fwd >= 0.0) & (lat <= hw)
        if not inside.any():
            return {"painted": 0, "range_m": rng, "level_derived": level_derived, "theta": theta}

        w_ang = np.exp(-0.5 * (lat / np.maximum(hw * 0.5, 1e-3)) ** 2)
        if level_derived:
            w_rad = np.exp(-0.5 * ((rr - rng) / sigma_rad) ** 2)
            w_rad = w_rad + float(p.range_floor_weight) * (rr <= rng)
        else:
            w_rad = np.ones_like(rr)
        w = np.where(inside, w_ang * w_rad, 0.0)
        peak = float(w.max())
        if peak <= 0.0:
            return {"painted": 0, "range_m": rng, "level_derived": level_derived, "theta": theta}
        # peak-normalised: one event deposits event_mass at its likeliest cell whatever the wedge size
        w *= (float(p.event_mass) * float(weight)) / peak
        w *= self.free[r0:r1, c0:c1]

        self.mass[r0:r1, c0:c1] += w.astype(np.float32)
        self.n_events += 1
        return {
            "painted": int(inside.sum()),
            "range_m": rng,
            "level_derived": level_derived,
            "theta": theta,
        }

    def normalized(self) -> np.ndarray:
        return np.clip(self.mass / self.params.full_scale(), 0.0, 1.0)

    def belief_int8(self) -> np.ndarray:
        return np.round(self.normalized() * 100.0).astype(np.int8)

    def dilated(self) -> np.ndarray:
        return dilate_belief(self.normalized(), self.resolution, self.params.reaction_radius_m)

    def speed_mask_int8(self) -> np.ndarray:
        return speed_mask_from_belief(self.normalized(), self.resolution, self.params)

    def argmax_world(self) -> tuple[float, float, float]:
        """(x, y, mass) of the maximum-mass cell.  mass 0.0 if the grid is empty."""
        if not np.isfinite(self.mass).any() or float(self.mass.max()) <= 0.0:
            return float("nan"), float("nan"), 0.0
        idx = int(np.argmax(self.mass))
        row, col = divmod(idx, self.width)
        x, y = self.cell_to_world(row, col)
        return x, y, float(self.mass[row, col])


def dilate_belief(belief: np.ndarray, resolution: float, reaction_radius_m: float) -> np.ndarray:
    """Max filter over the ``reaction_radius_m`` disc: one row-wise 1D filter per row offset, O(cells * radius)."""
    k = int(round(float(reaction_radius_m) / resolution))
    if k <= 0:
        return belief
    out = np.zeros_like(belief)
    h = belief.shape[0]
    for dy in range(-k, k + 1):
        half = int(math.floor(math.sqrt(k * k - dy * dy)))
        row = maximum_filter1d(belief, 2 * half + 1, axis=1, mode="constant", cval=0.0)
        src0, src1 = max(0, -dy), min(h, h - dy)
        if src1 > src0:
            np.maximum(out[src0 + dy : src1 + dy], row[src0:src1], out=out[src0 + dy : src1 + dy])
    return out


def speed_mask_from_belief(belief: np.ndarray, resolution: float, params: BeliefParams) -> np.ndarray:
    """Nav2 SpeedFilter mask from a normalised belief: 0 = no limit, else percentage of max speed.
    Cells below ``belief_threshold`` after dilation are free, above it the percentage falls linearly
    from ``speed_free_pct`` to ``speed_min_pct`` as the belief goes from the threshold to 1."""
    b = dilate_belief(belief, resolution, params.reaction_radius_m)
    thr = float(np.clip(params.belief_threshold, 0.0, 0.999))
    span = max(1.0 - thr, 1e-6)
    frac = np.clip((b - thr) / span, 0.0, 1.0)
    pct = float(params.speed_free_pct) - frac * (float(params.speed_free_pct) - float(params.speed_min_pct))
    pct = np.clip(np.round(pct), params.speed_min_pct, params.speed_free_pct)
    return np.where(b > thr, pct, SPEED_MASK_NO_LIMIT).astype(np.int8)
