"""Pure-Python maneuver schedule definitions for open-loop characterization."""

from __future__ import annotations

import dataclasses
import logging
import math
import pathlib
from collections.abc import Sequence
from enum import StrEnum

import yaml

_log = logging.getLogger(__name__)

VX_MAX = 2.0  # m/s, default maximum rated linear speed
VX_STEP = 0.25  # m/s
VY_MAX = 0.0  # m/s (non-holonomic default)
VY_STEP = 0.25  # m/s
WZ_MAX = 2.5  # rad/s, default maximum rated angular rate
WZ_STEP = 0.5  # rad/s
RADIUS_M = 0.5  # m, default footprint radius, anchors the arc radii

LINEAR_DWELL_S = 5.0  # s, steady-state capture per velocity step
LINEAR_SETTLE_S = 1.0  # s, settle at rest between directions
LATERAL_DWELL_S = 5.0  # s, steady-state capture per lateral step
ANGULAR_DWELL_S = 5.0  # s, per pivot rate
RAMP_SETTLE_S = 1.0  # s, settle at the ramp apex before decelerating
BRAKE_APPROACH_S = 3.0  # s, cruise at the rated speed before the step to zero
BRAKE_DWELL_S = 3.0  # s, capture settling after step deceleration
IDLE_DURATION_S = 10.0  # s, mandatory standstill blocks (baseline standby draw)
MIN_DWELL_S = 3.0  # s, a step shorter than this holds no usable steady state

ARC_SPEED_FACTORS = (0.25, 0.5, 0.75)  # of vx_max
ARC_RADIUS_FACTORS = (1.0, 2.5, 6.0)  # of the footprint radius
RAMP_HORIZONS_S = (1.0,)  # s, one acceleration/deceleration family per horizon

MAX_EXCURSION_M = 10.0  # m, maximum excursion from spawn within one maneuver
MAX_ORBIT_S = 60.0  # s, longest closed arc orbit worth sampling

CONTROL_RATE_HZ = 20.0  # cmd_vel publish rate during a maneuver
ODOM_STALL_TIMEOUT_S = 3.0  # odom silent for this long: zero cmd_vel + abort
MAX_SCHEDULE_DURATION_S = 3600.0  # global safety ceiling for one run

MODES = ("idle", "linear", "lateral", "arc", "ramps", "brake", "angular")

DURATION_DEFAULTS = {  # keys are build_schedule's duration keywords
    "idle_s": IDLE_DURATION_S,
    "linear_dwell_s": LINEAR_DWELL_S,
    "linear_settle_s": LINEAR_SETTLE_S,
    "lateral_dwell_s": LATERAL_DWELL_S,
    "angular_dwell_s": ANGULAR_DWELL_S,
    "ramp_settle_s": RAMP_SETTLE_S,
    "brake_approach_s": BRAKE_APPROACH_S,
    "brake_dwell_s": BRAKE_DWELL_S,
}

SWEEP_DEFAULTS = {  # keys are build_schedule's sweep keywords
    "arc_speed_factors": ARC_SPEED_FACTORS,
    "arc_radius_factors": ARC_RADIUS_FACTORS,
    "ramp_horizons_s": RAMP_HORIZONS_S,
}


class PhaseKind(StrEnum):
    IDLE = "idle"
    SETTLE = "settle"
    LINEAR = "linear"
    LATERAL = "lateral"
    ANGULAR = "angular"
    ARC = "arc"
    RAMP_UP = "ramp_up"
    RAMP_APEX = "ramp_apex"
    RAMP_DOWN = "ramp_down"
    BRAKE_APPROACH = "brake_approach"
    BRAKE = "brake"
    TRANSIENT = "transient"  # assigned offline, never scheduled


@dataclasses.dataclass(frozen=True)
class Phase:
    """One maneuver block: a target twist held for a duration (optionally a ramp)."""

    kind: PhaseKind
    name: str  # unique label, e.g. "linear_vx_1.00", "arc_vx_1.00_r_1.00_left"
    vx_target: float = 0.0  # m/s
    vy_target: float = 0.0  # m/s (lateral / holonomic)
    wz_target: float = 0.0  # rad/s
    duration_s: float = 0.0  # s to hold the target (ramp: the ramp horizon)
    ramp_s: float = 0.0  # >0 for ramps: linearly interpolate vx from 0 to target over this horizon
    radius_m: float = 0.0  # m (turn radius for arc maneuvers)


def _steps(start: float, stop: float, step: float) -> list[float]:
    """Step range that never overshoots stop and always ends exactly on it."""
    if step <= 0.0 or stop <= start:
        return [round(stop, 6)]
    n = int((stop - start) / step + 1e-9)
    out = [round(start + i * step, 6) for i in range(n + 1)]
    if stop - out[-1] > 1e-3:
        out.append(round(stop, 6))
    return out


def _dwell(dwell_s: float, speed: float, label: str) -> float | None:
    """Dwell capped by the excursion budget, None when the budget cannot fund a steady state."""
    if speed <= 0.0:
        return None
    capped = min(dwell_s, MAX_EXCURSION_M / speed)
    if capped < MIN_DWELL_S:
        _log.warning(f"characterization: skipping {label}, the {MAX_EXCURSION_M:.0f}m budget funds only {capped:.2f}s (<{MIN_DWELL_S:.1f}s)")
        return None
    return capped


def resolve_envelope(
    robot_name: str | None = None,
    *,
    caps_dir: pathlib.Path | None = None,
    fallback: tuple[float, float] | None = None,
) -> dict[str, float | bool]:
    """Resolve the operating envelope (vx_max, vy_max, wz_max, radius, is_holonomic) for a robot."""
    vx_max = VX_MAX
    vy_max = VY_MAX
    wz_max = WZ_MAX
    radius = RADIUS_M
    is_holonomic = False

    if fallback is not None:
        vx_max, wz_max = fallback

    mobile: pathlib.Path | None = None
    try:
        if caps_dir is not None:
            mobile = pathlib.Path(caps_dir) / f"{robot_name}/caps/mobile.yaml" if robot_name else None
            if mobile is None or not mobile.is_file():
                return {"vx_max": vx_max, "vy_max": vy_max, "wz_max": wz_max, "radius": radius, "is_holonomic": is_holonomic}
        else:
            from ament_index_python.packages import get_package_share_directory

            mobile = pathlib.Path(get_package_share_directory("arena_robots")) / "robots" / (robot_name or "") / "caps" / "mobile.yaml"

        cfg = yaml.safe_load(mobile.read_text()) or {}
        is_holonomic = bool(cfg.get("is_holonomic", False))
        if cfg.get("radius") is not None:
            radius = float(cfg["radius"])

        continuous = cfg.get("actions", {}).get("continuous", {})
        if isinstance(continuous, dict):
            linear = continuous.get("linear")
            if isinstance(linear, dict) and linear.get("max") is not None:
                vx_max = float(linear["max"])
            lateral = continuous.get("lateral")
            if isinstance(lateral, dict) and lateral.get("max") is not None:
                vy_max = float(lateral["max"])
                is_holonomic = True
            elif is_holonomic:
                vy_max = vx_max
                _log.info(f"characterization: robot {robot_name!r} declares no lateral limit, sweeping vy up to vx_max={vy_max}")
            angular = continuous.get("angular")
            if isinstance(angular, dict) and angular.get("max") is not None:
                wz_max = float(angular["max"])
    except (OSError, KeyError, TypeError, ValueError, AttributeError, yaml.YAMLError) as e:
        _log.warning(f"characterization: robot {robot_name!r} envelope falls back to vx_max={vx_max} wz_max={wz_max} radius={radius}, could not read {mobile}: {e!r}")
    return {"vx_max": vx_max, "vy_max": vy_max, "wz_max": wz_max, "radius": radius, "is_holonomic": is_holonomic}


def build_schedule(
    *,
    modes: list[str] | None = None,
    idle_s: float = IDLE_DURATION_S,
    linear_dwell_s: float = LINEAR_DWELL_S,
    linear_settle_s: float = LINEAR_SETTLE_S,
    lateral_dwell_s: float = LATERAL_DWELL_S,
    angular_dwell_s: float = ANGULAR_DWELL_S,
    ramp_settle_s: float = RAMP_SETTLE_S,
    brake_approach_s: float = BRAKE_APPROACH_S,
    brake_dwell_s: float = BRAKE_DWELL_S,
    vx_max: float = VX_MAX,
    vx_step: float = VX_STEP,
    vy_max: float = VY_MAX,
    vy_step: float = VY_STEP,
    wz_min: float | None = None,
    wz_max: float = WZ_MAX,
    wz_step: float = WZ_STEP,
    radius: float = RADIUS_M,
    is_holonomic: bool = False,
    arc_speed_factors: Sequence[float] = ARC_SPEED_FACTORS,
    arc_radius_factors: Sequence[float] = ARC_RADIUS_FACTORS,
    ramp_horizons_s: Sequence[float] = RAMP_HORIZONS_S,
) -> list[Phase]:
    """Build one robot's maneuver schedule. Labels encode the working point (speed, radius, ramp horizon), never a dwell."""
    phases: list[Phase] = []
    active = set(modes) if modes is not None else set(MODES)
    if wz_min is None:
        wz_min = -wz_max

    def settle(name: str) -> None:
        if linear_settle_s > 0.0:
            phases.append(Phase(PhaseKind.SETTLE, name, duration_s=linear_settle_s))

    def pre_idle(name: str) -> None:
        if "idle" in active:
            phases.append(Phase(PhaseKind.IDLE, name, duration_s=idle_s / 2.0))

    if "idle" in active:
        phases.append(Phase(PhaseKind.IDLE, "idle_start", duration_s=idle_s))

    if "linear" in active:
        for vx in _steps(vx_step, vx_max, vx_step):
            tag = f"{vx:.2f}"
            dur = _dwell(linear_dwell_s, abs(vx), f"linear_vx_{tag}")
            if dur is None:
                continue
            phases.append(Phase(PhaseKind.LINEAR, f"linear_vx_{tag}", vx_target=vx, duration_s=dur))
            settle(f"linear_settle_{tag}")
            phases.append(Phase(PhaseKind.LINEAR, f"linear_vx_-{tag}", vx_target=-vx, duration_s=dur))
            settle(f"linear_settle_-{tag}")

    if "lateral" in active and is_holonomic and vy_max > 0.0:
        pre_idle("idle_lateral_pre")
        for vy in _steps(vy_step, vy_max, vy_step):
            tag = f"{vy:.2f}"
            dur = _dwell(lateral_dwell_s, abs(vy), f"lateral_vy_{tag}")
            if dur is None:
                continue
            phases.append(Phase(PhaseKind.LATERAL, f"lateral_vy_{tag}", vy_target=vy, duration_s=dur))
            settle(f"lateral_settle_{tag}")
            phases.append(Phase(PhaseKind.LATERAL, f"lateral_vy_-{tag}", vy_target=-vy, duration_s=dur))
            settle(f"lateral_settle_-{tag}")

    if "arc" in active and vx_max > 0.0 and radius > 0.0:
        pre_idle("idle_arc_pre")
        for speed_factor in arc_speed_factors:
            vx = round(speed_factor * vx_max, 6)
            for radius_factor in arc_radius_factors:
                r = round(radius_factor * radius, 6)
                tag = f"{vx:.2f}_r_{r:.2f}"
                if 2.0 * r > MAX_EXCURSION_M:
                    _log.warning(f"characterization: skipping arc_vx_{tag}, orbit diameter {2.0 * r:.2f}m exceeds the {MAX_EXCURSION_M:.0f}m budget")
                    continue
                wz = vx / r
                if wz > wz_max:
                    _log.warning(f"characterization: skipping arc_vx_{tag}, it needs wz={wz:.2f} rad/s over the rated {wz_max:.2f}")
                    continue
                t_orbit = round(2.0 * math.pi * r / vx, 2)
                if t_orbit > MAX_ORBIT_S:
                    _log.warning(f"characterization: skipping arc_vx_{tag}, one closed orbit takes {t_orbit:.1f}s (>{MAX_ORBIT_S:.0f}s)")
                    continue
                phases.append(Phase(PhaseKind.ARC, f"arc_vx_{tag}_left", vx_target=vx, wz_target=wz, duration_s=t_orbit, radius_m=r))
                settle(f"arc_settle_{tag}_left")
                phases.append(Phase(PhaseKind.ARC, f"arc_vx_{tag}_right", vx_target=vx, wz_target=-wz, duration_s=t_orbit, radius_m=r))
                settle(f"arc_settle_{tag}_right")

    if "ramps" in active:
        pre_idle("idle_ramps_pre")
        for horizon in ramp_horizons_s:
            if horizon <= 0.0:
                continue
            for vx in _steps(vx_step, vx_max, vx_step):
                for target in (vx, -vx):
                    tag = f"{target:.2f}_h_{horizon:.2f}"
                    phases.append(Phase(PhaseKind.RAMP_UP, f"ramp_up_vx_{tag}", vx_target=target, duration_s=horizon, ramp_s=horizon))
                    if ramp_settle_s > 0.0:
                        phases.append(Phase(PhaseKind.RAMP_APEX, f"ramp_apex_vx_{tag}", vx_target=target, duration_s=ramp_settle_s))
                    phases.append(Phase(PhaseKind.RAMP_DOWN, f"ramp_down_vx_{tag}", vx_target=target, duration_s=horizon, ramp_s=horizon))

    if "brake" in active:
        pre_idle("idle_brake_pre")
        for target in (vx_max, -vx_max):
            tag = f"{target:.2f}"
            phases.append(Phase(PhaseKind.BRAKE_APPROACH, f"brake_approach_vx_{tag}", vx_target=target, duration_s=brake_approach_s))
            phases.append(Phase(PhaseKind.BRAKE, f"brake_step_vx_{tag}", duration_s=brake_dwell_s))

    if "angular" in active:
        pre_idle("idle_angular_pre")
        for wz in _steps(wz_min, wz_max, wz_step):
            phases.append(Phase(PhaseKind.ANGULAR, f"angular_wz_{wz:+.2f}", wz_target=wz, duration_s=angular_dwell_s))

    if "idle" in active:
        phases.append(Phase(PhaseKind.IDLE, "idle_end", duration_s=idle_s))

    total = schedule_duration(phases)
    if total > MAX_SCHEDULE_DURATION_S:
        _log.warning(f"characterization: schedule runs {total:.0f}s, over the {MAX_SCHEDULE_DURATION_S:.0f}s ceiling, the sweep will abort part-way")

    return phases


def schedule_duration(phases: list[Phase]) -> float:
    """Total wall/sim time of a schedule (sum of phase durations)."""
    return sum(p.duration_s for p in phases)
