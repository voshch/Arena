"""Pure-Python maneuver schedule definitions for open-loop characterization."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
from enum import StrEnum

import yaml

_log = logging.getLogger(__name__)

VX_MIN = 0.0  # m/s
VX_MAX = 2.0  # m/s, default maximum rated linear speed
VX_STEP = 0.25  # m/s
LINEAR_DWELL_S = 5.0  # s, steady-state capture per velocity step
RAMP_HORIZONS_S = (0.5, 1.0, 2.0)  # s, zero-to-vx_max acceleration/deceleration horizons
RAMP_SETTLE_S = 1.0  # s, settle at the ramp apex before decelerating
WZ_MIN = -2.5  # rad/s
WZ_MAX = 2.5  # rad/s, default maximum rated angular rate
WZ_STEP = 0.5  # rad/s
ANGULAR_DWELL_S = 5.0  # s, per pivot rate
IDLE_DURATION_S = 10.0  # s, mandatory standstill blocks (baseline standby draw)

CONTROL_RATE_HZ = 20.0  # cmd_vel publish rate during a maneuver
ODOM_STALL_TIMEOUT_S = 3.0  # odom silent for this long: zero cmd_vel + abort
MAX_SCHEDULE_DURATION_S = 3600.0  # global safety ceiling for one run


class PhaseKind(StrEnum):
    IDLE = "idle"
    LINEAR = "linear"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    ANGULAR = "angular"


@dataclasses.dataclass(frozen=True)
class Phase:
    """One maneuver block: a target twist held for a duration (optionally a ramp)."""

    kind: PhaseKind
    name: str  # unique label, e.g. "linear_vx_1.00"
    vx_target: float  # m/s
    wz_target: float  # rad/s
    duration_s: float  # s to hold the target (ramp: the ramp horizon)
    ramp_s: float = 0.0  # >0 for ramps: linearly interpolate vx from 0 to target over this horizon

    @property
    def key(self) -> tuple[str, float, float]:
        """Grouping key for offline aggregation: (kind, vx_target, wz_target)."""
        return self.kind.value, round(self.vx_target, 3), round(self.wz_target, 3)


def _steps(start: float, stop: float, step: float) -> list[float]:
    """Floating-point safe step range (inclusive of stop)."""
    n = round((stop - start) / step) + 1
    return [round(start + i * step, 6) for i in range(n)]


def resolve_envelope(
    robot_name: str | None = None,
    *,
    caps_dir: pathlib.Path | None = None,
    fallback: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Resolve the operating envelope (vx_max, wz_max) for a robot."""
    vx_max = VX_MAX
    wz_max = WZ_MAX
    if fallback is not None:
        vx_max, wz_max = fallback

    mobile: pathlib.Path | None = None
    try:
        if caps_dir is not None:
            mobile = pathlib.Path(caps_dir) / f"{robot_name}/caps/mobile.yaml" if robot_name else None
            if mobile is None or not mobile.is_file():
                return {"vx_max": vx_max, "wz_max": wz_max}
        else:
            from ament_index_python.packages import get_package_share_directory

            mobile = pathlib.Path(get_package_share_directory("arena_robots")) / "robots" / (robot_name or "") / "caps" / "mobile.yaml"

        cfg = yaml.safe_load(mobile.read_text())
        continuous = (cfg or {}).get("actions", {}).get("continuous", {})
        if isinstance(continuous, dict):
            linear = continuous.get("linear")
            if isinstance(linear, dict) and linear.get("max") is not None:
                vx_max = float(linear["max"])
            angular = continuous.get("angular")
            if isinstance(angular, dict) and angular.get("max") is not None:
                wz_max = float(angular["max"])
    except (OSError, KeyError, TypeError, ValueError, AttributeError, yaml.YAMLError) as e:
        _log.warning(f"characterization: robot {robot_name!r} envelope falls back to vx_max={vx_max} wz_max={wz_max}, could not read {mobile}: {e!r}")
    return {"vx_max": vx_max, "wz_max": wz_max}


def build_schedule(
    *,
    idle_s: float = IDLE_DURATION_S,
    linear_dwell_s: float = LINEAR_DWELL_S,
    angular_dwell_s: float = ANGULAR_DWELL_S,
    ramp_horizons: tuple[float, ...] = RAMP_HORIZONS_S,
    vx_min: float = VX_MIN,
    vx_max: float = VX_MAX,
    vx_step: float = VX_STEP,
    wz_min: float = WZ_MIN,
    wz_max: float = WZ_MAX,
    wz_step: float = WZ_STEP,
) -> list[Phase]:
    """Build the full open-loop maneuver schedule."""
    phases: list[Phase] = []

    phases.append(Phase(PhaseKind.IDLE, "idle_start", 0.0, 0.0, idle_s))

    for vx in _steps(vx_step, vx_max, vx_step):
        phases.append(Phase(PhaseKind.LINEAR, f"linear_vx_{vx:.2f}", vx, 0.0, linear_dwell_s))
        phases.append(Phase(PhaseKind.LINEAR, f"linear_vx_-{vx:.2f}", -vx, 0.0, linear_dwell_s))

    phases.append(Phase(PhaseKind.IDLE, "idle_mid", 0.0, 0.0, idle_s))

    for horizon in ramp_horizons:
        phases.append(Phase(PhaseKind.RAMP_UP, f"ramp_up_{horizon:.1f}", vx_max, 0.0, horizon, ramp_s=horizon))
        phases.append(Phase(PhaseKind.LINEAR, f"ramp_apex_{horizon:.1f}", vx_max, 0.0, RAMP_SETTLE_S))
        phases.append(Phase(PhaseKind.RAMP_UP, f"ramp_back_{horizon:.1f}", -vx_max, 0.0, horizon, ramp_s=horizon))

    phases.append(Phase(PhaseKind.IDLE, "idle_mid2", 0.0, 0.0, idle_s))

    for wz in _steps(wz_min, wz_max, wz_step):
        phases.append(Phase(PhaseKind.ANGULAR, f"angular_wz_{wz:+.2f}", 0.0, wz, angular_dwell_s))

    phases.append(Phase(PhaseKind.IDLE, "idle_end", 0.0, 0.0, idle_s))

    return phases


def schedule_duration(phases: list[Phase]) -> float:
    """Total wall/sim time of a schedule (sum of phase durations)."""
    return sum(p.duration_s for p in phases)


def classify_cmd_point(vx_cmd: float, wz_cmd: float) -> tuple[PhaseKind, float, float]:
    """Offline fallback: classify a (cmd_vel) sample into a phase key."""
    vx = round(float(vx_cmd or 0.0), 3)
    wz = round(float(wz_cmd or 0.0), 3)
    if vx == 0.0 and wz == 0.0:
        return PhaseKind.IDLE, 0.0, 0.0
    if wz != 0.0:
        return PhaseKind.ANGULAR, 0.0, wz
    return PhaseKind.LINEAR, vx, 0.0
