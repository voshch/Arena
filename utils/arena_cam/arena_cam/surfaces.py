"""Viewport surface addressing: which cameras a driver drives, and their env offsets.

Shared by `client.CamNode` (asyncio + `ClientWrapper`) and `drive.Driver` (plain
rclpy on a GUI-owned node): the handle types differ, the addressing does not.
"""

from __future__ import annotations

import dataclasses
import typing

from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from viewport_control_msgs.srv import ViewportSetReferenceFrame

if typing.TYPE_CHECKING:
    from collections.abc import Iterable

    from arena_runtime_msgs.msg import EnvRegistry

    from .curves import Quat, Vec3

ARENA_ROOT = "/arena"
ENVS_TOPIC = "/arena/state/envs"
VIEW_SUFFIX = "/viewport/set_view"

# Best-effort: a keyframe that arrives late is worse than one that never arrives.
STREAM_QOS = QoSProfile(depth=8, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.BEST_EFFORT)

# Match the latched EnvRegistry publisher so the env reference table arrives at once.
ENVS_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, durability=DurabilityPolicy.TRANSIENT_LOCAL)

REFERENCE_MODES = {
    "full": ViewportSetReferenceFrame.Request.FULL,
    "yaw": ViewportSetReferenceFrame.Request.YAW_ONLY,
    "position": ViewportSetReferenceFrame.Request.POSITION_ONLY,
}


@dataclasses.dataclass(frozen=True)
class TargetSelection:
    """Which viewport surfaces a driver drives. Resolved against the live graph at setup."""

    include_sim: bool
    viz_all: bool
    viz_env: int | None


def env_id_from_ns(ns: str) -> int | None:
    """Parse the env index from a viewport namespace (`/arena/env_3/...` -> 3)."""
    for part in ns.strip("/").split("/"):
        if part.startswith("env_"):
            try:
                return int(part[len("env_") :])
            except ValueError:
                return None
    return None


def env_refs(msg: EnvRegistry) -> dict[int, tuple[float, float]]:
    return {r.env_id: (float(r.reference[0]), float(r.reference[1])) for r in msg.envs}


def find_targets(service_names: Iterable[str], selection: TargetSelection, refs: dict[int, tuple[float, float]]) -> list[tuple[str, tuple[float, float]]]:
    """Match live `set_view` services against the selection, pairing each with its env offset."""
    out: list[tuple[str, tuple[float, float]]] = []
    for name in service_names:
        if not name.endswith(VIEW_SUFFIX):
            continue
        ns = name[: -len(VIEW_SUFFIX)]
        if ns == ARENA_ROOT:
            if selection.include_sim:
                out.append((ns, (0.0, 0.0)))
            continue
        env_id = env_id_from_ns(ns)
        if env_id is None:
            continue
        if selection.viz_all or selection.viz_env == env_id:
            out.append((ns, refs.get(env_id, (0.0, 0.0))))
    return out


def localize(offset: tuple[float, float], position: Vec3) -> Vec3:
    """World coords -> a surface's local frame (pure planar offset, no rotation)."""
    return (position[0] - offset[0], position[1] - offset[1], position[2])


def ros_pose(position: Vec3, quat: Quat) -> Pose:
    return Pose(
        position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
        orientation=Quaternion(w=float(quat[0]), x=float(quat[1]), y=float(quat[2]), z=float(quat[3])),
    )
