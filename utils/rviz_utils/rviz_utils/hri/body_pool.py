"""Pooled robot_state_publisher lifecycle manager for ROS4HRI body frames.

Design: POOLED RSP SUBPROCESSES.

Each slot in the pool holds a pre-spawned robot_state_publisher process.
On body assignment the slot receives a new robot_description (via `--ros-args
-p robot_description:=...`) and a remapped joint_states topic.  Recycling
reuses the process slot rather than spawning and tearing down an OS process
each episode, which matters when pedestrian counts oscillate across resets.

Why not FK-in-producer: robot_state_publisher already performs correct URDF
parsing, joint-limit-free FK, and TF broadcasting.  Re-implementing that
inside the producer would duplicate well-tested C++ logic and couple us to
urdf_parser_py version skew.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import subprocess
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import xacro
import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from urdf_parser_py import urdf as urdf_parser

if TYPE_CHECKING:
    import rclpy.node

_LOG = logging.getLogger(__name__)

_DEFAULT_HEIGHT = 1.65


def _origin_matrix(origin: urdf_parser.Pose | None) -> np.ndarray:
    """4x4 homogeneous transform for a URDF joint origin (xyz + rpy)."""
    m = np.eye(4)
    if origin is None:
        return m
    r, p, y = origin.rpy or (0.0, 0.0, 0.0)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    m[:3, :3] = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )
    m[:3, 3] = origin.xyz or (0.0, 0.0, 0.0)
    return m


def _hip_to_foot_offset(urdf_xml: str) -> float:
    """Distance to lift the hip-origin root so the lowest link frame sits at z=0.

    Forward kinematics with every joint at zero (the standing pose), composing
    only joint origin transforms.  The rig root `body` is the REP-155 hip frame,
    so the legs reach below it, this is how far down the lowest link extends.
    """
    robot = urdf_parser.URDF.from_xml_string(urdf_xml)
    root = robot.get_root()
    world: dict[str, np.ndarray] = {root: np.eye(4)}
    min_z = 0.0
    stack = [root]
    while stack:
        parent = stack.pop()
        for joint_name, child in robot.child_map.get(parent, ()):
            world[child] = world[parent] @ _origin_matrix(robot.joint_map[joint_name].origin)
            min_z = min(min_z, world[child][2, 3])
            stack.append(child)
    return -min_z


@dataclasses.dataclass
class _Slot:
    proc: subprocess.Popen[bytes]
    body_id: str | None = None
    params_file: str | None = None


class BodyPool:
    """Fixed-size pool of robot_state_publisher child processes.

    Each slot is assigned to one body_id at a time.  Assignment restarts the
    rsp with a freshly generated URDF.  Clearance sends SIGTERM.  Callers must
    call `teardown()` on shutdown.
    """

    def __init__(
        self,
        node: rclpy.node.Node,
        humans_ns: str,
        max_bodies: int = 32,
    ) -> None:
        self._node = node
        self._humans_ns = humans_ns.rstrip("/")
        self._max_bodies = max_bodies
        self._slots: list[_Slot] = []
        self._id_to_slot: dict[str, _Slot] = {}
        self._urdf: dict[str, str] = {}
        self._foot_offset: dict[str, float] = {}
        self._hd_share: str = get_package_share_directory("human_description")
        self._rsp_bin: str = f"{get_package_prefix('robot_state_publisher')}/lib/robot_state_publisher/robot_state_publisher"

    def _generate_urdf(self, body_id: str, height: float) -> str:
        xacro_path = f"{self._hd_share}/urdf/human-tpl.xacro"
        doc = xacro.process_file(xacro_path, mappings={"id": body_id, "height": str(height)})
        return doc.toprettyxml(indent="  ")

    @staticmethod
    def _write_params(body_id: str, urdf: str) -> str:
        params = {"/**": {"ros__parameters": {"robot_description": urdf, "use_sim_time": True}}}
        fd, path = tempfile.mkstemp(prefix=f"rsp_{body_id}_", suffix=".yaml")
        with os.fdopen(fd, "w") as handle:
            # width disables line-wrapping: rcl's YAML param parser cannot handle the
            # backslash continuations PyYAML emits for a long robot_description scalar.
            yaml.safe_dump(params, handle, width=10**9)
        return path

    @staticmethod
    def _cleanup_params(path: str | None) -> None:
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass

    def _spawn_rsp(self, body_id: str, urdf: str) -> tuple[subprocess.Popen[bytes], str]:
        joint_states_topic = f"{self._humans_ns}/bodies/{body_id}/joint_states"
        params_file = self._write_params(body_id, urdf)
        cmd = [
            self._rsp_bin,
            "--ros-args",
            "--log-level",
            "warn",
            "--params-file",
            params_file,
            "--remap",
            f"__node:=rsp_{body_id}",
            "--remap",
            f"joint_states:={joint_states_topic}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL)
        _LOG.debug("spawned rsp pid=%d for body %s", proc.pid, body_id)
        return proc, params_file

    def _wait_terminate(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.returncode is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _terminate_slot(self, slot: _Slot) -> None:
        self._wait_terminate(slot.proc)
        self._cleanup_params(slot.params_file)
        _LOG.debug("terminated rsp pid=%d (was body %s)", slot.proc.pid, slot.body_id)
        slot.body_id = None
        slot.params_file = None

    def assign(self, body_id: str, height: float = _DEFAULT_HEIGHT) -> bool:
        """Assign body_id to a pool slot, starting its rsp.

        Returns True if assigned, False if the pool is exhausted.
        """
        if body_id in self._id_to_slot:
            return True

        urdf = self._generate_urdf(body_id, height)
        self._urdf[body_id] = urdf
        self._foot_offset[body_id] = _hip_to_foot_offset(urdf)

        free_slot: _Slot | None = None
        for slot in self._slots:
            if slot.body_id is None:
                free_slot = slot
                break

        if free_slot is not None:
            self._wait_terminate(free_slot.proc)
            self._cleanup_params(free_slot.params_file)
            free_slot.proc, free_slot.params_file = self._spawn_rsp(body_id, urdf)
            free_slot.body_id = body_id
            self._id_to_slot[body_id] = free_slot
            return True

        if len(self._slots) >= self._max_bodies:
            _LOG.warning(
                "body pool exhausted (max_bodies=%d), dropping body %s",
                self._max_bodies,
                body_id,
            )
            return False

        proc, params_file = self._spawn_rsp(body_id, urdf)
        slot = _Slot(proc=proc, body_id=body_id, params_file=params_file)
        self._slots.append(slot)
        self._id_to_slot[body_id] = slot
        return True

    def release(self, body_id: str) -> None:
        """Release a body_id, terminating its rsp slot."""
        slot = self._id_to_slot.pop(body_id, None)
        self._urdf.pop(body_id, None)
        self._foot_offset.pop(body_id, None)
        if slot is None:
            return
        self._terminate_slot(slot)

    def urdf_for(self, body_id: str) -> str | None:
        return self._urdf.get(body_id)

    def foot_offset(self, body_id: str) -> float:
        """Hip-to-floor lift for body_id, 0.0 if not assigned."""
        return self._foot_offset.get(body_id, 0.0)

    def active_ids(self) -> frozenset[str]:
        return frozenset(self._id_to_slot.keys())

    def sync(self, current_ids: set[str], heights: dict[str, float]) -> None:
        """Reconcile pool state against the current set of body ids.

        Assigns new ids and releases ids no longer present.
        """
        existing = self.active_ids()
        for bid in current_ids - existing:
            self.assign(bid, heights.get(bid, _DEFAULT_HEIGHT))
        for bid in existing - current_ids:
            self.release(bid)

    def teardown(self) -> None:
        """Terminate all active rsp processes.  Call on node shutdown."""
        for slot in self._slots:
            if slot.proc.returncode is None:
                slot.proc.terminate()
        for slot in self._slots:
            try:
                slot.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                slot.proc.kill()
                slot.proc.wait()
        for slot in self._slots:
            self._cleanup_params(slot.params_file)
        self._slots.clear()
        self._id_to_slot.clear()
        self._urdf.clear()
        self._foot_offset.clear()
