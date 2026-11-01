"""Batched, differentiable contract FK (torch): ``skeleton.fk`` for N poses at once, root at the origin, yaw 0.

Axes, chain order and bone lengths are read from the contract module (``SK``), the math mirrors ``SK.fk`` line by
line, and ``tests/test_fk_torch.py`` holds the two to 1e-5.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .contract import SK, WIRE

IDX = {k: i for i, k in enumerate(WIRE)}


class Axis:
    """Rotations about one fixed axis, batched over angles: R = I + sin(a) K + (1 - cos(a)) K^2."""

    def __init__(self, axis: Sequence[float] | np.ndarray, device: torch.device | str, dtype: torch.dtype) -> None:
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        self.K = torch.as_tensor(k, device=device, dtype=dtype)
        self.K2 = self.K @ self.K
        self.I = torch.eye(3, device=device, dtype=dtype)

    def __call__(self, angle: torch.Tensor) -> torch.Tensor:
        s, c = torch.sin(angle)[..., None, None], torch.cos(angle)[..., None, None]
        return self.I + s * self.K + (1 - c) * self.K2


class ContractFK:
    """``fk(theta)`` with theta (N, 40) in ``WIRE`` order -> (positions, frames), dicts of (N, 3) / (N, 3, 3)."""

    def __init__(self, height: float = 1.75, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.body = b = SK.Body(height)
        self.device, self.dtype = device, dtype
        t = lambda v: torch.as_tensor(np.asarray(v, dtype=float), device=device, dtype=dtype)  # noqa: E731
        ax = lambda v: Axis(v, device, dtype)  # noqa: E731
        self.rx, self.rz, self.ry = ax(SK.TORSO_AXIS["r"]), ax(SK.TORSO_AXIS["y"]), ax(SK.TORSO_AXIS["p"])
        self.head_p = ax([0, -1.0, 0])
        self.collar = {s: tuple(ax(a) for a in SK.COLLAR_AXES[s]) for s in "lr"}
        self.arm = tuple(ax(a) for a in SK.ARM_AXES)
        self.hip = {s: tuple(ax(a) for a in SK.HIP_AXES[s]) for s in "lr"}
        self.hinge = ax(SK.ELBOW_AXIS)
        self.wrist = {s: tuple(ax(a) for a in SK.WRIST_AXES[s]) for s in "lr"}
        self.ankle_yaw = ax([0, 0, -1.0])
        self.UP, self.DOWN, self.FWD = t(SK.UP), t(SK.DOWN), t(SK.FWD)
        self.pelvis = t([0.0, 0.0, b.hip_height])
        self.head_off = t([b.head_radius / 2, 0.0, b.head_radius])
        self.shoulder_off = {s: t([0.0, SK.REFLECT[s] * b.neck_shoulder, 0.0]) for s in "lr"}
        self.hip_off = {s: t([0.0, SK.REFLECT[s] * b.waist_length / 2, 0.0]) for s in "lr"}
        self.lengths = {"arm": b.upperarm + b.forearm, "leg": b.thigh + b.tibia, "clavicle": b.neck_shoulder, "height": b.height}

    @staticmethod
    def _v(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return m @ v  # (N, 3, 3) @ (3,) -> (N, 3)

    def __call__(self, theta: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        a = lambda k: theta[:, IDX[k]]  # noqa: E731
        b, n = self.body, theta.shape[0]
        pos, fr = {}, {}
        pelvis = self.pelvis.expand(n, 3)
        pos["pelvis"] = pelvis
        r = torch.eye(3, device=theta.device, dtype=theta.dtype).expand(n, 3, 3)
        cursor = pelvis
        for seg in SK.SPINE:
            r = r @ self.rx(a(f"r_{seg}")) @ self.rz(a(f"y_{seg}")) @ self.ry(a(seg))
            fr[seg], pos[seg] = r, cursor
            cursor = cursor + self._v(r, self.UP) * b.spine_segment
        torso, torso_r = cursor, r
        pos["torso"], fr["torso"] = torso, torso_r
        neck = torso + self._v(torso_r, self.UP) * b.neck_length
        pos["neck"] = neck
        h_r = torso_r @ self.rx(a("r_head")) @ self.rz(a("y_head")) @ self.head_p(a("p_head"))
        fr["head"], pos["head"] = h_r, neck + self._v(h_r, self.head_off)
        for s in "lr":
            cy, cp = self.collar[s]
            c_r = torso_r @ cy(a(f"{s}_y_collar")) @ cp(a(f"{s}_p_collar"))
            fr[f"{s}_collar"] = c_r
            sh = torso + self._v(c_r, self.shoulder_off[s])
            a1, a2, a3 = self.arm
            s_r = c_r @ a1(a(f"{s}_y_shoulder")) @ a2(a(f"{s}_p_shoulder")) @ a3(a(f"{s}_r_shoulder"))
            el = sh + self._v(s_r, self.DOWN) * b.upperarm
            f_r = s_r @ self.hinge(a(f"{s}_elbow"))
            pr, ex = self.wrist[s]
            pos[f"{s}_shoulder"], pos[f"{s}_elbow"] = sh, el
            pos[f"{s}_wrist"] = el + self._v(f_r, self.DOWN) * b.forearm
            fr[f"{s}_upperarm"], fr[f"{s}_forearm"] = s_r, f_r
            fr[f"{s}_hand"] = f_r @ pr(a(f"{s}_r_wrist")) @ ex(a(f"{s}_wrist"))
        for s in "lr":
            hip = pelvis + self.hip_off[s]
            x1, x2, x3 = self.hip[s]
            h = x1(a(f"{s}_y_hip")) @ x2(a(f"{s}_p_hip")) @ x3(a(f"{s}_r_hip"))
            knee = hip + self._v(h, self.DOWN) * b.thigh
            k = h @ self.hinge(a(f"{s}_knee"))
            ankle = knee + self._v(k, self.DOWN) * b.tibia
            f = k @ self.ankle_yaw(a(f"{s}_y_ankle")) @ self.hinge(a(f"{s}_ankle"))
            pos[f"{s}_hip"], pos[f"{s}_knee"], pos[f"{s}_ankle"] = hip, knee, ankle
            pos[f"{s}_foot"] = ankle + self._v(f, self.FWD) * 0.15
            fr[f"{s}_thigh"], fr[f"{s}_shin"], fr[f"{s}_foot"] = h, k, f
        return pos, fr
