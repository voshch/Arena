"""Baked PointAt: ``bake`` solves an (azimuth, elevation) hold grid offline, ``BakedPointAt`` plays it back without IK.

Rebake: ``python3 -m task_generator.simulators.human.pointing.table`` (output lands next to the templates).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import contract as C
from . import skeleton as S
from .generator import FIXED_POINT_DAMPING, TEMPLATE_DIR, HoldPose, PointAtClip, PointAtGenerator, PointAtOptions, _triple_in_limits, angles_from_direction

SIDES = ("l", "r")
AZ_STEP = 5.0
EL_STEP = 5.0
EL_MAX = 85.0  # the poles are degenerate for the (az, el) grid, clamp just short of them
COLLAR_SCALE = 0.3  # fraction of the template's clavicle shrug kept in the bake
SEAM_RAD = math.radians(30.0)  # corner holds further apart than this are not blended, the nearest wins
FIXED_POINT_ITERS = 12  # target-point aim refinement through the baked wrist


def table_path(template: str) -> str:
    return os.path.join(TEMPLATE_DIR, f"{template}.table.npz")


def _quat(m: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1e-12, 1.0 + m[i, i] - m[j, j] - m[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (m[k, j] - m[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (m[j, i] + m[i, j]) / s
    q[1 + k] = (m[k, i] + m[i, k]) / s
    return q


def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass(frozen=True)
class Cell:
    """Interpolated hold for one aim: blended DOFs, collar-frame shoulder rotation, bake verdict."""

    angles: dict[str, float]
    r_shoulder: np.ndarray
    swivel: float
    wrist: np.ndarray
    ok: bool
    clearance: float
    aim_error: float
    relaxed: bool


class PointTable:
    """The baked grid: ``cell(side, aim)`` bilinearly interpolates the four surrounding holds."""

    def __init__(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as z:
            self.path = path
            self.template = str(z["template"])
            self.height = float(z["height"])
            self.collar_scale = float(z["collar_scale"])
            self.az = z["az"].astype(float)
            self.el = z["el"].astype(float)
            self.dofs = [str(n) for n in z["dofs"]]
            self.linear = z["linear"].astype(float)
            self.quat = z["quat"].astype(float)
            self.swivel = z["swivel"].astype(float)
            self.wrist = z["wrist"].astype(float)
            self.ok = z["ok"].astype(bool)
            self.clearance = z["clearance"].astype(float)
            self.aim_error = z["aim_error"].astype(float)
            self.relaxed = z["relaxed"].astype(bool)
        self.az_step = float(self.az[1] - self.az[0])
        self.el_step = float(self.el[1] - self.el[0])

    def _corners(self, aim: np.ndarray) -> list[tuple[int, int, float]]:
        """(az index, el index, weight) of the four cells around ``aim``, azimuth wrapping."""
        az, el = angles_from_direction(aim)
        el = min(max(el, float(self.el[0])), float(self.el[-1]))
        fa = (az - float(self.az[0])) / self.az_step
        fe = (el - float(self.el[0])) / self.el_step
        ia, ie = int(math.floor(fa)), int(math.floor(fe))
        wa, we = fa - ia, fe - ie
        ie = min(ie, len(self.el) - 2)
        we = min(fe - ie, 1.0)
        n_az = len(self.az) - 1  # first and last azimuth are the same direction
        out = []
        for da, xa in ((0, 1.0 - wa), (1, wa)):
            for de, xe in ((0, 1.0 - we), (1, we)):
                w = xa * xe
                if w > 1e-9:
                    out.append(((ia + da) % n_az, ie + de, w))
        return out

    def cell(self, side: str, aim: np.ndarray) -> Cell:
        s = SIDES.index(side)
        corners = self._corners(aim)
        quats = [self.quat[s, ia, ie] for ia, ie, _ in corners]
        q0 = quats[0]
        aligned = [q if float(q @ q0) >= 0.0 else -q for q in quats]
        seam = any(2.0 * math.acos(min(1.0, abs(float(a @ b)))) > SEAM_RAD for a in aligned for b in aligned)
        if seam:
            corners = [max(corners, key=lambda c: c[2])]
            aligned = [self.quat[s, corners[0][0], corners[0][1]]]
        weights = np.array([w for _, _, w in corners])
        weights /= weights.sum()
        idx = [(ia, ie) for ia, ie, _ in corners]
        linear = sum(w * self.linear[s, ia, ie] for (ia, ie), w in zip(idx, weights, strict=True))
        q = sum(w * a for a, w in zip(aligned, weights, strict=True))
        r_shoulder = _rot(np.asarray(q))
        sw = complex(0.0)
        for (ia, ie), w in zip(idx, weights, strict=True):
            sw += w * complex(math.cos(self.swivel[s, ia, ie]), math.sin(self.swivel[s, ia, ie]))
        wrist = sum(w * self.wrist[s, ia, ie] for (ia, ie), w in zip(idx, weights, strict=True))
        angles = {n.replace("{side}", side): float(v) for n, v in zip(self.dofs, linear, strict=True)}
        for name, value in zip((f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"), S.arm_rot_inverse(r_shoulder), strict=True):
            angles[name] = float(value)
        return Cell(
            angles=angles,
            r_shoulder=r_shoulder,
            swivel=float(math.atan2(sw.imag, sw.real)),
            wrist=np.asarray(wrist, dtype=float),
            ok=all(self.ok[s, ia, ie] for ia, ie in idx),
            clearance=float(min(self.clearance[s, ia, ie] for ia, ie in idx)),
            aim_error=float(max(self.aim_error[s, ia, ie] for ia, ie in idx)),
            relaxed=any(self.relaxed[s, ia, ie] for ia, ie in idx),
        )


# ---------------------------------------------------------------------------
# bake
# ---------------------------------------------------------------------------

_BAKE_GEN: dict[tuple[str, float, float], PointAtGenerator] = {}


def anchor_index(gen: PointAtGenerator) -> int:
    """Mid-hold source frame the hold is baked on and swung about."""
    hold = np.flatnonzero(gen.template.w >= 0.999)
    return int(hold[len(hold) // 2]) if hold.size else gen.template.peak


def _linear_dofs(side: str) -> list[str]:
    tri = {f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"}
    return [f"y_{s}" for s in C.SPINE_SEGMENTS] + list(C.SPINE_SEGMENTS) + [d for d in C.arm_dofs(side) if d not in tri]


def _solve_cell(template: str, height: float, collar_scale: float, side: str, az: float, el: float) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, bool, float, float, bool]:
    key = (template, height, collar_scale)
    gen = _BAKE_GEN.get(key)
    if gen is None:
        gen = _BAKE_GEN[key] = PointAtGenerator(template=template, height=height, options=PointAtOptions(collar_scale=collar_scale))
    clip = gen.point_at(azimuth=az, elevation=el, hand=side)
    ang = clip.frames[anchor_index(gen)]["angles"]
    pos, _ = S.fk(ang, gen.body)
    tri = (ang[f"{side}_y_shoulder"], ang[f"{side}_p_shoulder"], ang[f"{side}_r_shoulder"])
    rep = clip.report
    return (
        np.array([ang[n] for n in _linear_dofs(side)], dtype=float),
        _quat(S.arm_rot(*tri)),
        float(rep["swivel_rad"]),
        np.asarray(pos[f"{side}_wrist"], dtype=float),
        bool(rep["ok"]),
        float(rep["clearance_m"]),
        float(rep["aim_error_deg"]),
        bool(rep["relaxed"]),
    )


def bake(
    template: str = "point_to_right",
    height: float = 1.65,
    *,
    collar_scale: float = COLLAR_SCALE,
    path: str | None = None,
    az_step: float = AZ_STEP,
    el_step: float = EL_STEP,
    el_max: float = EL_MAX,
    workers: int | None = None,
    progress: bool = False,
) -> str:
    """Solve every grid cell for both arms and write the table. Returns the path."""
    path = path or table_path(template)
    az = np.arange(-180.0, 180.0 + 1e-9, az_step)
    el = np.arange(-el_max, el_max + 1e-9, el_step)
    jobs = [(s, ia, ie) for s in range(len(SIDES)) for ia in range(len(az) - 1) for ie in range(len(el))]
    shape = (len(SIDES), len(az), len(el))
    n_lin = len(_linear_dofs("l"))
    linear = np.zeros((*shape, n_lin), dtype=np.float32)
    quat = np.zeros((*shape, 4), dtype=np.float32)
    swivel = np.zeros(shape, dtype=np.float32)
    wrist = np.zeros((*shape, 3), dtype=np.float32)
    ok = np.zeros(shape, dtype=bool)
    clearance = np.zeros(shape, dtype=np.float32)
    aim_error = np.zeros(shape, dtype=np.float32)
    relaxed = np.zeros(shape, dtype=bool)

    def store(job: tuple[int, int, int], res: tuple) -> None:
        s, ia, ie = job
        linear[s, ia, ie], quat[s, ia, ie], swivel[s, ia, ie], wrist[s, ia, ie], ok[s, ia, ie], clearance[s, ia, ie], aim_error[s, ia, ie], relaxed[s, ia, ie] = res

    if workers == 0:
        for k, job in enumerate(jobs):
            store(job, _solve_cell(template, height, collar_scale, SIDES[job[0]], float(az[job[1]]), float(el[job[2]])))
            if progress and k % 50 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_solve_cell, template, height, collar_scale, SIDES[s], float(az[ia]), float(el[ie])): (s, ia, ie) for s, ia, ie in jobs}
            for k, fut in enumerate(concurrent.futures.as_completed(futs)):
                store(futs[fut], fut.result())
                if progress and k % 50 == 0:
                    print(f"{k}/{len(jobs)}", flush=True)
    # az = +180 is the az = -180 column
    for arr in (linear, quat, swivel, wrist, ok, clearance, aim_error, relaxed):
        arr[:, -1] = arr[:, 0]
    np.savez_compressed(
        path,
        template=np.array(template),
        height=np.array(height),
        collar_scale=np.array(collar_scale),
        az=az.astype(np.float32),
        el=el.astype(np.float32),
        dofs=np.array([d.replace("l_", "{side}_", 1) if d.startswith("l_") else d for d in _linear_dofs("l")]),
        linear=linear,
        quat=quat,
        swivel=swivel,
        wrist=wrist,
        ok=ok,
        clearance=clearance,
        aim_error=aim_error,
        relaxed=relaxed,
    )
    return path


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------


class BakedPointAt(PointAtGenerator):
    """``PointAtGenerator`` whose clips come from the baked table instead of the solver."""

    def __init__(self, template: str = "point_to_right", height: float = 1.65, options: PointAtOptions | None = None, table: PointTable | None = None) -> None:
        table = table or PointTable(table_path(template))
        options = options or PointAtOptions(collar_scale=table.collar_scale)
        super().__init__(template=template, height=height, options=options)
        self.table = table
        if table.template != template or abs(table.height - height) > 1e-6 or abs(table.collar_scale - options.collar_scale) > 1e-9:
            raise ValueError(f"table {table.path} was baked for {table.template} at {table.height} m, collar_scale {table.collar_scale}, not {template} at {height} m, collar_scale {options.collar_scale}")

    def _cell(self, side: str, aim0: np.ndarray, target_point: np.ndarray | None) -> tuple[Cell, np.ndarray]:
        """Hold cell for an aim, fixed-point refined through the baked wrist for 3-D targets."""
        aim = aim0
        cell = self.table.cell(side, aim)
        if target_point is not None:
            for _ in range(FIXED_POINT_ITERS):
                delta = target_point - cell.wrist
                if np.linalg.norm(delta) < 1e-6:
                    break
                want = S.unit(delta)
                if float(want @ aim) > math.cos(math.radians(0.1)):
                    break
                aim = S.unit(aim + FIXED_POINT_DAMPING * (want - aim))
                cell = self.table.cell(side, aim)
        return cell, aim

    def _swing(self, side: str, cell: Cell, upright: bool) -> tuple[dict[str, float], np.ndarray]:
        """Linear DOF deltas vs the anchor frame, and the world rotation swinging the recorded arm onto the hold."""
        tri = [f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"]
        anchor = anchor_index(self)
        src = self._source_pose(anchor, side)
        delta = {n: v - src[n] for n, v in cell.angles.items() if n not in tri}
        out = self._posed(side, src, delta, 1.0)
        r_hold = self._collar(side, out, False) @ cell.r_shoulder
        r_delta = r_hold @ (self._collar(side, src, upright) @ S.arm_rot(*(src[n] for n in tri))).T
        return delta, r_delta

    def _collar(self, side: str, angles: dict, upright: bool) -> np.ndarray:
        """Collar frame of a pose, spine zeroed when the torso DOFs are not blended (upright)."""
        r = np.eye(3)
        if not upright:
            for seg in C.SPINE_SEGMENTS:
                r = r @ S.segment_rot(angles[f"r_{seg}"], angles[f"y_{seg}"], angles[seg])
        return r @ S.collar_rot(side, angles[f"{side}_y_collar"], angles[f"{side}_p_collar"])

    def _posed(self, side: str, src: dict, delta: dict[str, float], w: float) -> dict:
        out = dict(src)
        for n, dv in delta.items():
            out[n] = C.clamp(n, C.to_limits(n, float(src[n] + w * dv)))
        return out

    def _report(self, cell: Cell, aim: np.ndarray, near: bool, n_frames: int, upright: bool) -> dict:
        az, el = angles_from_direction(aim)
        return {
            "template": self.template.name,
            "baked": True,
            "upright": upright,
            "azimuth_deg": float(az),
            "elevation_deg": float(el),
            "aim_error_deg": cell.aim_error,
            "clearance_m": cell.clearance,
            "ok": cell.ok,
            "near_target_fallback": near,
            "swivel_rad": cell.swivel,
            "n_frames": n_frames,
            "duration_s": float((n_frames - 1) * self.template.dt),
        }

    def _emit(self, side: str, out: dict, r_world: np.ndarray, prev_tri: tuple[float, float, float] | None, emit: dict[str, tuple[float, float]]) -> tuple[float, float, float]:
        tri = (f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder")
        vals = _triple_in_limits(side, S.arm_rot_inverse(r_world), prev=prev_tri, limits=emit)
        for n, v in zip(tri, vals, strict=True):
            out[n] = v
        for name in self._blended_dofs(side):
            out[name] = C.clamp(name, out[name], limits=emit)
        return vals

    def point_at(
        self,
        target: Sequence[float] | np.ndarray | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        deg: bool = True,
        options: PointAtOptions | None = None,
        **kw: object,
    ) -> PointAtClip:
        opts = options or self.opts
        if kw:
            opts = PointAtOptions(**{**opts.__dict__, **kw})
        aim0, target_point = self._resolve_aim(target, azimuth, elevation, deg)
        side = self.choose_hand(aim0, opts)
        near = False
        if target_point is not None and self._near_target(target_point, side):
            near, target_point = True, None
        cell, _ = self._cell(side, aim0, target_point)
        delta, r_delta = self._swing(side, cell, opts.upright)
        tpl = self.template
        w_track, idx = self._timeline(opts)
        emit = C.UNWRAP_LIMITS if opts.unwrap_shoulder else C.EMIT_LIMITS
        tri = [f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"]
        frames: list[dict] = []
        prev_tri = None
        for k, (i, w) in enumerate(zip(idx, w_track, strict=True)):
            src = self._source_pose(i, side)
            out = self._posed(side, src, delta, float(w))
            r_world = S.slerp_rot(np.eye(3), r_delta, float(w)) @ self._collar(side, src, opts.upright) @ S.arm_rot(*(src[n] for n in tri))
            prev_tri = self._emit(side, out, self._collar(side, out, opts.upright).T @ r_world, prev_tri, emit)
            frames.append(self._frame(out, tpl.frames[i], k, opts))
        clip = PointAtClip(frames=frames, side=side, target_dir=aim0, target_point=target_point, template=tpl.name, envelope=np.asarray(w_track, dtype=float))
        clip.report = self._report(cell, aim0, near, len(frames), opts.upright)
        return clip

    def retarget(
        self,
        prev: HoldPose,
        target: Sequence[float] | np.ndarray | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
        deg: bool = True,
        options: PointAtOptions | None = None,
        transition_s: float = 0.4,
    ) -> PointAtClip:
        opts = options or self.opts
        tpl = self.template
        side = prev.side
        aim0, target_point = self._resolve_aim(target, azimuth, elevation, deg)
        if self.choose_hand(aim0, opts) != side:
            raise ValueError("hand switch")
        near = False
        if target_point is not None and self._near_target(target_point, side):
            near, target_point = True, None
        cell, _ = self._cell(side, aim0, target_point)
        delta, r_delta = self._swing(side, cell, opts.upright)
        emit = C.UNWRAP_LIMITS if opts.unwrap_shoulder else C.EMIT_LIMITS
        tri = [f"{side}_y_shoulder", f"{side}_p_shoulder", f"{side}_r_shoulder"]
        n_trans = max(1, int(round(transition_s / tpl.dt)))
        h0, h1 = tpl.settled
        n_hold = (h1 - h0) if opts.hold_s is None else max(1, int(round(opts.hold_s / tpl.dt)))
        idx = self._plateau(n_trans + n_hold)
        s_track = [0.5 - 0.5 * math.cos(math.pi * k / n_trans) for k in range(n_trans)] + [1.0] * n_hold
        prev_tri = tuple(float(prev.angles[n]) for n in tri)
        start = self._source_pose(idx[0], side)
        start.update(prev.angles)
        r_prev = self._collar(side, start, opts.upright) @ S.arm_rot(*prev_tri)
        frames: list[dict] = []
        for k, (i, s) in enumerate(zip(idx, s_track, strict=True)):
            src = self._source_pose(i, side)
            out = self._posed(side, src, delta, 1.0)
            for n in prev.angles:
                if n not in tri:
                    out[n] = (1.0 - s) * prev.angles[n] + s * out[n]
            r_world = S.slerp_rot(r_prev, r_delta @ self._collar(side, src, opts.upright) @ S.arm_rot(*(src[n] for n in tri)), float(s))
            prev_tri = self._emit(side, out, self._collar(side, out, opts.upright).T @ r_world, prev_tri, emit)
            frames.append(self._frame(out, tpl.frames[i], k, opts))
        clip = PointAtClip(frames=frames, side=side, target_dir=aim0, target_point=target_point, template=tpl.name, envelope=np.asarray(s_track, dtype=float))
        clip.report = {**self._report(cell, aim0, near, len(frames), opts.upright), "transition_frames": n_trans}
        return clip

    def _frame(self, out: dict, src: dict, k: int, opts: PointAtOptions) -> dict:
        root = (0.0, 0.0, 0.0) if opts.root == "zero" else tuple(src["root_xy_yaw"])
        return {
            "angles": {n: float(out[n]) for n in C.ROS_JOINT_ORDER},
            "root_xy_yaw": root,
            "animation_state": src.get("animation_state", 0),
            "t": float(k * self.template.dt),
        }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="bake the PointAt hold table")
    ap.add_argument("--template", default="point_to_right")
    ap.add_argument("--height", type=float, default=1.65)
    ap.add_argument("--collar-scale", type=float, default=COLLAR_SCALE)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--az-step", type=float, default=AZ_STEP)
    ap.add_argument("--el-step", type=float, default=EL_STEP)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    print(bake(a.template, a.height, collar_scale=a.collar_scale, path=a.out, az_step=a.az_step, el_step=a.el_step, workers=a.workers, progress=True))


if __name__ == "__main__":
    main()
