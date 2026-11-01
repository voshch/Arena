"""Reference: the per-frame scipy retarget (rotation init + least-squares limb fit), kept to check the batched
pipeline against. Not used by the package."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from arena_simulation_setup.adapters.arena_animation_retarget.contract import SK, WIRE
from arena_simulation_setup.adapters.arena_animation_retarget.retarget import ARM_REST, heading, rest_maps
from arena_simulation_setup.adapters.arena_animation_retarget.smplx import J, Motion


def _euler(m: np.ndarray, seq: str) -> np.ndarray:
    return Rotation.from_matrix(m).as_euler(seq)  # upper case = intrinsic, matches the contract's right-multiplied chains


def _hinge(m: np.ndarray) -> float:
    """Angle of the contract hinge rot_axis((0,-1,0), a) = Ry(-a) closest to m."""
    return float(-np.arctan2(m[0, 2], m[2, 2]))


def _wrap_near(v: float, ref: float) -> float:
    return ref + float(np.angle(np.exp(1j * (v - ref))))


def _arm(m: np.ndarray, prev: tuple[float, float, float]) -> tuple[float, float, float]:
    """Inverse of SK.arm_rot = Rz(y) R(-Y, p) R(-Z, r), on the branch nearest the previous frame (no flips)."""
    d = m @ SK.DOWN
    p = float(np.arccos(np.clip(-d[2], -1, 1)))
    y = float(np.arctan2(d[1], d[0])) if np.sin(p) > 1e-4 else prev[0]  # arm straight down/up: hold the azimuth
    best = None
    for yy, pp in ((y, p), (y + np.pi, -p)):  # spherical double cover
        n = (SK.rot_axis(SK.ARM_AXES[0], yy) @ SK.rot_axis(SK.ARM_AXES[1], pp)).T @ m
        rr = float(np.arctan2(-n[1, 0], n[0, 0]))
        cand = (_wrap_near(yy, prev[0]), pp, _wrap_near(rr, prev[2]))
        cost = sum(abs(c - q) for c, q in zip(cand, prev))
        if best is None or cost < best[0]:
            best = (cost, cand)
    return best[1]


def _fit_limb(frames_of, x0: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Refine a 3-DOF + hinge limb so BOTH bone directions match. The rotation-retarget solution x0 is the start and a
    weak anchor, and the target's forward axis holds the twist. Keep that anchor: SMPL-X knees and elbows are ball
    joints, and forcing both directions exact through a hinge makes the twist of a near-straight limb ill-conditioned
    (the hip twist swings ~70 deg to fit a few degrees of knee ab-duction). The anchor costs ~1 deg of direction."""
    d1, d2, fwd = upper @ SK.DOWN, lower @ SK.DOWN, upper[:, 0]

    def res(x):
        u, f = frames_of(x)
        return np.concatenate([u @ SK.DOWN - d1, f @ SK.DOWN - d2, 0.1 * (u[:, 0] - fwd), 0.01 * (x - x0)])

    return least_squares(res, x0, method="lm").x


def retarget_frames(motion: Motion, fold_tilt: bool = True) -> dict:
    """Retarget every source frame (no resampling). Returns wire angles plus the root tracks, per source frame."""
    G, P, T = motion.global_rots, motion.joints, len(motion)
    Q = rest_maps(motion)
    out = {k: np.zeros(T) for k in WIRE}
    root, root_z, root_q = np.zeros((T, 3)), np.zeros(T), np.zeros((T, 4))
    prev_arm = {"l": (0.0, 0.0, 0.0), "r": (0.0, 0.0, 0.0)}
    for t in range(T):
        g = G[t]
        yaw = heading(g[J["pelvis"]])
        root_r = SK.rot_axis(SK.UP, yaw)
        root[t] = (P[t, 0, 0], P[t, 0, 1], yaw)
        root_z[t] = P[t, 0, 2] - P[t, :, 2].min()  # pelvis above the lowest joint (the floor)
        root_q[t] = Rotation.from_matrix(root_r.T @ g[J["pelvis"]]).as_quat()
        base = root_r if fold_tilt else g[J["pelvis"]]  # fold: the waist and hips absorb the pelvis tilt

        # spine: three segments, each takes its SMPL-X counterpart's frame; segment_rot = Rx(r) Rz(y) Ry(p)
        cur = base
        for seg, j in (("waist", "pelvis"), ("spine", "spine1"), ("chest", "spine2")):
            r, y, p = _euler(cur.T @ g[J[j]], "XZY")
            out[f"r_{seg}"][t], out[f"y_{seg}"][t], out[seg][t] = r, y, p
            cur = cur @ SK.segment_rot(r, y, p)
        chest = cur
        r, y, q = _euler(chest.T @ g[J["head"]], "XZY")  # head_rot = Rx(r) Rz(y) R(-Y, p)
        out["r_head"][t], out["y_head"][t], out["p_head"][t] = r, y, -q

        for s in "lr":
            sign = 1.0 if s == "l" else -1.0
            # collar: where the rest clavicle (0, +-1, 0) swings to, solved for protraction y and elevation p
            d = (chest.T @ g[J[f"{s}_collar"]]) @ np.array([0.0, sign, 0.0])
            cp = float(np.arcsin(np.clip(d[2], -1, 1)))
            cy = float(np.arctan2(d[0], sign * d[1]))
            out[f"{s}_y_collar"][t], out[f"{s}_p_collar"][t] = cy, cp
            c_r = chest @ SK.collar_rot(s, cy, cp)

            # arm: rotation retarget as the start, then fit both bone directions
            U = g[J[f"{s}_shoulder"]] @ Q[f"{s}_shoulder"]
            Fo = g[J[f"{s}_elbow"]] @ Q[f"{s}_elbow"]
            ay, ap, ar = _arm(c_r.T @ U, prev_arm[s])
            e = _hinge((c_r @ SK.arm_rot(ay, ap, ar)).T @ Fo)

            def arm_frames(x, c_r=c_r):
                u = c_r @ SK.arm_rot(*x[:3])
                return u, u @ SK.rot_axis(SK.ELBOW_AXIS, x[3])

            ay, ap, ar, e = _fit_limb(arm_frames, np.array([ay, ap, ar, e]), U, Fo)
            prev_arm[s] = (ay, ap, ar)
            out[f"{s}_y_shoulder"][t], out[f"{s}_p_shoulder"][t], out[f"{s}_r_shoulder"][t] = ay, ap, ar
            out[f"{s}_elbow"][t] = e
            f_r = c_r @ SK.arm_rot(ay, ap, ar) @ SK.rot_axis(SK.ELBOW_AXIS, e)
            # wrist: pronation (+-Z) then extension (+-X), relative to the forearm without its alignment offset
            a1, b1, _ = _euler((f_r @ Q[f"{s}_elbow"].T @ ARM_REST[s]).T @ (g[J[f"{s}_wrist"]] @ ARM_REST[s]), "ZXY")
            out[f"{s}_r_wrist"][t], out[f"{s}_wrist"][t] = (-a1, b1) if s == "l" else (a1, -b1)

            # leg: hip triple Rz(-y) Rx(+-p) Ry(-r) + knee hinge, fit like the arm
            Th, Sh = g[J[f"{s}_hip"]] @ Q[f"{s}_hip"], g[J[f"{s}_knee"]] @ Q[f"{s}_knee"]
            hy, hp, hr = _euler(base.T @ Th, "ZXY")
            x1, x2, x3 = SK.HIP_AXES[s]

            def leg_frames(x, base=base, x1=x1, x2=x2, x3=x3):
                h = base @ SK.rot_axis(x1, x[0]) @ SK.rot_axis(x2, x[1]) @ SK.rot_axis(x3, x[2])
                return h, h @ SK.rot_axis(SK.ELBOW_AXIS, x[3])

            x0 = np.array([-hy, sign * hp, -hr, 0.0])
            x0[3] = _hinge(leg_frames(x0)[0].T @ Sh)
            yh, ph, rh, k = _fit_limb(leg_frames, x0, Th, Sh)
            if t:  # each angle on the branch nearest the previous frame
                yh, ph, rh = (_wrap_near(v, out[n][t - 1]) for n, v in ((f"{s}_y_hip", yh), (f"{s}_p_hip", ph), (f"{s}_r_hip", rh)))
            out[f"{s}_y_hip"][t], out[f"{s}_p_hip"][t], out[f"{s}_r_hip"][t] = yh, ph, rh
            out[f"{s}_knee"][t] = k
            k_r = leg_frames(np.array([yh, ph, rh, k]))[1]
            fy, fb, _ = _euler((k_r @ Q[f"{s}_knee"].T).T @ g[J[f"{s}_ankle"]], "ZYX")  # Rz(-y_ankle) Ry(-ankle)
            out[f"{s}_y_ankle"][t], out[f"{s}_ankle"][t] = -fy, -fb

    for k in out:  # never a 2 pi jump between frames
        out[k] = np.unwrap(out[k])
    root[:, 2] = np.unwrap(root[:, 2])
    return {"angles": out, "root_xy_yaw": root, "root_z": root_z, "root_quat": root_q}
