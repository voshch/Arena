"""Pure-math helpers for camera curves: vectors, quaternions, interpolation, easing.

Quaternions are (w, x, y, z). The look-at convention matches the gz camera
(forward +X, up +Z) and the ViewportCamera plugin: zero roll, yaw = atan2(dy, dx),
pitch = -asin(dz / |dir|).
"""

from __future__ import annotations

import math

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vscale(a: Vec3, s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vlen(a: Vec3) -> float:
    return math.sqrt(vdot(a, a))


def vnorm(a: Vec3) -> Vec3:
    n = vlen(a)
    return a if n < 1e-9 else vscale(a, 1.0 / n)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def vlerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def quat_from_euler(roll: float, pitch: float, yaw: float) -> Quat:
    """ZYX (gz convention): R = Rz(yaw) Ry(pitch) Rx(roll). Returns (w, x, y, z)."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def look_at_quat(eye: Vec3, target: Vec3) -> Quat:
    """Orient a gz camera (forward +X, up +Z) from eye toward target, zero roll."""
    d = vsub(target, eye)
    if vlen(d) < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    d = vnorm(d)
    yaw = math.atan2(d[1], d[0])
    pitch = -math.asin(max(-1.0, min(1.0, d[2])))
    return quat_from_euler(0.0, pitch, yaw)


def quat_forward(q: Quat) -> Vec3:
    """The camera view direction: the +X axis rotated by q."""
    w, x, y, z = q
    return (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
    )


def slerp(q0: Quat, q1: Quat, t: float) -> Quat:
    d = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]
    if d < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        d = -d
    if d > 0.9995:
        r = (
            lerp(q0[0], q1[0], t),
            lerp(q0[1], q1[1], t),
            lerp(q0[2], q1[2], t),
            lerp(q0[3], q1[3], t),
        )
        n = math.sqrt(sum(c * c for c in r))
        return (r[0] / n, r[1] / n, r[2] / n, r[3] / n) if n > 1e-9 else q0
    theta0 = math.acos(max(-1.0, min(1.0, d)))
    theta = theta0 * t
    s0 = math.sin(theta0 - theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    return (
        s0 * q0[0] + s1 * q1[0],
        s0 * q0[1] + s1 * q1[1],
        s0 * q0[2] + s1 * q1[2],
        s0 * q0[3] + s1 * q1[3],
    )


def catmull_rom(points: list[Vec3], t: float) -> Vec3:
    """Uniform Catmull-Rom through points, t in [0, 1] over the whole chain.
    Endpoints are duplicated as phantom controls so the spline hits them."""
    n = len(points)
    if n == 1:
        return points[0]
    if n == 2:
        return vlerp(points[0], points[1], t)
    t = max(0.0, min(1.0, t))
    segs = n - 1
    u = t * segs
    i = min(int(u), segs - 1)
    f = u - i
    p0 = points[i - 1] if i > 0 else points[0]
    p1 = points[i]
    p2 = points[i + 1]
    p3 = points[i + 2] if i + 2 < n else points[n - 1]
    f2 = f * f
    f3 = f2 * f

    def comp(a: float, b: float, c: float, d: float) -> float:
        return 0.5 * ((2 * b) + (-a + c) * f + (2 * a - 5 * b + 4 * c - d) * f2 + (-a + 3 * b - 3 * c + d) * f3)

    return (
        comp(p0[0], p1[0], p2[0], p3[0]),
        comp(p0[1], p1[1], p2[1], p3[1]),
        comp(p0[2], p1[2], p2[2], p3[2]),
    )


def _ease_inout(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 2 / 2


EASES = {
    "linear": lambda t: t,
    "in": lambda t: t * t,
    "out": lambda t: 1 - (1 - t) * (1 - t),
    "inout": _ease_inout,
    "sine_inout": lambda t: -(math.cos(math.pi * t) - 1) / 2,
}


def ease(name: str, t: float) -> float:
    t = max(0.0, min(1.0, t))
    return EASES.get(name, EASES["linear"])(t)
