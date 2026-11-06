"""Pure conversion of visualization_msgs/Marker lists into rerun archetypes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import rerun as rr
from visualization_msgs.msg import Marker

Color = tuple[int, int, int, int]
MIN_RADIUS = 0.005


def _marker_color(m: Marker) -> Color:
    c = m.color
    return (int(c.r * 255), int(c.g * 255), int(c.b * 255), int(c.a * 255))


def _point_colors(m: Marker, n: int) -> list[Color]:
    if len(m.colors) == n:
        return [(int(c.r * 255), int(c.g * 255), int(c.b * 255), int(c.a * 255)) for c in m.colors]
    return [_marker_color(m)] * n


def _rotation(m: Marker) -> np.ndarray:
    q = m.pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def posed(m: Marker, points: Iterable) -> np.ndarray:
    """`points` of a list marker from its pose into its frame."""
    local = np.array([[p.x, p.y, p.z] for p in points], dtype=float).reshape(-1, 3)
    p = m.pose.position
    return local @ _rotation(m).T + [p.x, p.y, p.z]


def _center(m: Marker) -> list[float]:
    p = m.pose.position
    return [p.x, p.y, p.z]


def _quat(m: Marker) -> rr.Quaternion:
    q = m.pose.orientation
    return rr.Quaternion(xyzw=[q.x, q.y, q.z, q.w] if (q.x, q.y, q.z, q.w) != (0.0, 0.0, 0.0, 0.0) else [0.0, 0.0, 0.0, 1.0])


@dataclass
class Boxes:
    centers: list[list[float]] = field(default_factory=list)
    sizes: list[list[float]] = field(default_factory=list)
    quaternions: list[rr.Quaternion] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)


@dataclass
class Spheres:
    centers: list[list[float]] = field(default_factory=list)
    half_sizes: list[list[float]] = field(default_factory=list)
    quaternions: list[rr.Quaternion] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)


@dataclass
class Cylinders:
    centers: list[list[float]] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)
    quaternions: list[rr.Quaternion] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)


@dataclass
class Arrows:
    origins: list[list[float]] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)


@dataclass
class Strips:
    strips: list[list[list[float]]] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)


@dataclass
class Points:
    positions: list[list[float]] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)
    radii: list[float] = field(default_factory=list)


@dataclass
class Triangles:
    vertex_positions: list[list[float]] = field(default_factory=list)
    vertex_colors: list[Color] = field(default_factory=list)


@dataclass
class Texts:
    positions: list[list[float]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    colors: list[Color] = field(default_factory=list)


@dataclass
class MarkerGeometry:
    """What the markers of one namespace and frame draw, by archetype."""

    boxes: Boxes = field(default_factory=Boxes)
    spheres: Spheres = field(default_factory=Spheres)
    cylinders: Cylinders = field(default_factory=Cylinders)
    arrows: Arrows = field(default_factory=Arrows)
    strips: Strips = field(default_factory=Strips)
    points: Points = field(default_factory=Points)
    triangles: Triangles = field(default_factory=Triangles)
    texts: Texts = field(default_factory=Texts)

    def add(self, m: Marker) -> None:
        t = m.type
        if t == Marker.CUBE:
            self.boxes.centers.append(_center(m))
            self.boxes.sizes.append([m.scale.x, m.scale.y, m.scale.z])
            self.boxes.quaternions.append(_quat(m))
            self.boxes.colors.append(_marker_color(m))
        elif t == Marker.SPHERE:
            self.spheres.centers.append(_center(m))
            self.spheres.half_sizes.append([m.scale.x / 2, m.scale.y / 2, m.scale.z / 2])
            self.spheres.quaternions.append(_quat(m))
            self.spheres.colors.append(_marker_color(m))
        elif t == Marker.CYLINDER:
            self.cylinders.centers.append(_center(m))
            self.cylinders.lengths.append(m.scale.z)
            self.cylinders.radii.append(max(m.scale.x, m.scale.y) / 2)
            self.cylinders.quaternions.append(_quat(m))
            self.cylinders.colors.append(_marker_color(m))
        elif t == Marker.ARROW:
            if len(m.points) >= 2:
                a, b = posed(m, list(m.points)[:2])
                self.arrows.origins.append(a.tolist())
                self.arrows.vectors.append((b - a).tolist())
                self.arrows.radii.append(max(m.scale.x / 2, MIN_RADIUS))
            else:
                self.arrows.origins.append(_center(m))
                self.arrows.vectors.append((_rotation(m) @ [m.scale.x, 0.0, 0.0]).tolist())
                self.arrows.radii.append(max(m.scale.y / 2, MIN_RADIUS))
            self.arrows.colors.append(_marker_color(m))
        elif t in (Marker.LINE_STRIP, Marker.LINE_LIST):
            pts = posed(m, m.points)
            segments = [pts] if t == Marker.LINE_STRIP else [pts[i : i + 2] for i in range(0, len(pts) - 1, 2)]
            for seg in segments:
                if len(seg) >= 2:
                    self.strips.strips.append(seg.tolist())
                    self.strips.colors.append(_marker_color(m))
                    self.strips.radii.append(max(m.scale.x / 2, MIN_RADIUS))
        elif t in (Marker.POINTS, Marker.SPHERE_LIST):
            pts = posed(m, m.points)
            self.points.positions.extend(pts.tolist())
            self.points.colors.extend(_point_colors(m, len(pts)))
            self.points.radii.extend([max(m.scale.x / 2, MIN_RADIUS)] * len(pts))
        elif t == Marker.CUBE_LIST:
            pts = posed(m, m.points)
            self.boxes.centers.extend(pts.tolist())
            self.boxes.sizes.extend([[m.scale.x, m.scale.y, m.scale.z]] * len(pts))
            self.boxes.quaternions.extend([_quat(m)] * len(pts))
            self.boxes.colors.extend(_point_colors(m, len(pts)))
        elif t == Marker.TRIANGLE_LIST and len(m.points) >= 3:
            pts = posed(m, list(m.points)[: len(m.points) // 3 * 3])
            self.triangles.vertex_positions.extend(pts.tolist())
            self.triangles.vertex_colors.extend(_point_colors(m, len(m.points))[: len(pts)])
        elif t == Marker.TEXT_VIEW_FACING and m.text:
            self.texts.positions.append(_center(m))
            self.texts.labels.append(m.text)
            self.texts.colors.append(_marker_color(m))

    def archetypes(self) -> dict[str, rr.AsComponents]:
        """One archetype per non-empty kind, keyed by the entity part it is logged under."""
        out: dict[str, rr.AsComponents] = {}
        if self.boxes.centers:
            b = self.boxes
            out["boxes"] = rr.Boxes3D(fill_mode="solid", centers=b.centers, sizes=b.sizes, quaternions=b.quaternions, colors=b.colors)
        if self.spheres.centers:
            s = self.spheres
            out["spheres"] = rr.Ellipsoids3D(fill_mode="solid", centers=s.centers, half_sizes=s.half_sizes, quaternions=s.quaternions, colors=s.colors)
        if self.cylinders.centers:
            c = self.cylinders
            out["cylinders"] = rr.Cylinders3D(fill_mode="solid", centers=c.centers, lengths=c.lengths, radii=c.radii, quaternions=c.quaternions, colors=c.colors)
        if self.arrows.origins:
            out["arrows"] = rr.Arrows3D(origins=self.arrows.origins, vectors=self.arrows.vectors, colors=self.arrows.colors, radii=self.arrows.radii)
        if self.strips.strips:
            out["lines"] = rr.LineStrips3D(self.strips.strips, colors=self.strips.colors, radii=self.strips.radii)
        if self.points.positions:
            out["points"] = rr.Points3D(self.points.positions, colors=self.points.colors, radii=self.points.radii)
        if self.triangles.vertex_positions:
            n = len(self.triangles.vertex_positions)
            out["triangles"] = rr.Mesh3D(vertex_positions=self.triangles.vertex_positions, vertex_colors=self.triangles.vertex_colors, triangle_indices=np.arange(n).reshape(-1, 3))
        if self.texts.positions:
            out["text"] = rr.Points3D(self.texts.positions, radii=0.01, colors=self.texts.colors, labels=self.texts.labels, show_labels=True)
        return out


def convert(markers: Iterable[Marker]) -> MarkerGeometry:
    geometry = MarkerGeometry()
    for m in markers:
        geometry.add(m)
    return geometry
