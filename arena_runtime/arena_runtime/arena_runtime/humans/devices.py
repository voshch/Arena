"""Assistive devices pedestrians hold, posed on the rendered skeleton and projected on the ground."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import attrs
import numpy as np
import shapely
import yaml
from arena_people_msgs.msg import HeldDevice, Pedestrian
from arena_simulation_setup.shared.body import SEMANTIC_DOFS
from geometry_msgs.msg import Point32

from arena_runtime.humans.hands import Grip, palm_frame

KINDS = (HeldDevice.CANE, HeldDevice.ROLLATOR, HeldDevice.WALKING_FRAME, HeldDevice.IV_STAND)
_SEMANTIC = frozenset(SEMANTIC_DOFS)
_NONE = np.empty((0, 2))


@attrs.frozen
class DeviceSpec:
    """An agent type's `device:` block: the prop bone its character carries, or the object model held in both hands."""

    kind: str
    bone: str = ""
    model: str = ""

    @classmethod
    def parse(cls, raw: object) -> DeviceSpec:
        if not isinstance(raw, Mapping) or raw.get("kind") not in KINDS or set(raw) - {"kind", "bone", "model"}:
            raise ValueError(f"device: {{kind: {' | '.join(KINDS)}, bone: <prop bone> | model: <object held in both hands>}}, got {raw!r}")
        if bool(raw.get("bone")) == bool(raw.get("model")):
            raise ValueError(f"device: needs exactly one of bone and model, got {raw!r}")
        return cls(str(raw["kind"]), str(raw.get("bone", "")), str(raw.get("model", "")))


def bundle_of(model_uri: str) -> Path:
    """The character bundle of an actor SDF path (`<bundle>/<name>.sdf` or `<bundle>/rig/<name>.sdf`)."""
    parent = Path(model_uri).parent
    return parent.parent if parent.name == "rig" else parent


def box_footprint(bounding_box: Sequence[Sequence[float]], face: str) -> tuple[float, float, float, float]:
    """(length along the heading, width, centre ahead of and left of the origin) of a model's box, its front on the heading."""
    from arena_assets.impl import Face

    yaw = math.remainder(math.radians(Face(face).angle + 180.0), math.tau)
    c, s = round(math.cos(yaw)), round(math.sin(yaw))
    if abs(c) + abs(s) != 1:
        raise ValueError(f"face {face!r}: a held box's front must lie along a model axis")
    (x0, x1), (y0, y1) = bounding_box[0], bounding_box[1]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    length, width = (x1 - x0, y1 - y0) if c else (y1 - y0, x1 - x0)
    return length, width, c * cx + s * cy, -s * cx + c * cy


class HeldBody:
    """A character posed as the simulators draw it, with the ground outline of the device its type holds."""

    def __init__(self, bundle: Path, spec: DeviceSpec) -> None:
        from arena_simulation_setup.shared.actor_pose import ActorPose
        from scipy.spatial import ConvexHull

        from arena_assets.utils.human import read_skin

        self.kind = spec.kind
        self._actor = ActorPose(bundle)
        self._bone = -1
        if spec.bone:
            names = [b.name for b in self._actor._bones]
            if spec.bone not in names:
                raise ValueError(f"{bundle.name} has no prop bone {spec.bone!r}")
            skin = read_skin(self._actor.skin)
            points = np.array([p for p, owner in zip(skin.points, skin.dominant, strict=True) if owner == spec.bone])
            local = (skin.inverse_binds[spec.bone] @ np.c_[points, np.ones(len(points))].T).T[:, :3]
            self._hull = local[ConvexHull(local).vertices]  # the prop mesh's extreme points in its bone frame
            self._bone = names.index(spec.bone)
        else:
            from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

            annotation = yaml.safe_load((Path(asyncio.run(ObjectIdentifier.parse(spec.model).resolve_path())) / "annotation.yaml").read_text())
            self._box = box_footprint(annotation["bounding_box"], str(annotation.get("face", "+x")))

    def footprint(self, angles: Mapping[str, float], hold: Mapping[str, Grip]) -> np.ndarray:
        """Ground outline (N, 2) of the device in the body frame (x forward, y left) at the pose's semantic angles, empty when it cannot be posed."""
        angles = {k: v for k, v in angles.items() if k in _SEMANTIC}
        if self._bone >= 0:
            R, P, _ = self._actor.bone_worlds(angles)
            points = (self._hull @ R[self._bone].T + P[self._bone])[:, :2]
            if not np.isfinite(points).all():  # GEOS aborts on NaN
                return _NONE
            hull = shapely.MultiPoint(points).convex_hull
            return np.asarray(hull.exterior.coords)[:-1] if isinstance(hull, shapely.Polygon) else _NONE
        if not hold:
            return _NONE
        positions = self._actor.positions(angles)
        mx, my = np.mean([palm_frame(positions, side).grip(grip.radius, grip.angle) for side, grip in hold.items()], axis=0)[:2]
        length, width, ahead, left = self._box
        return np.array([(mx + ahead + u * length / 2, my + left + v * width / 2) for u, v in ((1, 1), (1, -1), (-1, -1), (-1, 1))])


class HeldBodies:
    """HeldBody per (bundle, device), loaded off the publish path: a pedestrian holds no device until its body is loaded."""

    def __init__(self, warn: Callable[[str], None]) -> None:
        self._bodies: dict[tuple[str, DeviceSpec], Future[HeldBody]] = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="held_bodies")
        self._warn = warn
        self._warned: set[tuple[str, DeviceSpec]] = set()

    def get(self, model_uri: str, spec: DeviceSpec) -> HeldBody | None:
        key = (str(bundle_of(model_uri)), spec)
        future = self._bodies.get(key)
        if future is None:
            self._bodies[key] = self._pool.submit(HeldBody, Path(key[0]), spec)
            return None
        if not future.done():
            return None
        if (error := future.exception()) is not None:
            if key not in self._warned:
                self._warned.add(key)
                self._warn(f"{Path(key[0]).name}: {spec.kind} rejected ({error}), it holds none")
            return None
        return future.result()


def place(ped: Pedestrian, kind: str, outline: np.ndarray, yaw: float) -> None:
    """Write a body-frame ground outline into `ped.device` at the pedestrian's pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = ped.pose.position.x, ped.pose.position.y
    ped.device.kind = kind
    ped.device.footprint = [Point32(x=float(x + c * u - s * v), y=float(y + s * u + c * v), z=0.0) for u, v in outline]


def shift(device: HeldDevice, dx: float, dy: float) -> None:
    for point in device.footprint:
        point.x, point.y = point.x + dx, point.y + dy


def shapes(ped: Pedestrian, default_radius: float) -> tuple[shapely.Geometry | None, shapely.Geometry | None]:
    """(body disc, device outline) of a pedestrian on the ground, None for a part it lacks or whose pose is not finite."""
    x, y = ped.pose.position.x, ped.pose.position.y
    body = shapely.Point(x, y).buffer(ped.radius if ped.radius > 0.0 else default_radius) if math.isfinite(x) and math.isfinite(y) else None
    outline = [(p.x, p.y) for p in ped.device.footprint]
    device = None
    if ped.device.kind and len(outline) >= 3 and all(math.isfinite(v) for xy in outline for v in xy):  # GEOS aborts on NaN
        device = shapely.Polygon(outline)
    return body, device
