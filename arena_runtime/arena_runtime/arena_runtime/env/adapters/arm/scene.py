"""Planning-scene geometry of episode objects: URDF link kinematics, OBJ meshes and MoveIt collision objects."""

from __future__ import annotations

import functools
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import attrs
import numpy as np
from arena_simulation_setup.utils.articulation import matrix, rotation
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position, Scale
from geometry_msgs.msg import Point
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive

ID_PREFIX = "arena"
LINK_SEPARATOR = "/"

type Extent = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def object_id(entity: str, link: str | None = None) -> str:
    return f"{ID_PREFIX}{LINK_SEPARATOR}{entity}" if link is None else f"{ID_PREFIX}{LINK_SEPARATOR}{entity}{LINK_SEPARATOR}{link}"


def _triple(text: str | None, default: float) -> tuple[float, float, float]:
    if not text:
        return (default, default, default)
    x, y, z = (float(part) for part in text.split())
    return (x, y, z)


def _origin(element: ET.Element | None) -> Pose:
    node = None if element is None else element.find("origin")
    if node is None:
        return Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity())
    xyz = _triple(node.get("xyz"), 0.0)
    rpy = _triple(node.get("rpy"), 0.0)
    return Pose(position=Position(*xyz), orientation=Orientation.from_euler(rpy))


def _end(joint: ET.Element, tag: str) -> str:
    return str(joint.findall(tag)[0].get("link"))


@attrs.frozen
class UrdfMesh:
    path: Path
    origin: Pose
    scale: tuple[float, float, float]


@attrs.frozen
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    origin: Pose
    axis: Position
    limit: tuple[float, float] | None

    def motion(self, value: float) -> Pose:
        """Child frame relative to its joint frame at this joint value."""
        if self.type in ("revolute", "continuous"):
            return Pose(position=Position(0.0, 0.0, 0.0), orientation=rotation(self.axis, value))
        if self.type == "prismatic":
            return Pose(position=self.axis * value, orientation=Orientation.identity())
        return Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity())

    def clamp(self, value: float) -> float:
        return value if self.limit is None else min(max(value, self.limit[0]), self.limit[1])


@attrs.frozen
class Urdf:
    """Links with their collision meshes and the joints between them, as an asset bundle's URDF writes them."""

    meshes: Mapping[str, tuple[UrdfMesh, ...]]
    joints: tuple[UrdfJoint, ...]

    @property
    def links(self) -> frozenset[str]:
        return frozenset(self.meshes)

    @property
    def root(self) -> str:
        children = {joint.child for joint in self.joints}
        return next(link for link in self.meshes if link not in children)

    def link_frames(self, values: Mapping[str, float]) -> dict[str, Pose]:
        """Pose of every link frame in the model frame at the joint values, unnamed joints at 0."""
        frames = {self.root: Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity())}
        pending = list(self.joints)
        while pending:
            ready = [joint for joint in pending if joint.parent in frames]
            if not ready:
                break
            for joint in ready:
                frames[joint.child] = frames[joint.parent] * joint.origin * joint.motion(joint.clamp(values.get(joint.name, 0.0)))
                pending.remove(joint)
        return frames


@functools.cache
def read_urdf(path: Path) -> Urdf:
    """Links with collision meshes (paths relative to the URDF) and joints of the URDF file."""
    root = ET.parse(path).getroot()
    meshes: dict[str, tuple[UrdfMesh, ...]] = {}
    for link in root.findall("link"):
        found = []
        for collision in link.findall("collision"):
            mesh = collision.find("geometry/mesh")
            if mesh is None:
                continue
            found.append(UrdfMesh(path=path.parent / str(mesh.get("filename")), origin=_origin(collision), scale=_triple(mesh.get("scale"), 1.0)))
        meshes[str(link.get("name"))] = tuple(found)
    joints = []
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        bounds = None if limit is None or limit.get("lower") is None or limit.get("upper") is None else (float(str(limit.get("lower"))), float(str(limit.get("upper"))))
        axis = joint.find("axis")
        direction = Position(1.0, 0.0, 0.0) if axis is None or axis.get("xyz") is None else Position(*_triple(axis.get("xyz"), 0.0)).normalized()
        joints.append(
            UrdfJoint(
                name=str(joint.get("name")),
                type=str(joint.get("type")),
                parent=_end(joint, "parent"),
                child=_end(joint, "child"),
                origin=_origin(joint),
                axis=direction,
                limit=bounds,
            )
        )
    return Urdf(meshes=meshes, joints=tuple(joints))


def urdf_root_link(description: str) -> str:
    """The link of a URDF string that is no joint's child."""
    root = ET.fromstring(description)
    children = {_end(joint, "child") for joint in root.findall("joint")}
    return next(str(link.get("name")) for link in root.findall("link") if link.get("name") not in children)


@functools.cache
def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (N, 3) and fan-triangulated faces (M, 3) of an OBJ file, from its `v` and `f` lines."""
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "f":
            corners = []
            for token in parts[1:]:
                index = int(token.split("/")[0])
                corners.append(len(vertices) + index if index < 0 else index - 1)
            triangles.extend((corners[0], corners[i], corners[i + 1]) for i in range(1, len(corners) - 1))
    points = np.array(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.array(triangles, dtype=np.int64).reshape(-1, 3)
    points.flags.writeable = False
    faces.flags.writeable = False
    return points, faces


def _scale_vector(scale: Scale | None) -> np.ndarray:
    return np.ones(3) if scale is None else np.array([scale.x, scale.y, scale.z])


def _carry(pose: Pose, points: np.ndarray) -> np.ndarray:
    transform = matrix(pose)
    return points @ transform[:3, :3].T + transform[:3, 3]


def scaled_pose(pose: Pose, scale: Scale | None) -> Pose:
    """The pose with its translation scaled componentwise."""
    if scale is None:
        return pose
    return Pose(position=Position(pose.position.x * scale.x, pose.position.y * scale.y, pose.position.z * scale.z), orientation=pose.orientation)


@attrs.frozen
class Articulated:
    """An articulated episode object: its map-frame pose, URDF and current joint values."""

    name: str
    pose: Pose
    scale: Scale | None
    urdf: Urdf
    values: Mapping[str, float]

    def link_frame(self, link: str) -> Pose:
        """Map-frame pose of a URDF link frame at the current joint values."""
        return self.pose * scaled_pose(self.urdf.link_frames(self.values)[link], self.scale)

    def link_mesh(self, link: str) -> tuple[np.ndarray, np.ndarray]:
        """Vertices in the object frame and triangles of a link's collision meshes at the current joint values, empty for a link without any."""
        placed = self.urdf.link_frames(self.values)[link]
        scale = _scale_vector(self.scale)
        points: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        offset = 0
        for mesh in self.urdf.meshes[link]:
            vertices, triangles = read_obj(mesh.path)
            points.append(_carry(placed * mesh.origin, vertices * np.array(mesh.scale)) * scale)
            faces.append(triangles + offset)
            offset += len(vertices)
        if not points:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(points), np.concatenate(faces)


@attrs.frozen
class StaticBox:
    """A rigid episode object known by its annotated box in the object frame."""

    name: str
    pose: Pose
    scale: Scale | None
    extent: Extent


def mesh_message(vertices: np.ndarray, triangles: np.ndarray) -> Mesh:
    mesh = Mesh()
    mesh.vertices = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in vertices]
    mesh.triangles = [MeshTriangle(vertex_indices=[int(i) for i in triangle]) for triangle in triangles]
    return mesh


def _distance_to_base(corners: np.ndarray) -> float:
    low, high = corners[:, :2].min(axis=0), corners[:, :2].max(axis=0)
    gap = np.maximum(np.maximum(low, -high), 0.0)
    return float(math.hypot(gap[0], gap[1]))


def _object(identifier: str, frame: str, pose: Pose) -> CollisionObject:
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = identifier
    obj.pose = pose.to_msg()
    obj.operation = CollisionObject.ADD
    return obj


def collision_objects(entities: Iterable[Articulated | StaticBox], root: Pose, frame: str, radius: float) -> list[CollisionObject]:
    """ADD objects of the entities expressed in `frame`, whose map-frame pose is `root`, leaving out the ones farther than `radius` from its origin on the floor plane."""
    objects: list[CollisionObject] = []
    for entity in entities:
        placed = root.inverse() * entity.pose
        if isinstance(entity, Articulated):
            for link in entity.urdf.links:
                vertices, triangles = entity.link_mesh(link)
                if len(vertices) == 0 or _distance_to_base(_carry(placed, vertices)) > radius:
                    continue
                obj = _object(object_id(entity.name, link), frame, placed)
                obj.meshes = [mesh_message(vertices, triangles)]
                obj.mesh_poses = [Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.identity()).to_msg()]
                objects.append(obj)
        else:
            scale = _scale_vector(entity.scale)
            low = np.array([side[0] for side in entity.extent]) * scale
            high = np.array([side[1] for side in entity.extent]) * scale
            corners = _carry(placed, np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]))
            if _distance_to_base(corners) > radius:
                continue
            obj = _object(object_id(entity.name), frame, placed)
            obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(size) for size in high - low])]
            obj.primitive_poses = [Pose(position=Position(*((low + high) / 2.0)), orientation=Orientation.identity()).to_msg()]
            objects.append(obj)
    return objects


def scene_diff(previous: Iterable[str], objects: Sequence[CollisionObject], frame: str) -> PlanningScene:
    """One planning scene diff that removes the previous ids and then adds the objects."""
    scene = PlanningScene()
    scene.is_diff = True
    removals = []
    for identifier in previous:
        gone = CollisionObject()
        gone.header.frame_id = frame
        gone.id = identifier
        gone.operation = CollisionObject.REMOVE
        removals.append(gone)
    scene.world.collision_objects = [*removals, *objects]
    return scene
