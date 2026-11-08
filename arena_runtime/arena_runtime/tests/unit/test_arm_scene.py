from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import pytest

try:
    from arena_runtime.env.adapters.arm.scene import (
        Articulated,
        StaticBox,
        collision_objects,
        object_id,
        read_obj,
        read_urdf,
        scene_diff,
        urdf_root_link,
    )
    from arena_simulation_setup.utils.articulation import Articulation, matrix
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position, Scale
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

MESH_TURN = (math.pi / 2, 0.0, 0.0)
CHEST = {
    "links": {"body": ["body"], "drawer_a": ["drawer_a"], "drawer_b": ["drawer_b"], "door": ["door"], "knob": []},
    "joints": [
        {"name": "slide_a", "type": "prismatic", "parent": "body", "child": "drawer_a", "origin": [0.3, -0.2, 0.8], "axis": [0, 1, 0], "limit": [-0.3, 0.0]},
        {"name": "slide_b", "type": "prismatic", "parent": "body", "child": "drawer_b", "origin": [-0.3, -0.2, 0.5], "axis": [0, 1, 0], "limit": [-0.3, 0.0]},
        {"name": "hinge", "type": "revolute", "parent": "body", "child": "door", "origin": [0.5, 0.2, 0.1], "axis": [0, 0, 1], "limit": [0.0, 1.6]},
        {"name": "turn", "type": "revolute", "parent": "door", "child": "knob", "origin": [0.6, 0.3, 0.4], "axis": [0, 1, 0], "limit": [0.0, 2.0]},
    ],
}
CUBE = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)]


def _vector(values: tuple[float, ...]) -> str:
    return " ".join(repr(float(value)) for value in values)


def _write_cube(path: Path, size: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"v {_vector(tuple(size * c for c in corner))}\n" for corner in CUBE) + "f 1 2 3 4\nf 5 6 7 8\n")


def _write_bundle(root: Path, articulation: Articulation) -> tuple[Path, dict[str, np.ndarray]]:
    origins: dict[str, np.ndarray] = {articulation.root: np.zeros(3)}
    for joint in articulation.order():
        origins[joint.child] = np.array(list(joint.origin))
    lines = ['<?xml version="1.0"?>', '<robot name="chest">']
    for link, origin in origins.items():
        mesh = f"links/{link}/{link}.obj/{link}.obj"
        _write_cube(root / mesh, 0.1)
        lines.append(f'<link name="{link}"><collision><origin xyz="{_vector(tuple(-origin))}" rpy="{_vector(MESH_TURN)}"/><geometry><mesh filename="{mesh}"/></geometry></collision></link>')
    for joint in articulation.order():
        relative = np.array(list(joint.origin)) - origins[joint.parent]
        lower, upper = joint.limit or (0.0, 0.0)
        lines.append(f'<joint name="{joint.name}" type="{joint.type}"><origin xyz="{_vector(tuple(relative))}" rpy="0 0 0"/><parent link="{joint.parent}"/><child link="{joint.child}"/><axis xyz="{_vector(tuple(joint.axis))}"/><limit lower="{lower}" upper="{upper}" effort="1" velocity="1"/></joint>')
    lines.append("</robot>")
    (root / "chest.urdf").write_text("\n".join(lines))
    return root / "chest.urdf", origins


def _seeded(articulation: Articulation, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {joint.name: rng.uniform(*joint.bounds) for joint in articulation.movable}


def _yaw(degrees: float) -> Orientation:
    return Orientation.from_yaw(math.radians(degrees))


@pytest.fixture
def chest(tmp_path: Path) -> tuple[Articulation, Path, dict[str, np.ndarray]]:
    articulation = Articulation.parse(CHEST)
    urdf, origins = _write_bundle(tmp_path / "chest", articulation)
    return articulation, urdf, origins


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_urdf_link_frames_match_articulation_link_poses_shifted_by_the_link_origins(chest, seed: int) -> None:
    articulation, urdf_path, origins = chest
    urdf = read_urdf(urdf_path)
    values = _seeded(articulation, seed)
    frames = urdf.link_frames(values)
    expected = articulation.link_poses(values)
    assert urdf.root == articulation.root
    for link, frame in frames.items():
        geometry = frame * Pose(position=Position(*(-origins[link])), orientation=Orientation.identity())
        assert np.allclose(matrix(geometry), matrix(expected[link]), atol=1e-9)


@pytest.mark.parametrize("seed", [4, 5])
def test_articulated_link_meshes_equal_the_turned_cube_carried_by_the_link_poses(chest, seed: int) -> None:
    articulation, urdf_path, _ = chest
    values = _seeded(articulation, seed)
    entity = Articulated(name="chest", pose=Pose(), scale=None, urdf=read_urdf(urdf_path), values=values)
    poses = articulation.link_poses(values)
    turn = matrix(Pose(position=Position(0.0, 0.0, 0.0), orientation=Orientation.from_euler(MESH_TURN)))
    cube = 0.1 * np.array(CUBE)
    for link in articulation.links:
        vertices, triangles = entity.link_mesh(link)
        carried = matrix(poses[link]) @ turn
        assert np.allclose(vertices, cube @ carried[:3, :3].T + carried[:3, 3], atol=1e-9)
        assert triangles.shape == (4, 3)


def test_link_frame_of_a_placed_scaled_object_scales_the_motion_and_the_origin(chest) -> None:
    _, urdf_path, _ = chest
    placed = Articulated(name="chest", pose=Pose(position=Position(1.0, 2.0, 0.0), orientation=_yaw(90)), scale=Scale(2.0, 2.0, 2.0), urdf=read_urdf(urdf_path), values={"slide_a": -0.1})
    frame = placed.link_frame("drawer_a")
    assert (frame.position.x, frame.position.y, frame.position.z) == pytest.approx((1.6, 2.6, 1.6))
    assert frame.orientation.to_yaw() == pytest.approx(math.pi / 2)


def test_obj_reader_triangulates_every_face_form_and_resolves_negative_indices(tmp_path: Path) -> None:
    path = tmp_path / "cube.obj"
    lines = [f"v {_vector(corner)} 1.0" for corner in CUBE]
    lines += ["# comment", "vn 0 0 1", "f 1 2 3 4", "f 5/1 6/2 7/3", "f 5/1/1 6/2/2 7/3/3 8/4/4", "f -8 -7 -6"]
    path.write_text("\n".join(lines) + "\n")
    vertices, triangles = read_obj(path)
    assert vertices.shape == (8, 3)
    assert np.allclose(vertices, CUBE)
    assert triangles.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 5, 6], [4, 6, 7], [0, 1, 2]]


def test_obj_reader_resolves_negative_indices_against_the_vertices_read_so_far(tmp_path: Path) -> None:
    path = tmp_path / "late.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nv 0 0 1\nf -1 -3 -2\n")
    _, triangles = read_obj(path)
    assert triangles.tolist() == [[0, 1, 2], [3, 1, 2]]


def test_urdf_root_link_is_the_link_no_joint_makes_a_child() -> None:
    description = '<robot name="r"><link name="b"/><link name="a"/><link name="c"/><joint name="j" type="fixed"><parent link="a"/><child link="b"/></joint><joint name="k" type="fixed"><parent link="b"/><child link="c"/></joint></robot>'
    assert urdf_root_link(description) == "a"


def _episode(chest) -> list[Articulated | StaticBox]:
    articulation, urdf_path, _ = chest
    near = Articulated(name="chest", pose=Pose(position=Position(2.0, 0.0, 0.0), orientation=_yaw(90)), scale=None, urdf=read_urdf(urdf_path), values={"slide_a": -0.2, "slide_b": 0.0, "hinge": 0.0, "turn": 0.0})
    far = StaticBox(name="crate", pose=Pose(position=Position(9.0, 0.0, 0.0), orientation=Orientation.identity()), scale=None, extent=((-0.2, 0.2), (-0.3, 0.3), (0.0, 0.4)))
    return [near, far]


def test_collision_objects_name_links_and_leave_out_objects_beyond_the_radius(chest) -> None:
    root = Pose(position=Position(1.0, 0.0, 0.0), orientation=Orientation.identity())
    objects = collision_objects(_episode(chest), root, "env/tiago/base_footprint", 1.5)
    assert {obj.id for obj in objects} == {object_id("chest", link) for link in ("body", "drawer_a", "drawer_b", "door", "knob")}
    assert {obj.header.frame_id for obj in objects} == {"env/tiago/base_footprint"}
    assert {obj.operation for obj in objects} == {CollisionObject.ADD}
    for obj in objects:
        assert (obj.pose.position.x, obj.pose.position.y, obj.pose.position.z) == pytest.approx((1.0, 0.0, 0.0))
        assert Orientation.from_msg(obj.pose.orientation).to_yaw() == pytest.approx(math.pi / 2)
        assert len(obj.meshes) == 1
        assert len(obj.meshes[0].triangles) == 4
        assert len(obj.meshes[0].vertices) == 8


def test_collision_objects_keep_a_box_object_near_a_root_far_from_the_chest(chest) -> None:
    root = Pose(position=Position(8.5, 0.0, 0.0), orientation=Orientation.identity())
    objects = {obj.id: obj for obj in collision_objects(_episode(chest), root, "base_footprint", 1.0)}
    assert set(objects) == {object_id("crate")}
    crate = objects[object_id("crate")]
    assert crate.primitives[0].type == SolidPrimitive.BOX
    assert list(crate.primitives[0].dimensions) == pytest.approx([0.4, 0.6, 0.4])
    assert (crate.primitive_poses[0].position.x, crate.primitive_poses[0].position.y, crate.primitive_poses[0].position.z) == pytest.approx((0.0, 0.0, 0.2))
    assert (crate.pose.position.x, crate.pose.position.y) == pytest.approx((0.5, 0.0))


def test_collision_objects_scale_a_static_box_and_carry_it_into_the_robot_frame() -> None:
    crate = StaticBox(name="crate", pose=Pose(position=Position(0.0, 3.0, 0.0), orientation=Orientation.identity()), scale=Scale(2.0, 1.0, 1.0), extent=((-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0)))
    root = Pose(position=Position(0.0, 2.0, 0.0), orientation=_yaw(90))
    (obj,) = collision_objects([crate], root, "base_footprint", 5.0)
    assert list(obj.primitives[0].dimensions) == pytest.approx([2.0, 1.0, 1.0])
    assert (obj.pose.position.x, obj.pose.position.y) == pytest.approx((1.0, 0.0))
    assert Orientation.from_msg(obj.pose.orientation).to_yaw() == pytest.approx(-math.pi / 2)


def test_scene_diff_removes_the_previous_ids_before_adding_the_current_objects(chest) -> None:
    root = Pose(position=Position(1.0, 0.0, 0.0), orientation=Orientation.identity())
    objects = collision_objects(_episode(chest), root, "base_footprint", 1.5)
    scene = scene_diff(["arena/gone/a", "arena/chest/body"], objects, "base_footprint")
    assert scene.is_diff
    operations = [(obj.operation, obj.id) for obj in scene.world.collision_objects]
    assert operations[:2] == [(CollisionObject.REMOVE, "arena/gone/a"), (CollisionObject.REMOVE, "arena/chest/body")]
    assert all(operation == CollisionObject.ADD for operation, _ in operations[2:])
    assert len(operations) == 2 + len(objects)
