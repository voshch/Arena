# pyright: reportMissingImports=false
"""Create a simple, articulated lever-door source asset with Blender."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

WIDTH = 1.2
HEIGHT = 2.0
LEAF_THICKNESS = 0.045
FRAME_DEPTH = 0.12
HINGE_X = 0.06
HANDLE_X = WIDTH - 0.20
HANDLE_Z = 1.0
HANDLE_STANDOFF = 0.08
HANDLE_PIVOT_Y = -(LEAF_THICKNESS / 2 + HANDLE_STANDOFF / 2)
HANDLE_LIMIT = 0.65
DOOR_LIMIT = math.pi / 2
MODEL_NAME = "lever_door"


def _output_dir() -> Path:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if args:
        return Path(args[0]).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "assets_source" / "Common" / "Object" / MODEL_NAME


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Metallic"].default_value = 0.25 if name == "Lever metal" else 0.0
        principled.inputs["Roughness"].default_value = 0.55
    return material


def _box_mesh(
    name: str,
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    material: bpy.types.Material,
) -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    cube_faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )
    for center, size in boxes:
        cx, cy, cz = center
        sx, sy, sz = (dimension / 2 for dimension in size)
        base = len(vertices)
        vertices.extend(
            (
                (cx - sx, cy - sy, cz - sz),
                (cx + sx, cy - sy, cz - sz),
                (cx + sx, cy + sy, cz - sz),
                (cx - sx, cy + sy, cz - sz),
                (cx - sx, cy - sy, cz + sz),
                (cx + sx, cy - sy, cz + sz),
                (cx + sx, cy + sy, cz + sz),
                (cx - sx, cy + sy, cz + sz),
            )
        )
        faces.extend(tuple(base + index for index in face) for face in cube_faces)
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _cylinder_mesh(
    name: str,
    cylinders: list[tuple[tuple[float, float, float], tuple[float, float, float], float, float]],
    material: bpy.types.Material,
) -> bpy.types.Object:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []
    sides = 20
    for start_raw, end_raw, radius, bevel_radius in cylinders:
        start, end = Vector(start_raw), Vector(end_raw)
        axis = end - start
        rotation = Vector((0, 0, 1)).rotation_difference(axis.normalized())
        base = len(vertices)
        for z, ring_radius in (
            (0.0, bevel_radius),
            (bevel_radius, radius),
            (axis.length - bevel_radius, radius),
            (axis.length, bevel_radius),
        ):
            for side in range(sides):
                angle = 2 * math.pi * side / sides
                point = Vector((ring_radius * math.cos(angle), ring_radius * math.sin(angle), z))
                vertices.append(tuple(start + rotation @ point))
        faces.append(tuple(base + side for side in reversed(range(sides))))
        faces.append(tuple(base + 3 * sides + side for side in range(sides)))
        for ring in range(3):
            lower = base + ring * sides
            upper = lower + sides
            for side in range(sides):
                next_side = (side + 1) % sides
                faces.append((lower + side, lower + next_side, upper + next_side, upper + side))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(material)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _annotation_text() -> str:
    return f"""desc: Simple articulated hinged door with a press-down lever handle.
note: Prototype geometry; verify dimensions, axes, collision and contact forces before experiments.
tags:
- domain::common
- kind::door
articulation:
  links:
    frame: [frame]
    leaf: [leaf]
    handle_link: [handle]
  joints:
  - name: hinge
    type: revolute
    parent: frame
    child: leaf
    origin: [{HINGE_X:.4f}, 0.0, 0.0]
    axis: [0.0, 0.0, 1.0]
    limit: [0.0, {DOOR_LIMIT:.6f}]
    velocity: 1.0
    role: door
    latch: {{by: handle_joint, release: 0.5}}
  - name: handle_joint
    type: revolute
    parent: leaf
    child: handle_link
    origin: [{HANDLE_X:.4f}, {HANDLE_PIVOT_Y:.4f}, {HANDLE_Z:.4f}]
    axis: [0.0, 1.0, 0.0]
    limit: [0.0, {HANDLE_LIMIT:.4f}]
    velocity: 2.0
    role: handle
  states:
    closed: {{hinge: 0.0, handle_joint: 0.0}}
    open: {{hinge: {DOOR_LIMIT:.6f}}}
"""


def main() -> None:
    output = _output_dir()
    output.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    frame_material = _material("Frame paint", (0.30, 0.32, 0.35, 1.0))
    leaf_material = _material("Door paint", (0.48, 0.34, 0.22, 1.0))
    handle_material = _material("Lever metal", (0.72, 0.70, 0.62, 1.0))

    frame_depth = FRAME_DEPTH
    post_width = 0.08
    _box_mesh(
        "frame",
        [
            ((0.0, 0.0, HEIGHT / 2), (post_width, frame_depth, HEIGHT)),
            ((WIDTH, 0.0, HEIGHT / 2), (post_width, frame_depth, HEIGHT)),
            ((WIDTH / 2, 0.0, HEIGHT), (WIDTH + post_width, frame_depth, post_width)),
        ],
        frame_material,
    )

    leaf_x_min = HINGE_X
    leaf_x_max = WIDTH - 0.04
    leaf_height = HEIGHT - 0.08
    _box_mesh(
        "leaf",
        [(((leaf_x_min + leaf_x_max) / 2, 0.0, HEIGHT / 2), (leaf_x_max - leaf_x_min, LEAF_THICKNESS, leaf_height))],
        leaf_material,
    )

    handle_y = HANDLE_PIVOT_Y
    handle_pivot = (HANDLE_X, handle_y, HANDLE_Z)
    hub_length = HANDLE_STANDOFF
    lever_length = 0.14
    lever_radius = 0.012
    grip_radius = 0.016
    handle_cylinders = [
        (
            (HANDLE_X, handle_y - hub_length / 2, HANDLE_Z),
            (HANDLE_X, handle_y + hub_length / 2, HANDLE_Z),
            0.018,
            0.004,
        ),
        (
            handle_pivot,
            (HANDLE_X + lever_length, handle_y, HANDLE_Z),
            lever_radius,
            0.003,
        ),
        (
            (HANDLE_X + lever_length, handle_y, HANDLE_Z),
            (HANDLE_X + lever_length + 0.045, handle_y, HANDLE_Z),
            grip_radius,
            0.008,
        ),
    ]
    _cylinder_mesh("handle", handle_cylinders, handle_material)

    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["frame"]
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    glb_path = output / f"{MODEL_NAME}.glb"
    bpy.ops.export_scene.gltf(
        filepath=str(glb_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
    )
    (output / "annotation.yaml").write_text(_annotation_text(), encoding="utf-8")
    print(f"Created {glb_path}")
    print(f"Created {output / 'annotation.yaml'}")


if __name__ == "__main__":
    main()
