from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from arena_mujoco_msgs.msg import Material
from PIL import Image

from arena_mujoco.mesh_mat import COLLISION_GROUP, add_box_body, add_hull_geoms, add_mesh_body, collision_hulls, ensure_material, mesh_parts, scale_body
from arena_mujoco.scene import RejectedBodies, SceneStore
from arena_mujoco.sensors.laser import cast_rays
from arena_mujoco.services.SpawnUrdf import _attach_robot

_RED = (1.0, 0.0, 0.0, 1.0)
_GREEN = (0.0, 1.0, 0.0, 1.0)


def _colored_box(extents: tuple[float, float, float], center: tuple[float, float, float], rgba: tuple) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    box.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(baseColorFactor=[int(c * 255) for c in rgba]))
    return box


def _two_material_glb(tmp_path: Path) -> str:
    scene = trimesh.Scene(
        [
            _colored_box((1.0, 1.0, 0.1), (0.0, 0.0, 0.5), _RED),
            _colored_box((0.1, 0.1, 0.5), (0.4, 0.4, 0.2), _GREEN),
            _colored_box((0.1, 0.1, 0.5), (-0.4, -0.4, 0.2), _GREEN),
        ]
    )
    path = tmp_path / 'table.glb'
    scene.export(str(path))
    return str(path)


def _textured_glb(tmp_path: Path) -> str:
    plate = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    image = Image.new('RGB', (2048, 1024), (10, 200, 30))
    plate.visual = trimesh.visual.TextureVisuals(
        uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=image),
    )
    path = tmp_path / 'plate.glb'
    plate.export(str(path))
    return str(path)


@pytest.fixture
def store() -> SceneStore:
    store = SceneStore()
    store.ensure_env(0)
    return store


def test_converted_mesh_has_one_part_per_material(tmp_path: Path):
    parts = mesh_parts(_two_material_glb(tmp_path))
    assert sorted(part.rgba for part in parts) == [pytest.approx(_GREEN, abs=0.01), pytest.approx(_RED, abs=0.01)]
    assert all(part.texture is None and part.file.endswith('.obj') for part in parts)
    faces = {tuple(round(c) for c in part.rgba): Path(part.file).read_text().count('\nf ') for part in parts}
    assert faces == {(1, 0, 0, 1): 12, (0, 1, 0, 1): 24}


def test_textured_part_keeps_uv_and_a_size_limited_texture(tmp_path: Path):
    (part,) = mesh_parts(_textured_glb(tmp_path))
    assert Path(part.file).read_text().count('vt ') == 4
    with Image.open(part.texture) as image:
        assert image.size == (1024, 512)
        assert image.getpixel((5, 5)) == (10, 200, 30)


def _ball(name: str, pos: tuple[float, float, float]):
    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name=name, pos=list(pos))
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05, 0.0, 0.0])
        body.add_freejoint().name = f'{name}_free'
        return body

    return build


def test_mesh_body_shows_its_materials_and_collides_as_convex_pieces(store: SceneStore, tmp_path: Path):
    path = _two_material_glb(tmp_path)
    hulls = collision_hulls(path)
    assert len(hulls) >= 3
    cache = store.asset_cache(0)
    store.add_body(0, lambda spec: add_mesh_body(spec, cache, 'table', (2.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0), path))
    store.add_body(0, lambda spec: add_mesh_body(spec, cache, 'table_2', (5.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0), path))
    store.add_body(0, _ball('on_top', (2.0, 0.0, 0.8)))
    store.add_body(0, _ball('below', (2.0, 0.0, 0.05)))
    store.recompile(0)
    model = store.get_model(0)
    meshes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH)
    visuals = meshes[model.geom_group[meshes] == 0]
    colliders = meshes[model.geom_group[meshes] == COLLISION_GROUP]
    assert (len(visuals), len(colliders)) == (4, 2 * len(hulls))
    assert model.nmesh == 2 + len(hulls)
    assert not model.geom_contype[visuals].any()
    assert not model.geom_conaffinity[visuals].any()
    assert model.geom_contype[colliders].all()
    colors = {tuple(model.mat_rgba[model.geom_matid[g]].round(2).tolist()) for g in visuals}
    assert colors == {_RED, _GREEN}

    store.step(500)
    assert store.get_body_pose(0, 'on_top')[0] == pytest.approx((2.0, 0.0, 0.65), abs=0.01)
    assert store.get_body_pose(0, 'below')[0] == pytest.approx((2.0, 0.0, 0.05), abs=0.01)


def test_rays_pass_through_colliders(store: SceneStore, tmp_path: Path):
    crate = tmp_path / 'crate.obj'
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(str(crate))
    cache = store.asset_cache(0)

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        body = spec.worldbody.add_body(name='crate', pos=[2.0, 0.0, 0.5])
        add_hull_geoms(spec, cache, body, [str(crate)])
        spec.worldbody.add_body(name='eye', pos=[0.0, 0.0, 0.5]).add_site(name='lidar')
        return body

    store.add_body(0, build)
    store.add_body(0, _ball('probe', (2.0, 0.0, 1.2)))
    store.recompile(0)
    model, data = store.get_model(0), store.get_data(0)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'lidar')
    assert np.isposinf(cast_rays(model, data, site, np.array([[1.0, 0.0, 0.0]]), np.array([], dtype=int), 0.01, 10.0)[0])
    store.step(500)
    assert store.get_body_pose(0, 'probe')[0][2] == pytest.approx(1.05, abs=0.01)


def test_prim_scale_edit_resizes_boxes_and_meshes(store: SceneStore, tmp_path: Path):
    path = _two_material_glb(tmp_path)
    cache = store.asset_cache(0)
    store.add_body(0, lambda spec: add_box_body(spec, 'crate', (0.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    store.add_body(0, lambda spec: add_mesh_body(spec, cache, 'table', (2.0, 0.0, 0.05), (1.0, 0.0, 0.0, 0.0), path))
    store.recompile(0)
    before = store.get_model(0)

    def resize(scale: tuple[float, float, float]):
        return lambda spec, body: scale_body(spec, cache, body, scale)

    assert store.edit_body(0, 'crate', resize((2.0, 1.0, 0.5)))
    assert store.edit_body(0, 'table', resize((2.0, 2.0, 2.0)))
    assert not store.edit_body(0, 'missing', resize((2.0, 2.0, 2.0)))
    store.compile_dirty()
    after = store.get_model(0)
    crate = after.body_geomadr[mujoco.mj_name2id(after, mujoco.mjtObj.mjOBJ_BODY, 'crate')]
    assert after.geom_size[crate] == pytest.approx((1.0, 0.5, 0.25))
    assert after.nmesh == before.nmesh
    assert np.ptp(after.mesh_vert, axis=0) == pytest.approx(2.0 * np.ptp(before.mesh_vert, axis=0), rel=1e-3)


def test_native_mesh_without_material_file_declares_no_color(tmp_path: Path):
    path = tmp_path / 'tetra.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 2 3 4\nf 1 4 3\n')
    (part,) = mesh_parts(str(path))
    assert part.file == str(path)
    assert part.rgba is None
    assert part.texture is None


def test_degenerate_box_gets_a_positive_thickness(store: SceneStore):
    store.add_body(0, lambda spec: add_box_body(spec, 'poster', (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.5)))
    store.recompile(0)
    model = store.get_model(0)
    geom = model.body_geomadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'poster')]
    assert model.geom_size[geom] == pytest.approx((0.5, 0.0005, 0.25))


def test_stl_beyond_the_native_face_limit_is_converted(tmp_path: Path):
    sphere = trimesh.creation.icosphere(subdivisions=7)
    assert len(sphere.faces) > 200_000
    path = tmp_path / 'dense.stl'
    sphere.export(str(path))
    (part,) = mesh_parts(str(path))
    assert part.file.endswith('.obj')
    assert part.rgba is None
    assert 1000 < Path(part.file).read_text().count('\nf ') < 200_000


def test_robot_with_an_unreadable_mesh_is_rolled_back_whole(store: SceneStore, tmp_path: Path):
    robot = mujoco.MjSpec()
    mesh = robot.add_mesh()
    mesh.name = 'chassis'
    mesh.file = str(tmp_path / 'missing.stl')
    robot.add_material(name='paint')
    base = robot.worldbody.add_body(name='base')
    base.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='chassis', material='paint')

    cache = store.asset_cache(0)
    store.add_body(0, lambda spec: _attach_robot(spec, cache, robot, 'r_', 'base', (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0), True))
    with pytest.raises(RejectedBodies):
        store.recompile(0)

    model = store.get_model(0)
    assert (model.nbody, model.nmesh, model.nmat, model.njnt) == (1, 0, 0, 0)
    store.add_body(0, lambda spec: add_box_body(spec, 'crate', (1.0, 0.0, 0.5), (1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    store.recompile(0)
    assert store.find(0, 'crate')


def _material(store: SceneStore, tmp_path: Path, mdl: str) -> tuple[mujoco.MjModel, int]:
    path = tmp_path / 'Sample.mdl'
    path.write_text(mdl)
    cache = store.asset_cache(0)
    message = Material(name='Sample', path=str(path))

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        return add_box_body(spec, 'wall', (0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0), (2.0, 0.1, 2.0), ensure_material(spec, cache, message))

    store.add_body(0, build)
    store.recompile(0)
    model = store.get_model(0)
    return model, int(model.geom_matid[model.body_geomadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wall')]])


def test_material_uses_the_diffuse_texture_slot(store: SceneStore, tmp_path: Path):
    Image.new('RGB', (8, 8), (200, 30, 30)).save(tmp_path / 'base.png')
    model, material = _material(store, tmp_path, 'diffuse_color_constant: color(0.2f, 0.2f, 0.2f),\n    diffuse_texture: texture_2d("./base.png", ::tex::gamma_srgb),\n')
    texture = model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB]
    assert texture >= 0
    assert tuple(model.tex_data[model.tex_adr[texture] : model.tex_adr[texture] + 3]) == (200, 30, 30)


def test_material_without_a_slot_finds_its_diffuse_image_by_name(store: SceneStore, tmp_path: Path):
    Image.new('RGB', (8, 8), (90, 60, 30)).save(tmp_path / 'cork_diff.png')
    Image.new('RGB', (8, 8), (0, 255, 0)).save(tmp_path / 'cork_multi_r_ao_b_diff.png')
    model, material = _material(store, tmp_path, 'x = texture_2d("./cork_multi_r_ao_b_diff.png");\ny = texture_2d("./cork_diff.png");\n')
    texture = model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB]
    assert tuple(model.tex_data[model.tex_adr[texture] : model.tex_adr[texture] + 3]) == (90, 60, 30)


def test_material_without_images_takes_its_declared_diffuse_color(store: SceneStore, tmp_path: Path):
    model, material = _material(store, tmp_path, 'export material Sample(\n    color diffuse_tint = color(0.838f, 0.802f, 0.775f) [[\n')
    assert model.mat_texid[material, mujoco.mjtTextureRole.mjTEXROLE_RGB] == -1
    assert model.mat_rgba[material] == pytest.approx((0.838, 0.802, 0.775, 1.0))


def test_link_visual_with_a_missing_mesh_is_dropped_and_the_robot_still_loads(tmp_path: Path):
    from arena_mujoco.services.SpawnUrdf import _mujoco_compat

    urdf = tmp_path / 'r.urdf'
    urdf.write_text(
        f'<robot name="r"><link name="base"><inertial><mass value="1"/><inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0" ixz="0" iyz="0"/></inertial><visual><geometry><mesh filename="file://{tmp_path}/missing.dae"/></geometry></visual><collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision></link></robot>'
    )
    compat, _, _ = _mujoco_compat(str(urdf))
    model = mujoco.MjSpec.from_file(compat).compile()
    assert model.nmesh == 0
    assert model.ngeom == 1


def test_deleting_bodies_compiles_once_and_frees_their_assets(store: SceneStore, tmp_path: Path):
    path = _two_material_glb(tmp_path)
    cache = store.asset_cache(0)
    for index in range(3):
        store.add_body(0, lambda spec, index=index: add_mesh_body(spec, cache, f'env_0/Obstacles/t{index}', (float(index), 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), path))
    store.add_body(0, lambda spec: add_box_body(spec, 'env_0/Walls/w', (0.0, 3.0, 1.0), (1.0, 0.0, 0.0, 0.0), (4.0, 0.1, 2.0)))
    store.recompile(0)
    before = store.get_model(0)
    assert (before.nmesh, before.nmat) == (2 + len(collision_hulls(path)), 2)

    for index in range(3):
        assert store.remove(0, f'env_0/Obstacles/t{index}')
    assert store.get_model(0) is before
    store.compile_dirty()
    after = store.get_model(0)
    assert after is not before
    assert (after.nbody, after.nmesh, after.nmat) == (2, 0, 0)
    store.compile_dirty()
    assert store.get_model(0) is after

    store.add_body(0, lambda spec: add_mesh_body(spec, cache, 'env_0/Obstacles/again', (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), path))
    store.recompile(0)
    assert store.get_model(0).nmesh == before.nmesh
