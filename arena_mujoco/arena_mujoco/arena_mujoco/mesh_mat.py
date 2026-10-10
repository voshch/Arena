"""Pure MuJoCo body/geom/material builder helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from arena_simulation_setup.utils.material import MdlUtil
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import trimesh
    from arena_mujoco_msgs.msg import Material as MaterialMsg

    from arena_mujoco.scene import AssetCache

TILE_M = 1.0

_NATIVE_MESH_EXTS = ('.obj', '.stl', '.msh')
_MAX_MESH_FACES = 200_000
_STL_HEADER_BYTES = 84
_STL_FACE_BYTES = 50
_HULL_POINTS = 20_000
_HULL_CONCAVITY = 0.1
_MAX_HULLS = 16
_MAX_HULL_VERTICES = 32
_HULL_WORKERS = min(8, os.cpu_count() or 1)
COLLISION_GROUP = 3
_MAX_TEXTURE_PX = 1024
_MIN_BOX_M = 1e-3

_CACHE_DIR = Path(tempfile.gettempdir()) / 'arena_mjmesh'

_MDL_TEXTURE = re.compile(r'texture_2d\("([^"]+)"')
_MDL_DIFFUSE_COLOR = re.compile(r'diffuse\w*\s*[:=]\s*color\(([^)]*)\)')
_DIFFUSE_SUFFIXES = ('diff', 'diffuse', 'albedo', 'basecolor')

_logger = logging.getLogger(__name__)

_mat_counter = itertools.count(1)
_tex_counter = itertools.count(1)
_mesh_counter = itertools.count(1)


@dataclass(frozen=True)
class MeshPart:
    """One single-material piece of a mesh file, in a format MuJoCo loads."""

    file: str
    rgba: tuple[float, float, float, float] | None
    texture: str | None


_parts_cache: dict[str, tuple[MeshPart, ...]] = {}


def mesh_parts(path: str) -> tuple[MeshPart, ...]:
    """Split a mesh file into per-material MuJoCo meshes, converting through an on-disk cache."""
    cached = _parts_cache.get(path)
    if cached is not None:
        return cached
    if _loads_natively(path):
        albedo = obj_albedo(path)
        parts: tuple[MeshPart, ...] = (
            MeshPart(
                file=path,
                rgba=albedo if isinstance(albedo, tuple) else None,
                texture=str(albedo) if isinstance(albedo, Path) else None,
            ),
        )
    else:
        parts = _converted_parts(Path(path))
    _parts_cache[path] = parts
    return parts


def _loads_natively(path: str) -> bool:
    """True for a mesh MuJoCo reads itself, its STL decoder stops at _MAX_MESH_FACES faces."""
    ext = os.path.splitext(path)[1].lower()
    if ext == '.stl':
        return os.path.getsize(path) <= _STL_HEADER_BYTES + _STL_FACE_BYTES * _MAX_MESH_FACES
    return ext in _NATIVE_MESH_EXTS


@dataclass
class PartGroup:
    """Triangles of one material, uv per vertex when image is set."""

    vertices: np.ndarray
    faces: np.ndarray
    rgba: tuple[float, float, float, float] | None
    image: Image.Image | None = None
    uv: np.ndarray | None = None


def cache_entry(sources: Sequence[Path], variant: str, manifest: str, write: Callable[[Path, str], None]) -> Path:
    """Cache directory for the content of sources, filled once by write(stage, stem) and renamed into place."""
    stats = [(source, source.stat()) for source in sources]
    key = '|'.join(f'{source}:{stat.st_mtime_ns}:{stat.st_size}' for source, stat in stats)
    digest = hashlib.md5(f'{key}|{variant}'.encode(), usedforsecurity=False).hexdigest()[:16]
    out_dir = _CACHE_DIR / digest
    if not (out_dir / manifest).is_file():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f'{digest}_', dir=_CACHE_DIR))
        write(stage, f'{sources[0].stem}_{digest[:8]}')
        try:
            stage.rename(out_dir)
        except OSError:
            shutil.rmtree(stage, ignore_errors=True)
    return out_dir


def cached_parts(sources: Sequence[Path], build: Callable[[], list[PartGroup]], texture_px: int = _MAX_TEXTURE_PX) -> tuple[MeshPart, ...]:
    """Parts of build() written once per content of sources under the mesh cache, read back from disk afterwards."""
    out_dir = cache_entry(sources, str(texture_px), 'parts.json', lambda stage, stem: _write_parts(stage, stem, build(), texture_px))
    return tuple(
        MeshPart(
            file=str(out_dir / entry['file']),
            rgba=None if entry['rgba'] is None else tuple(entry['rgba']),
            texture=None if entry['texture'] is None else str(out_dir / entry['texture']),
        )
        for entry in json.loads((out_dir / 'parts.json').read_text())
    )


def _write_parts(out_dir: Path, stem: str, groups: list[PartGroup], texture_px: int) -> None:
    """Write one <stem>_<n>.obj per group into out_dir, plus textures and the parts.json manifest."""
    entries = []
    for index, group in enumerate(groups):
        faces = group.faces + 1
        texture = None
        with open(out_dir / f'{stem}_{index}.obj', 'w') as fh:
            np.savetxt(fh, group.vertices, fmt='v %.6f %.6f %.6f')
            if group.image is not None and group.uv is not None:
                np.savetxt(fh, group.uv, fmt='vt %.6f %.6f')
                np.savetxt(fh, np.repeat(faces, 2, axis=1), fmt='f %d/%d %d/%d %d/%d')
                texture = f'{stem}_{index}.png'
                image = group.image.convert('RGB')
                image.thumbnail((texture_px, texture_px))
                image.save(out_dir / texture)
            else:
                np.savetxt(fh, faces, fmt='f %d %d %d')
        entries.append({'file': f'{stem}_{index}.obj', 'rgba': group.rgba if group.rgba is None else list(group.rgba), 'texture': texture})
    (out_dir / 'parts.json').write_text(json.dumps(entries))


def _converted_parts(source: Path) -> tuple[MeshPart, ...]:
    return cached_parts([source], lambda: _trimesh_groups(source))


def _albedo(visual: trimesh.visual.base.Visuals) -> tuple[tuple[float, float, float, float] | None, Image.Image | None]:
    """Diffuse color (None when the file sets none) and, with texture coordinates present, the diffuse image."""
    import trimesh

    if not isinstance(visual, trimesh.visual.TextureVisuals):
        return (tuple(float(c) / 255.0 for c in visual.main_color) if visual.defined else None), None
    material = visual.material
    if isinstance(material, trimesh.visual.material.PBRMaterial):
        material = material.to_simple()
    rgba = tuple(float(c) / 255.0 for c in material.diffuse)
    return rgba, material.image if visual.uv is not None else None


def _trimesh_groups(source: Path) -> list[PartGroup]:
    """Geometry of source merged per material, each sub-mesh a convex hull when the file exceeds the face budget."""
    import trimesh

    loaded = trimesh.load(str(source))
    meshes = loaded.dump(concatenate=False) if isinstance(loaded, trimesh.Scene) else [loaded]
    meshes = [mesh for mesh in meshes if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces)]
    oversized = sum(len(mesh.faces) for mesh in meshes) > _MAX_MESH_FACES

    merged: dict[object, dict] = {}
    for mesh in meshes:
        rgba, image = _albedo(mesh.visual)
        uv = None
        if oversized:
            step = max(1, len(mesh.vertices) // _HULL_POINTS)
            mesh = trimesh.convex.convex_hull(mesh.vertices[::step])
            image = None
        elif image is not None:
            uv = np.asarray(mesh.visual.uv)
        key = ('texture', id(image)) if image is not None else ('rgba', rgba if rgba is None else tuple(round(c, 3) for c in rgba))
        entry = merged.setdefault(key, {'rgba': rgba, 'image': image, 'vertices': [], 'faces': [], 'uv': [], 'count': 0})
        entry['faces'].append(np.asarray(mesh.faces) + entry['count'])
        entry['vertices'].append(np.asarray(mesh.vertices))
        entry['count'] += len(mesh.vertices)
        if uv is not None:
            entry['uv'].append(uv)
    return [
        PartGroup(
            vertices=np.concatenate(entry['vertices']),
            faces=np.concatenate(entry['faces']),
            rgba=entry['rgba'],
            image=entry['image'],
            uv=np.concatenate(entry['uv']) if entry['uv'] else None,
        )
        for entry in merged.values()
    ]


_hulls_cache: dict[str, tuple[str, ...]] = {}


def _write_hulls(out_dir: Path, stem: str, source: Path) -> None:
    """Write the convex pieces of source as <stem>_hull<n>.obj into out_dir, plus the hulls.json manifest."""
    import coacd
    import trimesh

    loaded = trimesh.load(str(source))
    mesh = loaded.to_geometry() if isinstance(loaded, trimesh.Scene) else loaded
    coacd.set_log_level('error')
    pieces = coacd.run_coacd(
        coacd.Mesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32)),
        threshold=_HULL_CONCAVITY,
        max_convex_hull=_MAX_HULLS,
    )
    files = []
    for index, (vertices, faces) in enumerate(pieces):
        files.append(f'{stem}_hull{index}.obj')
        with open(out_dir / files[-1], 'w') as fh:
            np.savetxt(fh, vertices, fmt='v %.6f %.6f %.6f')
            np.savetxt(fh, np.asarray(faces) + 1, fmt='f %d %d %d')
    (out_dir / 'hulls.json').write_text(json.dumps(files))


def collision_hulls(path: str) -> tuple[str, ...]:
    """Convex pieces of a mesh file as OBJ files, decomposed once per file content into the mesh cache."""
    cached = _hulls_cache.get(path)
    if cached is None:
        source = Path(path)
        out_dir = cache_entry([source], f'hulls:{_HULL_CONCAVITY}:{_MAX_HULLS}', 'hulls.json', lambda stage, stem: _write_hulls(stage, stem, source))
        cached = _hulls_cache[path] = tuple(str(out_dir / file) for file in json.loads((out_dir / 'hulls.json').read_text()))
    return cached


def prepare_hulls(paths: Sequence[str]) -> None:
    """Decompose the mesh files not in the cache yet, several at a time, a file that fails gets no hulls."""

    def decompose(path: str) -> None:
        try:
            collision_hulls(path)
        except Exception as exc:  # noqa: BLE001 - an undecomposable mesh still gets its visual
            _logger.warning(f'{path} collides with nothing, its convex decomposition failed: {exc!r}')
            _hulls_cache[path] = ()

    missing = [path for path in dict.fromkeys(paths) if path not in _hulls_cache]
    if missing:
        with ThreadPoolExecutor(max_workers=_HULL_WORKERS, thread_name_prefix='mj_hull') as pool:
            list(pool.map(decompose, missing))


def part_material(spec: mujoco.MjSpec, cache: AssetCache, part: MeshPart) -> str | None:
    """Material for a mesh part: its texture, else its flat color, None when the file declares neither."""
    if part.texture is not None:
        return ensure_texture_material(spec, cache, part.texture, tiled=False)
    if part.rgba is not None:
        return ensure_rgba_material(spec, cache, part.rgba)
    return None


def _rgba_for(key: str) -> list[float]:
    """Map a material id to a stable muted rgba so distinct materials read apart."""
    key = (key or '').strip()
    if not key:
        return [0.6, 0.6, 0.6, 1.0]
    h = 0
    for c in key:
        h = (h * 131 + ord(c)) & 0xFFFFFF
    return [
        0.45 + ((h >> 16) & 0xFF) / 255.0 * 0.4,
        0.45 + ((h >> 8) & 0xFF) / 255.0 * 0.4,
        0.45 + (h & 0xFF) / 255.0 * 0.4,
        1.0,
    ]


def _texture_file(image_path: str) -> str:
    """image_path itself, or a cached copy scaled down to the texture size limit."""
    with Image.open(image_path) as img:
        if max(img.size) <= _MAX_TEXTURE_PX:
            return image_path
        stat = os.stat(image_path)
        digest = hashlib.md5(f'{image_path}:{stat.st_mtime_ns}:{stat.st_size}'.encode(), usedforsecurity=False).hexdigest()[:16]
        out = _CACHE_DIR / f'texture_{digest}.png'
        if not out.is_file():
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            small = img.convert('RGB')
            small.thumbnail((_MAX_TEXTURE_PX, _MAX_TEXTURE_PX))
            stage = out.with_suffix(f'.{os.getpid()}.png')
            small.save(stage)
            stage.replace(out)
        return str(out)


def ensure_texture_material(
    spec: mujoco.MjSpec,
    cache: AssetCache,
    image_path: str,
    *,
    tiled: bool,
) -> str:
    """Register a 2d file texture and a bound material, returning the material name."""
    cache_key = ('tex_mat', image_path, tiled)
    known = cache.get(cache_key)
    if known is not None:
        return known

    tex = spec.add_texture(name=f'tex_{next(_tex_counter)}', type=mujoco.mjtTexture.mjTEXTURE_2D)
    tex.file = _texture_file(image_path)
    cache.add(None, tex)

    mat = spec.add_material(name=f'mat_{next(_mat_counter)}')
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    if tiled:
        mat.texuniform = True
        mat.texrepeat = (2.0 / TILE_M, 2.0 / TILE_M)
    return cache.add(cache_key, mat)


def ensure_rgba_material(spec: mujoco.MjSpec, cache: AssetCache, rgba: tuple[float, float, float, float]) -> str:
    """Register a flat-color material for rgba, returning its name. Deduplicates by ('rgba', rgba)."""
    cache_key = ('rgba', tuple(rgba))
    known = cache.get(cache_key)
    if known is not None:
        return known
    mat = spec.add_material(name=f'mat_{next(_mat_counter)}')
    mat.rgba = list(rgba)
    return cache.add(cache_key, mat)


def _mdl_albedo(mdl_path: Path) -> Path | tuple[float, float, float, float] | None:
    """Diffuse image of an .mdl material, else the diffuse color it declares, None when it names neither."""
    texture = MdlUtil(mdl_path).texture('diffuse_texture')
    if texture is not None:
        return texture
    source = mdl_path.read_text()
    for relative in _MDL_TEXTURE.findall(source):
        stem = Path(relative).stem.lower()
        if stem.endswith(_DIFFUSE_SUFFIXES) and 'multi' not in stem:
            return mdl_path.parent / relative
    color = _MDL_DIFFUSE_COLOR.search(source)
    if color is None:
        return None
    red, green, blue = (float(channel.strip().rstrip('f')) for channel in color.group(1).split(','))
    return (red, green, blue, 1.0)


def ensure_material(spec: mujoco.MjSpec, cache: AssetCache, mat_msg: MaterialMsg) -> str:
    """Realize a Material msg and return its name, a name-derived flat color when its file yields no albedo."""
    hash_key = mat_msg.name or mat_msg.path
    flat_cache_key = ('flat', hash_key)
    known = cache.get(flat_cache_key)
    if known is not None:
        return known

    try:
        albedo = _mdl_albedo(Path(mat_msg.path)) if mat_msg.path else None
        if isinstance(albedo, Path):
            return ensure_texture_material(spec, cache, str(albedo), tiled=True)
        if albedo is not None:
            return ensure_rgba_material(spec, cache, albedo)
        _logger.warning(f'material {mat_msg.name!r} ({mat_msg.path or "no file"}) declares no diffuse texture or color, drawn in a name-derived color')
    except (OSError, ValueError) as exc:
        _logger.warning(f'material {mat_msg.name!r} ({mat_msg.path}) drawn in a name-derived color: {exc}')

    mat = spec.add_material(name=f'mat_{next(_mat_counter)}')
    mat.rgba = _rgba_for(hash_key)
    return cache.add(flat_cache_key, mat)


def obj_albedo(obj_path: str) -> Path | tuple[float, float, float, float] | None:
    """Parse the .mtl sidecar for an OBJ file and return its diffuse albedo."""
    try:
        obj = Path(obj_path)
        if not obj.is_file():
            return None
        mtl_name: str | None = None
        with obj.open() as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith('mtllib '):
                    mtl_name = stripped[len('mtllib ') :].strip()
                    break
        if not mtl_name:
            return None
        mtl_path = obj.parent / mtl_name
        if not mtl_path.is_file():
            return None

        map_kd_path: Path | None = None
        kd_rgba: tuple[float, float, float, float] | None = None
        with mtl_path.open() as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.lower().startswith('map_kd') and map_kd_path is None:
                    tokens = stripped.split()
                    if len(tokens) >= 2:
                        candidate = mtl_path.parent / tokens[-1]
                        if candidate.is_file():
                            map_kd_path = candidate
                elif stripped.lower().startswith('kd ') and kd_rgba is None:
                    parts = stripped.split()
                    if len(parts) == 4:
                        kd_rgba = (float(parts[1]), float(parts[2]), float(parts[3]), 1.0)
        if map_kd_path is not None:
            return map_kd_path
        return kd_rgba
    except Exception:
        return None


def add_box_body(
    spec: mujoco.MjSpec,
    name: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    size: tuple[float, float, float],
    material_name: str | None = None,
    geom_quat: tuple[float, float, float, float] | None = None,
) -> mujoco.MjsBody:
    """Add a static body with a single box geom to spec."""
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = list(pos)
    body.quat = list(quat)

    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_BOX
    geom.size = [max(side, _MIN_BOX_M) / 2.0 for side in size]
    if geom_quat is not None:
        geom.quat = list(geom_quat)
    if material_name is not None:
        geom.material = material_name

    return body


def add_plane_body(
    spec: mujoco.MjSpec,
    name: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    half_size: tuple[float, float, float],
    material_name: str | None = None,
    group: int = 0,
) -> mujoco.MjsBody:
    """Add a static body with a single visual-only plane geom to spec."""
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = list(pos)
    body.quat = list(quat)

    geom = body.add_geom()
    geom.type = mujoco.mjtGeom.mjGEOM_PLANE
    geom.size = [half_size[0], half_size[1], half_size[2]]
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = group
    if material_name is not None:
        geom.material = material_name

    return body


def _mesh_asset(spec: mujoco.MjSpec, cache: AssetCache, kind: str, file: str, scale: tuple[float, float, float]) -> str:
    """Name of the mesh asset for file at scale, kind 'hull' capping its convex hull for collision."""
    mesh_cache_key = (kind, file, tuple(scale))
    mesh_name = cache.get(mesh_cache_key)
    if mesh_name is None:
        mesh = spec.add_mesh()
        mesh.name = f'mesh_{next(_mesh_counter)}'
        mesh.file = file
        mesh.scale = list(scale)
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        if kind == 'hull':
            mesh.maxhullvert = _MAX_HULL_VERTICES
        mesh_name = cache.add(mesh_cache_key, mesh)
    return mesh_name


def add_part_geoms(
    spec: mujoco.MjSpec,
    cache: AssetCache,
    body: mujoco.MjsBody,
    parts: Sequence[MeshPart],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    group: int = 0,
) -> list[mujoco.MjsGeom]:
    """Add one visual, non-colliding mesh geom per part to body."""
    geoms = []
    for part in parts:
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = _mesh_asset(spec, cache, 'mesh', part.file, scale)
        geom.pos = list(pos)
        geom.group = group
        material = part_material(spec, cache, part)
        if material is not None:
            geom.material = material
        geom.contype = 0
        geom.conaffinity = 0
        geoms.append(geom)
    return geoms


def add_hull_geoms(
    spec: mujoco.MjSpec,
    cache: AssetCache,
    body: mujoco.MjsBody,
    hulls: Sequence[str],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[mujoco.MjsGeom]:
    """Add one colliding convex mesh geom per hull file to body, in the group cameras and rays skip."""
    geoms = []
    for file in hulls:
        geom = body.add_geom()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = _mesh_asset(spec, cache, 'hull', file, scale)
        geom.group = COLLISION_GROUP
        geoms.append(geom)
    return geoms


def scale_body(spec: mujoco.MjSpec, cache: AssetCache, body: mujoco.MjsBody, scale: tuple[float, float, float]) -> None:
    """Resize a compiled prim body: a box takes scale as its side lengths, a mesh geom its file at that scale."""
    for geom in body.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            geom.size = [max(side, _MIN_BOX_M) / 2.0 for side in scale]
        elif geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            kind = 'hull' if geom.group == COLLISION_GROUP else 'mesh'
            geom.meshname = _mesh_asset(spec, cache, kind, spec.mesh(geom.meshname).file, scale)


def add_mesh_geoms(
    spec: mujoco.MjSpec,
    cache: AssetCache,
    body: mujoco.MjsBody,
    mesh_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Give body the visual, ray-visible geoms of mesh_path plus its convex pieces as colliders."""
    add_part_geoms(spec, cache, body, mesh_parts(mesh_path), scale)
    prepare_hulls([mesh_path])
    add_hull_geoms(spec, cache, body, collision_hulls(mesh_path), scale)


def add_mesh_body(
    spec: mujoco.MjSpec,
    cache: AssetCache,
    name: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    mesh_path: str,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> mujoco.MjsBody:
    """Add a static body that looks like mesh_path and collides as its convex decomposition."""
    body = spec.worldbody.add_body()
    body.name = name
    body.pos = list(pos)
    body.quat = list(quat)
    add_mesh_geoms(spec, cache, body, mesh_path, scale)
    return body


__all__ = [
    'COLLISION_GROUP',
    'TILE_M',
    'MeshPart',
    'PartGroup',
    'add_box_body',
    'add_hull_geoms',
    'add_mesh_body',
    'add_mesh_geoms',
    'add_part_geoms',
    'add_plane_body',
    'cache_entry',
    'cached_parts',
    'collision_hulls',
    'ensure_material',
    'ensure_rgba_material',
    'ensure_texture_material',
    'mesh_parts',
    'obj_albedo',
    'part_material',
    'prepare_hulls',
    'scale_body',
]
