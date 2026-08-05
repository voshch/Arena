from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import traceback
import typing
from pathlib import Path

import attrs

from arena_simulation_setup import ARENA_ASSETS_DIR
from arena_simulation_setup.tree import (
    DynamicPaths,
    ModifiersDomainAssetIdentifier,
    NetResolver,
)
from arena_simulation_setup.utils.material import ImgUtil, MdlUtil


@attrs.define(eq=False, hash=False)
class MaterialIdentifier(ModifiersDomainAssetIdentifier["Material"]):
    """Represents an identifier referencing a material asset."""

    _asset_type = 'Material'

    # Baking a tint re-encodes every diffuse texture, so results are cached on disk keyed by
    # source contents. Level 1 keeps the encode ~5x cheaper than zlib's default for ~13% more
    # bytes, which never leave this cache.
    _CACHE_DIR: typing.ClassVar[Path] = ARENA_ASSETS_DIR / '.cache' / 'Material'
    _COMPRESS_LEVEL: typing.ClassVar[int] = 1

    @classmethod
    def _tint_key(cls, basepath: Path, tint: str) -> str:
        digest = hashlib.sha256(tint.encode())
        for file in sorted(basepath.rglob('*')):
            if file.is_file():
                stat = file.stat()
                digest.update(f'{file.relative_to(basepath)}:{stat.st_mtime_ns}:{stat.st_size}'.encode())
        return digest.hexdigest()[:16]

    @classmethod
    def _apply_tint(cls, basepath: Path, material: Material, tint: str) -> Material:

        relpath = Path(material.path).relative_to(basepath)
        cachedir = cls._CACHE_DIR / cls._tint_key(basepath, tint)

        if not cachedir.is_dir():
            cls._CACHE_DIR.mkdir(parents=True, exist_ok=True)
            stagedir = Path(tempfile.mkdtemp(prefix='material_', dir=cls._CACHE_DIR))
            shutil.copytree(basepath, stagedir, dirs_exist_ok=True)

            for texture_path in MdlUtil(stagedir / relpath).diffuse_texture_paths:
                try:
                    if texture_path.exists() and texture_path.is_relative_to(stagedir):
                        tinted_img = ImgUtil.tint(texture_path, tint)
                        tinted_img.save(texture_path, compress_level=cls._COMPRESS_LEVEL)
                except Exception as e:
                    logging.error(f'Failed to tint texture {texture_path}: {e}\n{traceback.format_exc()}')

            try:
                stagedir.rename(cachedir)
            except OSError:
                # Lost the race against a concurrent bake of the same key.
                shutil.rmtree(stagedir, ignore_errors=True)

        return attrs.evolve(
            material,
            path=str((cachedir / relpath).resolve()),
        )

    def load(self, path: Path, /, **kwargs: object) -> Material:
        del kwargs  # unused
        mat = Material(
            name=self.name,
            path=os.path.join(path, f'{self.name}.mdl'),
        )

        if (tint := self.modifiers.get('tint')) is not None:
            mat = self._apply_tint(path, mat, tint)
        return mat


MaterialIdentifier.use(*DynamicPaths.as_resolvers(MaterialIdentifier))
MaterialIdentifier.use(*NetResolver.all(MaterialIdentifier))


@attrs.define
class Material:
    path: str
    name: str

    __DEFAULT: typing.ClassVar[MaterialIdentifier] = MaterialIdentifier("Marble")
    __DEFAULTS: typing.ClassVar[dict[str, MaterialIdentifier]] = {
        'wall': MaterialIdentifier('Marble'),
        'floor': MaterialIdentifier('Porcelain_Tile_4'),
        'door': MaterialIdentifier('Aluminum_Anodized'),
        'ceiling': MaterialIdentifier('Concrete_Smooth'),
    }

    @classmethod
    def default(cls, context: typing.Literal['floor', 'wall', 'door', 'ceiling'] | str = '') -> MaterialIdentifier:
        return cls.__DEFAULTS.get(context, cls.__DEFAULT)

    def asdict(self) -> dict:
        return attrs.asdict(self)
