"""Shared base of the importers that turn one wav into a Common/Sound item through the sound database builder."""

from __future__ import annotations

import abc
import shutil
import typing
from functools import cached_property
from pathlib import Path

import yaml

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode
from arena_assets.impl.build.SoundDatabaseBuilder import MANIFEST_VERSION
from arena_assets.importers import Importer

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.assets.sound_catalog import Kind, SoundAsset, SoundLibrary

DOMAIN = "Common"
ASSET_TYPE = "Sound"
PEAK_DBFS = -6.0
ENVIRONMENT_PEAK_DBFS = -9.0


def peak_dbfs(kind: Kind) -> float:
    """Peak the native assets of a kind were normalized to before their active level was taken."""
    from arena_simulation_setup.tree.assets.sound_catalog import AgentKind

    return ENVIRONMENT_PEAK_DBFS if kind.agent is AgentKind.ENVIRONMENT else PEAK_DBFS


def peak_normalized_level_dbfs(path: Path, peak: float) -> float:
    """Active level of the first channel once the peak over all channels is scaled to peak dBFS."""
    import numpy as np
    from scipy.io import wavfile

    from arena_simulation_setup.tree.assets.sound_level import active_rms, dbfs_from_rms

    rate, data = wavfile.read(path)
    samples = data.astype(np.float64) - (128.0 if data.dtype == np.uint8 else 0.0)
    if samples.ndim == 1:
        samples = samples[:, None]
    top = float(np.max(np.abs(samples))) if samples.size else 0.0
    if top == 0.0:
        raise ValueError(f"{path}: silent or empty wav")
    return dbfs_from_rms(active_rms(samples[:, 0] * (10.0 ** (peak / 20.0) / top), rate))


def normalize_dbfs(path: Path, peak: float) -> float:
    """The manifest normalize_dbfs of a wav, its peak-normalized active level to the nearest 0.5 dB."""
    return round(2.0 * peak_normalized_level_dbfs(path, peak)) / 2.0


def stage(name: str, kind: str, desc: str, tags: list[str], reference: SoundAsset, normalize: float, src: Path, dataset: Path) -> Path:
    """Lay one wav out as sound builder input under dataset/Common/Sound/<name> and return that directory."""
    source = dataset / DOMAIN / ASSET_TYPE / name
    if dataset.exists():
        shutil.rmtree(dataset)
    source.mkdir(parents=True)
    wav = f"{name}.wav"
    shutil.copyfile(src, source / wav)
    manifest = {
        "version": MANIFEST_VERSION,
        "kind": kind,
        "desc": desc,
        "tags": tags,
        "level_db": reference.level_db,
        "reference_distance_m": reference.reference_distance_m,
        "normalize_dbfs": normalize,
        "loop": False,
        "model": "wav",
        **({"surface": reference.surface} if reference.surface else {}),
        "variants": [{"id": f"{name}_01", "file": wav}],
    }
    with open(source / f"{name}.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return source


class SoundImporter(Importer):
    @abc.abstractmethod
    def asset_name(self, item: str) -> str: ...

    @cached_property
    def library(self) -> SoundLibrary:
        from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary

        return SoundLibrary.default()

    def reference(self, kind: str) -> SoundAsset | None:
        """The kind's default asset, whose emission level every imported item of that kind takes. None for a kind without one."""
        return self.library.default_asset(kind) if self.library.kind(kind).default_asset else None

    def relpath(self, item: str) -> Path:
        return Path(DOMAIN) / ASSET_TYPE / self.asset_name(item)

    def build(self, item: str, kind: str, reference: SoundAsset, desc: str, tags: list[str], src: Path, out: Path) -> str:
        """Stage the wav with the reference's emission, build it into out like a bundled sound and return its relpath."""
        normalize = normalize_dbfs(src, peak_dbfs(self.library.kind(kind)))
        relpath = self.relpath(item)
        target = out / relpath
        if target.exists():
            shutil.rmtree(target)
        dataset = self.cache / "stage" / item
        stage(self.asset_name(item), kind, desc, tags, reference, normalize, src, dataset)
        builder = DatabaseBuilder.Builder(AssetType.SOUND)(input_path=dataset / DOMAIN, output_path=out / DOMAIN, overwrite=OverwriteMode.OVERWRITE)
        try:
            builder.build()
        finally:
            shutil.rmtree(dataset)
        if not (target / ANNOTATION_NAME).is_file():
            raise RuntimeError(f"sound build failed for {item}, the build log above names the step")
        return str(relpath)
