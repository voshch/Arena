from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path

import attrs
import numpy as np
import yaml
from scipy.io import wavfile
from scipy.signal import resample_poly

from task_generator.auditory.octave_bands import (
    calculate_octave_band_levels_db,
)


@attrs.frozen
class AcousticSample:
    """Lightweight WAV metadata retained by the catalog at startup."""

    sample_id: str
    path: Path
    normalization_dbfs: float
    tags: frozenset[str]
    octave_band_levels_db: dict[int, float] | str | None


@attrs.frozen
class CachedSample:
    sample_id: str
    path: Path
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration_sec: float
    normalization_dbfs: float
    tags: frozenset[str]
    octave_band_levels_db: dict[int, float]


@attrs.frozen
class AcousticAsset:
    asset_id: str
    category: str
    semantic_tags: frozenset[str]
    reference_level_db: float
    reference_distance_m: float
    loop: bool
    variants: tuple[AcousticSample, ...]
    playback_gain_db: float = 0.0


class AcousticAssetCatalog:
    """Asset metadata plus a thread-safe, decode-on-first-use sample cache."""

    def __init__(
        self,
        config_path: Path | str,
        sound_dir: Path | str,
        *,
        output_sample_rate: int = 44100,
        output_channels: int = 2,
    ) -> None:
        config_path = Path(config_path)
        sound_dir = Path(sound_dir)

        if not config_path.is_file():
            raise FileNotFoundError(f"acoustic asset catalog does not exist: {config_path}")
        if not sound_dir.is_dir():
            raise FileNotFoundError(f"acoustic sound directory does not exist: {sound_dir}")

        self._target_rate = int(output_sample_rate)
        self._target_channels = int(output_channels)
        self._cache: dict[str, CachedSample] = {}
        self._cache_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0

        raw = yaml.safe_load(config_path.read_text())
        self._assets: dict[str, AcousticAsset] = {}
        known_sample_ids: set[str] = set()

        for asset_id, entry in raw.get("assets", {}).items():
            normalization_dbfs = float(entry.get("normalization_dbfs", -6.0))
            variant_entries = tuple(entry.get("variants", []))
            variants: list[AcousticSample] = []

            for variant in variant_entries:
                sample_id = str(variant["sample_id"])
                if sample_id in known_sample_ids:
                    raise ValueError(f"duplicate acoustic sample_id={sample_id!r}")
                known_sample_ids.add(sample_id)

                path = sound_dir / str(variant["file"])
                if not path.is_file():
                    raise FileNotFoundError(f"acoustic asset {asset_id!r} references missing WAV file: {path}")

                raw_bands = variant.get("octave_band_levels_db", "auto")
                bands: dict[int, float] | str | None
                if isinstance(raw_bands, dict):
                    bands = {int(frequency): float(level) for frequency, level in raw_bands.items()}
                else:
                    bands = raw_bands

                variants.append(
                    AcousticSample(
                        sample_id=sample_id,
                        path=path,
                        normalization_dbfs=normalization_dbfs,
                        tags=frozenset(map(str, variant.get("tags", []))),
                        octave_band_levels_db=bands,
                    )
                )

            if not variants:
                raise ValueError(f"asset {asset_id!r} has no variants")

            self._assets[asset_id] = AcousticAsset(
                asset_id=asset_id,
                category=str(entry["category"]),
                semantic_tags=frozenset(map(str, entry.get("semantic_tags", []))),
                reference_level_db=float(entry["reference_level_db"]),
                reference_distance_m=float(entry.get("reference_distance_m", 1.0)),
                playback_gain_db=float(entry.get("playback_gain_db", 0.0)),
                loop=bool(entry.get("loop", False)),
                variants=tuple(variants),
            )

    @property
    def cached_samples(self) -> int:
        with self._cache_lock:
            return len(self._cache)

    @property
    def cache_hits(self) -> int:
        with self._cache_lock:
            return self._cache_hits

    @property
    def cache_misses(self) -> int:
        with self._cache_lock:
            return self._cache_misses

    def get(self, asset_id: str) -> AcousticAsset | None:
        return self._assets.get(asset_id)

    def require(self, asset_id: str) -> AcousticAsset:
        asset = self.get(asset_id)
        if asset is None:
            raise KeyError(f"unknown acoustic asset_id={asset_id!r}")
        return asset

    def select(
        self,
        asset_id: str,
        *,
        episode_seed: int,
        agent_id: int,
        occurrence: int,
        required_tags: frozenset[str] = frozenset(),
    ) -> tuple[AcousticAsset, AcousticSample] | None:
        """Select metadata deterministically without reading the WAV file."""
        asset = self.get(asset_id)
        if asset is None:
            return None

        candidates = tuple(sample for sample in asset.variants if required_tags.issubset(sample.tags)) or asset.variants

        key = f"{episode_seed}:{agent_id}:{asset_id}:{occurrence}"
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % len(candidates)
        return asset, candidates[index]

    def load(self, sample: AcousticSample) -> CachedSample:
        """Decode a selected sample once and return the cached result."""
        # Decoding is serialized deliberately. Playback calls this method from
        # one worker, and the lock also prevents accidental duplicate decoding
        # if another caller uses the catalog concurrently.
        with self._cache_lock:
            cached = self._cache.get(sample.sample_id)
            if cached is not None:
                self._cache_hits += 1
                return cached

            decoded = self._decode_sample(
                sample_id=sample.sample_id,
                path=sample.path,
                tags=sample.tags,
                target_rate=self._target_rate,
                target_channels=self._target_channels,
                normalization_dbfs=sample.normalization_dbfs,
                octave_band_levels_db=sample.octave_band_levels_db,
            )
            self._cache[sample.sample_id] = decoded
            self._cache_misses += 1
            return decoded

    def load_many(
        self,
        samples: tuple[AcousticSample, ...],
    ) -> tuple[CachedSample, ...]:
        return tuple(self.load(sample) for sample in samples)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0

    @classmethod
    def _decode_sample(
        cls,
        *,
        sample_id: str,
        path: Path,
        tags: frozenset[str],
        target_rate: int,
        target_channels: int,
        normalization_dbfs: float,
        octave_band_levels_db: dict[int, float] | str | None = "auto",
    ) -> CachedSample:
        source_rate, data = wavfile.read(path)
        samples = cls._to_float32(data)

        if samples.ndim == 1:
            samples = samples[:, None]

        if samples.shape[1] == 1 and target_channels == 2:
            samples = np.repeat(samples, 2, axis=1)
        elif samples.shape[1] > target_channels:
            samples = samples[:, :target_channels]

        if source_rate != target_rate:
            divisor = math.gcd(source_rate, target_rate)
            samples = resample_poly(
                samples,
                target_rate // divisor,
                source_rate // divisor,
                axis=0,
            ).astype(np.float32)

        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        target_peak = 10.0 ** (normalization_dbfs / 20.0)
        if peak > 0.0:
            samples *= target_peak / peak

        samples = np.ascontiguousarray(samples, dtype=np.float32)
        if samples.size == 0 or len(samples) == 0:
            raise ValueError(f"empty WAV file: {path}")

        if octave_band_levels_db in (None, "auto"):
            measured_bands = calculate_octave_band_levels_db(
                samples,
                target_rate,
            )
        else:
            measured_bands = {int(frequency): float(level) for frequency, level in octave_band_levels_db.items()}

        return CachedSample(
            sample_id=sample_id,
            path=path,
            samples=samples,
            sample_rate=target_rate,
            channels=samples.shape[1],
            duration_sec=len(samples) / target_rate,
            normalization_dbfs=normalization_dbfs,
            tags=tags,
            octave_band_levels_db=measured_bands,
        )

    @staticmethod
    def _to_float32(data: np.ndarray) -> np.ndarray:
        if np.issubdtype(data.dtype, np.floating):
            return data.astype(np.float32)
        if data.dtype == np.uint8:
            return (data.astype(np.float32) - 128.0) / 128.0

        info = np.iinfo(data.dtype)
        scale = float(max(abs(info.min), info.max))
        return data.astype(np.float32) / scale
