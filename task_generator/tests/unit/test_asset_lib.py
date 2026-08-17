from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.io import wavfile

from task_generator.auditory.asset_lib import AcousticAssetCatalog, AcousticSample


def _catalog(tmp_path: Path) -> AcousticAssetCatalog:
    sound_dir = tmp_path / "sounds"
    sound_dir.mkdir()
    wavfile.write(
        sound_dir / "step.wav",
        22_050,
        np.asarray([0, 1000, -1000, 0], dtype=np.int16),
    )
    config = {
        "assets": {
            "footstep": {
                "category": "footstep",
                "semantic_tags": ["human"],
                "reference_level_db": 45.0,
                "normalization_dbfs": -6.0,
                "variants": [
                    {
                        "sample_id": "step_1",
                        "file": "step.wav",
                        "tags": ["default"],
                        "octave_band_levels_db": {
                            125: -10.0,
                            1000: -3.0,
                        },
                    }
                ],
            }
        }
    }
    config_path = tmp_path / "assets.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return AcousticAssetCatalog(config_path, sound_dir)


def test_catalog_select_is_metadata_only_and_load_is_cached(tmp_path):
    catalog = _catalog(tmp_path)

    selected = catalog.select(
        "footstep",
        episode_seed=1,
        agent_id=2,
        occurrence=3,
    )
    assert selected is not None
    _, sample_spec = selected

    assert catalog.cached_samples == 0
    assert isinstance(sample_spec, AcousticSample)

    first = catalog.load(sample_spec)
    second = catalog.load(sample_spec)

    assert first is second
    assert first.sample_rate == 44_100
    assert first.channels == 2
    assert first.octave_band_levels_db == {125: -10.0, 1000: -3.0}
    assert catalog.cached_samples == 1
    assert catalog.cache_misses == 1
    assert catalog.cache_hits == 1


def test_catalog_still_validates_missing_files_at_startup(tmp_path):
    config_path = tmp_path / "assets.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "assets": {
                    "footstep": {
                        "category": "footstep",
                        "reference_level_db": 45.0,
                        "variants": [
                            {
                                "sample_id": "missing",
                                "file": "missing.wav",
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    sound_dir = tmp_path / "sounds"
    sound_dir.mkdir()

    try:
        AcousticAssetCatalog(config_path, sound_dir)
    except FileNotFoundError as exc:
        assert "missing.wav" in str(exc)
    else:
        raise AssertionError("missing WAV was not rejected")


def test_material_catalog_surface_damping_uses_profiles(tmp_path):
    catalog_path = tmp_path / "acoustic_materials.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "octave_bands_hz": [125, 250, 500, 1000, 2000, 4000],
                "defaults": {
                    "canonical_name": "default",
                    "absorption": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
                    "transmission_loss_db": [12.0, 12.0, 12.0, 12.0, 12.0, 12.0],
                    "scattering": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "materials": {
                    "Concrete_Smooth": {
                        "canonical_name": "concrete_floor_smooth",
                        "absorption": [0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
                        "transmission_loss_db": [18.0, 18.0, 18.0, 18.0, 18.0, 18.0],
                        "scattering": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                    "Walnut_Planks": {
                        "canonical_name": "walnut_planks",
                        "absorption": [0.14, 0.12, 0.10, 0.09, 0.08, 0.07],
                        "transmission_loss_db": [14.0, 14.0, 14.0, 14.0, 14.0, 14.0],
                        "scattering": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = __import__(
        "task_generator.auditory.material_catalog",
        fromlist=["AcousticMaterialCatalog"],
    ).AcousticMaterialCatalog(catalog_path)

    concrete_floor = catalog.surface_damping_db("Concrete_Smooth", "floor")
    walnut_floor = catalog.surface_damping_db("Walnut_Planks", "floor")
    concrete_wall = catalog.surface_damping_db("Concrete_Smooth", "wall")

    assert 0.0 < concrete_floor <= 6.0
    assert walnut_floor > concrete_floor
    assert concrete_wall > concrete_floor
    assert concrete_wall > 0.0
