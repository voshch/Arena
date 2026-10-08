from __future__ import annotations

import wave
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary, Variant, parse_manifest, pattern_matches

KINDS_FILE = Path(__file__).resolve().parents[2] / "configs" / "sounds" / "kinds.yaml"


def _write_wav(path: Path, rate: int, frames: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(2 * frames))


def _write_asset(world: Path, name: str, manifest: Mapping[str, object], wavs: Mapping[str, tuple[int, int]]) -> Path:
    directory = world / "assets" / "Common" / "Sound" / name
    directory.mkdir(parents=True)
    for file, (rate, frames) in wavs.items():
        _write_wav(directory / file, rate, frames)
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(dict(manifest)), encoding="utf-8")
    return directory


def _step_manifest(**overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "version": 2,
        "kind": "footstep",
        "level_db": 45.0,
        "normalize_dbfs": -6.0,
        "variants": [
            {"id": "step_walnut", "file": "walnut.wav", "match": {"floor": ["walnut"]}, "tags": ["walnut_planks"]},
            {"id": "step_default", "file": "step.wav", "default": True, "tags": ["default"]},
        ],
    }
    manifest.update(overrides)
    return manifest


def _step_wavs() -> dict[str, tuple[int, int]]:
    return {
        "step.wav": (22_050, 4),
        "walnut.wav": (22_050, 4),
    }


@pytest.fixture
def library() -> Iterator[SoundLibrary]:
    library = SoundLibrary([KINDS_FILE])
    try:
        yield library
    finally:
        library.use_world(None)


def test_select_returns_the_matching_variant_with_its_metadata(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    _write_asset(world, "tmp_step", _step_manifest(), _step_wavs())
    library.use_world(world)
    asset = library.asset("tmp_step")

    variant = asset.select(context={"floor": "Walnut_Planks"}, seed=3)

    assert isinstance(variant, Variant)
    assert variant.id == "step_walnut"
    assert variant.path == world / "assets" / "Common" / "Sound" / "tmp_step" / "walnut.wav"
    assert library.variant("tmp_step", "step_walnut") is variant


def test_known_asset_ids_hold_the_world_assets_and_every_kind_default(tmp_path: Path) -> None:
    kinds = tmp_path / "kinds.yaml"
    kinds.write_text(yaml.safe_dump({"version": 2, "kinds": {"chime": {"stem": "ambient", "default_asset": "tmp_chime_nowhere_on_disk"}}}), encoding="utf-8")
    world = tmp_path / "world"
    _write_asset(world, "tmp_step", _step_manifest(kind="chime"), _step_wavs())
    library = SoundLibrary([kinds])
    try:
        library.use_world(world)
        known = library.known_asset_ids()
    finally:
        library.use_world(None)

    assert "tmp_step" in known
    assert "tmp_chime_nowhere_on_disk" in known
    assert list(known) == sorted(set(known))


def test_unknown_floor_selects_the_default_variant(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    _write_asset(world, "tmp_step", _step_manifest(), _step_wavs())
    library.use_world(world)

    selected = library.asset("tmp_step").select(context={"floor": "unknown_floor"}, seed=9)

    assert selected.id == "step_default"


def test_missing_variant_wav_is_rejected_when_the_asset_loads(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    manifest = _step_manifest(variants=[{"id": "missing", "file": "missing.wav"}])
    _write_asset(world, "tmp_broken", manifest, {})
    library.use_world(world)

    with pytest.raises(FileNotFoundError, match=r"missing\.wav"):
        library.asset("tmp_broken")


def test_desc_and_tags_are_parsed_onto_the_asset(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    _write_asset(world, "tmp_step", _step_manifest(desc="Footsteps on wood.", tags=["footstep", "wood"]), _step_wavs())
    library.use_world(world)

    asset = library.asset("tmp_step")

    assert asset.desc == "Footsteps on wood."
    assert asset.tags == ("footstep", "wood")
    assert asset.variant("step_walnut").tags == ("walnut_planks",)


def test_desc_that_is_not_a_string_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "tmp_step"
    directory.mkdir()

    with pytest.raises(ValueError, match="desc must be a string"):
        parse_manifest("tmp_step", directory, _step_manifest(desc=["not", "text"], variants=[{"id": "synth", "model": "drivetrain"}]))


@pytest.mark.parametrize(
    ("pattern", "value", "matches"),
    [
        ("oak", "Common/Material/Oak_Planks", True),
        ("oak", "Cloak_Room_Floor", False),
        ("ceramic_*", "Common/Material/Ceramic_Tile_6", True),
        ("ceramic_?", "Ceramic_Tile_6", False),
        ("oak", "", False),
    ],
)
def test_pattern_matches_whole_words_or_globs_on_the_leaf(pattern: str, value: str, matches: bool) -> None:
    assert pattern_matches(pattern, value) is matches
