from __future__ import annotations

import wave
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.tree.assets.sound_catalog import STEMS, AgentKind, Kind, SoundLibrary, parse_manifest

KINDS_FILE = Path(__file__).resolve().parents[2] / "configs" / "sounds" / "kinds.yaml"


def _write_wav(path: Path, rate: int, frames: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(2 * frames))


def _chime_manifest(kinds: Mapping[str, object] | None) -> dict[str, object]:
    manifest: dict[str, object] = {
        "version": 2,
        "kind": "chime",
        "level_db": 55.0,
        "normalize_dbfs": -18.0,
        "variants": [{"id": "chime_01", "file": "chime.wav"}],
    }
    if kinds is not None:
        manifest["kinds"] = dict(kinds)
    return manifest


def _write_chime(world: Path, kinds: Mapping[str, object] | None) -> Path:
    directory = world / "assets" / "Common" / "Sound" / "tmp_chime"
    directory.mkdir(parents=True)
    _write_wav(directory / "chime.wav", 16_000, 3200)
    (directory / "tmp_chime.yaml").write_text(yaml.safe_dump(_chime_manifest(kinds)), encoding="utf-8")
    return directory


@pytest.fixture
def library() -> Iterator[SoundLibrary]:
    library = SoundLibrary([KINDS_FILE])
    try:
        yield library
    finally:
        library.use_world(None)


def test_kinds_file_rows_carry_their_stem_agent_and_default_asset(library: SoundLibrary) -> None:
    footstep = library.kind("footstep")
    motor = library.kind("motor")

    assert footstep.stem == "pedestrian"
    assert footstep.agent is AgentKind.PEDESTRIAN
    assert footstep.default_asset == "footstep"
    assert motor.stem == "motor"
    assert library.kinds_of(AgentKind.ROBOT) == ("motor",)
    assert {kind.stem for kind in library.kinds().values()} <= set(STEMS)


def test_world_manifest_kinds_are_available_before_any_asset_loads(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    _write_chime(world, {"chime": {"stem": "ambient", "agent": "environment", "color": [0.1, 0.2, 0.3], "default_asset": "tmp_chime"}})

    assert "chime" not in library.kinds()
    library.use_world(world)
    chime = library.kind("chime")

    assert chime == Kind(name="chime", stem="ambient", agent="environment", color=(0.1, 0.2, 0.3), default_asset="tmp_chime")
    assert library.kinds_of(AgentKind.ENVIRONMENT)[-1] == "chime"
    assert library.default_asset("chime").kind == "chime"


def test_world_manifest_kinds_go_away_on_a_world_switch(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    _write_chime(world, {"chime": {"stem": "ambient"}})
    library.use_world(world)
    assert "chime" in library.kinds()

    library.use_world(tmp_path / "empty_world")

    assert "chime" not in library.kinds()
    with pytest.raises(KeyError, match="unknown sound kind 'chime'"):
        library.kind("chime")


def test_world_with_conflicting_kinds_is_refused_and_the_previous_world_stays(tmp_path: Path, library: SoundLibrary) -> None:
    from arena_simulation_setup.tree import DynamicPaths

    good = tmp_path / "good_world"
    _write_chime(good, {"chime": {"stem": "ambient"}})
    library.use_world(good)
    bad = tmp_path / "bad_world"
    _write_chime(bad, {"footstep": {"stem": "ambient"}})

    with pytest.raises(ValueError, match="redefines kind 'footstep'"):
        library.use_world(bad)

    assert DynamicPaths.WORLD.path == good
    assert "chime" in library.kinds()
    library.use_world(tmp_path / "empty_world")
    assert "chime" not in library.kinds()
    library.use_world(good)
    assert "chime" in library.kinds()


def test_named_world_with_a_level_filter_selects_the_whole_world(library: SoundLibrary) -> None:
    from arena_simulation_setup.tree import DynamicPaths
    from arena_simulation_setup.tree.World import WorldIdentifier

    library.use_world_named("map_empty[0]")

    assert DynamicPaths.WORLD.path == Path(WorldIdentifier("map_empty").resolve_sync().path)


def test_asset_of_an_undeclared_kind_is_rejected(tmp_path: Path, library: SoundLibrary) -> None:
    world = tmp_path / "world"
    _write_chime(world, None)
    library.use_world(world)

    with pytest.raises(ValueError, match="unknown kind 'chime'"):
        library.asset("tmp_chime")


def test_manifest_redefining_a_table_kind_is_rejected(tmp_path: Path) -> None:
    kinds_file = tmp_path / "sounds.yaml"
    kinds_file.write_text(yaml.safe_dump({"version": 2, "kinds": {"chime": {"stem": "ambient"}}}), encoding="utf-8")
    world = tmp_path / "world"
    _write_chime(world, {"chime": {"stem": "motor"}})
    library = SoundLibrary([kinds_file])

    try:
        with pytest.raises(ValueError, match="redefines kind 'chime'"):
            library.use_world(world)
    finally:
        library.use_world(tmp_path / "empty_world")
        library.use_world(None)


def test_kind_without_a_stem_is_rejected(tmp_path: Path) -> None:
    directory = _write_chime(tmp_path, None)

    with pytest.raises(ValueError, match="kind 'chime'"):
        parse_manifest("tmp_chime", directory, _chime_manifest({"chime": {"agent": "environment"}}))


@pytest.mark.parametrize("stem", ["music", "", "Ambient"])
def test_kind_with_a_stem_outside_the_renderer_stems_is_rejected(tmp_path: Path, stem: str) -> None:
    kinds_file = tmp_path / "sounds.yaml"
    kinds_file.write_text(yaml.safe_dump({"version": 2, "kinds": {"chime": {"stem": stem}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="kind 'chime'"):
        SoundLibrary([kinds_file])


@pytest.mark.parametrize("stem", STEMS)
def test_kind_accepts_each_renderer_stem(stem: str) -> None:
    assert Kind.from_dict("chime", {"stem": stem}, where="test").stem == stem
