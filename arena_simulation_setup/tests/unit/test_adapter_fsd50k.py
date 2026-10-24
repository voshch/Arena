from __future__ import annotations

import csv
import shutil
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml
from arena_assets.impl import ANNOTATION_NAME, DATABASE_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder
from arena_assets.impl.build.SoundDatabaseBuilder import SoundAnnotation, SoundDatabaseBuilder
from arena_assets.impl.listall import list_database
from arena_assets.impl.query import query_database
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.providers import HFHub
from arena_assets.semantic import CURRENT_ASA
from arena_assets.semantic.lint import lint_file
from arena_assets.semantic.taxonomy import load_spec
from arena_assets.utils.Database import Database
from arena_simulation_setup.tree import DynamicPathResolver
from arena_simulation_setup.tree.assets.Sound import SoundIdentifier
from arena_simulation_setup.tree.assets.sound_catalog import WAV_MODELS, AgentKind, SoundLibrary, canonical_asset_id, kinds_file, local_sound_dirs, parse_manifest, selection_seed
from scipy.io import wavfile

from arena_simulation_setup.adapters.fsd50k import FSD50K, KINDS_FILE, REPO, asset_name, asset_tags, kind_of, load_kinds, spdx_license
from arena_simulation_setup.adapters.sound import ENVIRONMENT_PEAK_DBFS, PEAK_DBFS, normalize_dbfs, peak_dbfs, peak_normalized_level_dbfs, stage

PACKAGE = Path(__file__).resolve().parents[2]
FIXTURE = PACKAGE / "tests" / "fixtures" / "fsd50k"
CLIPS = {"386695": "footstep", "182351": "alarm", "377266": "speech"}
REFERENCES = {"footstep": "footstep", "alarm": "alarm_loop", "speech": "greeting"}
EMISSION = {"footstep": (45.0, "floor"), "alarm": (88.0, ""), "speech": (60.0, "")}
RELPATH = Path("Common") / "Sound"
NATIVE_TOLERANCE_DB = 0.4


@pytest.fixture
def importer(tmp_path: Path) -> Iterator[FSD50K]:
    cache = tmp_path / "cache"
    shutil.copytree(FIXTURE, cache, ignore=shutil.ignore_patterns("ATTRIBUTION.yaml"))
    yield FSD50K(cache=cache, provider=HFHub(REPO))


@pytest.fixture
def library(default_sounds: SoundLibrary) -> Iterator[SoundLibrary]:
    library = SoundLibrary([kinds_file()])
    try:
        yield library
    finally:
        library.use_world(None)


@pytest.fixture(scope="session")
def defaults(default_sounds: SoundLibrary) -> dict[str, Path]:
    """Directory of every kind's default asset, as the default sounds resolve."""
    names = sorted({kind.default_asset for kind in default_sounds.kinds().values() if kind.default_asset})
    return {name: Path(SoundIdentifier.parse(name).resolve_sync().path) for name in names}


@pytest.fixture(scope="module")
def native(defaults: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = tmp_path_factory.mktemp("defaults") / "Common"
    for name, directory in defaults.items():
        shutil.copytree(directory, source / "Sound" / name, ignore=shutil.ignore_patterns(ANNOTATION_NAME, ATTRIBUTION_FILE, ".*"))
    out = tmp_path_factory.mktemp("native") / "Common"
    DatabaseBuilder.Builder(AssetType.SOUND)(input_path=source, output_path=out).build()
    return out


def _yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _files(directory: Path) -> list[str]:
    return sorted(path.name.replace(directory.name, "NAME") for path in directory.iterdir())


def test_peak_normalized_active_level_reproduces_every_default_normalize_dbfs(library: SoundLibrary, defaults: dict[str, Path]) -> None:
    peaks = {}
    for name, directory in defaults.items():
        manifest = _yaml(directory / f"{name}.yaml")
        peaks[name] = peak = peak_dbfs(library.kind(manifest["kind"]))
        levels = [peak_normalized_level_dbfs(directory / variant["file"], peak) for variant in manifest["variants"] if variant.get("file")]
        assert min(levels) == pytest.approx(manifest["normalize_dbfs"], abs=NATIVE_TOLERANCE_DB), name
        if len(levels) == 1:
            assert normalize_dbfs(directory / manifest["variants"][-1]["file"], peak) == manifest["normalize_dbfs"], name
    assert peaks == {"alarm_loop": ENVIRONMENT_PEAK_DBFS, "footstep": PEAK_DBFS, "greeting": PEAK_DBFS, "motor": PEAK_DBFS, "radio_loop": ENVIRONMENT_PEAK_DBFS}


def test_registry_loads_this_class() -> None:
    assert IMPORTERS.load("sound", "fsd50k") is FSD50K
    assert FSD50K.meta.kind == "sound"
    assert FSD50K.default_provider().url("clips/dev/1.wav").endswith("/Fhrozen/FSD50k/resolve/main/clips/dev/1.wav")


@pytest.mark.usefixtures("default_sounds")
def test_items_keep_cc0_and_cc_by_clips_of_a_kind_with_a_reference_asset(importer: FSD50K) -> None:
    assert list(importer.items()) == list(CLIPS)
    assert {item: importer.kind(item) for item in CLIPS} == CLIPS
    assert importer.clip("377266").license == "CC-BY-3.0"
    assert importer.clip("371717").license == "CC0-1.0"
    assert kind_of(importer.clip("371717").labels, importer.kinds) is None
    assert importer.clip("327432").license is None
    assert importer.clip("327432").split == "eval"
    with pytest.raises(KeyError, match="Freesound ids"):
        importer.clip("1")


@pytest.mark.usefixtures("default_sounds")
def test_a_kind_without_a_default_asset_is_not_importable(importer: FSD50K, tmp_path: Path) -> None:
    assert importer.clip("336390").license == "CC0-1.0"
    assert kind_of(importer.clip("336390").labels, importer.kinds) == "onset"
    assert importer.reference("onset") is None
    assert importer.reference("alarm").id == "alarm_loop"
    with pytest.raises(ValueError, match="kind 'onset', which has no default asset"):
        importer.kind("336390")
    with pytest.raises(ValueError, match="no label mapped"):
        importer.convert("371717", FIXTURE / "clips" / "dev" / "182351.wav", tmp_path / "out")


def test_spdx_license_from_freesound_urls() -> None:
    assert spdx_license("http://creativecommons.org/publicdomain/zero/1.0/") == "CC0-1.0"
    assert spdx_license("http://creativecommons.org/licenses/by/3.0/") == "CC-BY-3.0"
    assert spdx_license("https://creativecommons.org/licenses/by/4.0/") == "CC-BY-4.0"
    assert spdx_license("http://creativecommons.org/licenses/by-nc/3.0/") is None
    assert spdx_license("http://creativecommons.org/licenses/sampling+/1.0/") is None


def test_tags_are_the_origin_facet_then_labels_and_freesound_tags_as_bare_words(importer: FSD50K) -> None:
    assert asset_tags(importer.clip("336390")) == ["origin::fsd50k", "knock", "door", "domestic_sounds_and_home_sounds", "knocking", "knuckle_rap"]
    assert asset_tags(importer.clip("182351")) == ["origin::fsd50k", "alarm", "beep", "clock"]
    for item in ("386695", "377266"):
        assert all("::" not in tag for tag in asset_tags(importer.clip(item))[1:])


def test_stage_is_sound_builder_input_with_the_upstream_wav(importer: FSD50K, library: SoundLibrary, tmp_path: Path) -> None:
    src = importer.fetch("386695")
    source = stage(asset_name("386695"), "footstep", "climbing steps", asset_tags(importer.clip("386695")), library.default_asset("footstep"), -30.0, src, tmp_path / "dataset")
    assert source == tmp_path / "dataset" / RELPATH / "fsd50k_386695"
    assert sorted(path.name for path in source.iterdir()) == ["fsd50k_386695.wav", "fsd50k_386695.yaml"]
    assert (source / "fsd50k_386695.wav").read_bytes() == src.read_bytes()
    found = list(SoundDatabaseBuilder(input_path=tmp_path / "dataset" / "Common", output_path=tmp_path / "build").discover())
    assert [annotation.name for annotation in found] == ["fsd50k_386695"]
    assert Path(found[0].path) == source


def test_manifest_has_the_keys_and_emission_of_the_kind_reference(importer: FSD50K, library: SoundLibrary, defaults: dict[str, Path], tmp_path: Path) -> None:
    for item, kind in CLIPS.items():
        name = asset_name(item)
        target = tmp_path / "out" / importer.convert(item, importer.fetch(item), tmp_path / "out")
        manifest = _yaml(target / f"{name}.yaml")
        reference = _yaml(defaults[REFERENCES[kind]] / f"{REFERENCES[kind]}.yaml")
        assert list(manifest) == [key for key in ("version", "kind", "desc", "tags", "level_db", "reference_distance_m", "normalize_dbfs", "loop", "model", "surface", "variants") if key in reference]
        for key in ("version", "kind", "level_db", "reference_distance_m"):
            assert manifest[key] == reference[key]
        assert (manifest["level_db"], manifest.get("surface", "")) == EMISSION[kind]
        assert manifest["normalize_dbfs"] == normalize_dbfs(FIXTURE / "clips" / "dev" / f"{item}.wav", peak_dbfs(library.kind(kind)))
        assert manifest["loop"] is False
        assert manifest["model"] == "wav"
        assert manifest["variants"] == [{"id": f"{name}_01", "file": f"{name}.wav"}]
        assert manifest["tags"][0] == "origin::fsd50k"
        asset, fragment = parse_manifest(name, target, manifest)
        assert fragment == {}
        assert asset.variants[0].path == target / f"{name}.wav"
    assert _yaml(tmp_path / "out" / RELPATH / "fsd50k_386695" / "fsd50k_386695.yaml")["desc"] == "climbing steps"
    assert _yaml(tmp_path / "out" / RELPATH / "fsd50k_377266" / "fsd50k_377266.yaml")["desc"] == "Come"


def test_convert_output_matches_a_native_build_of_the_default_sounds(importer: FSD50K, native: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    spec = load_spec()
    for item, kind in CLIPS.items():
        name = asset_name(item)
        assert importer.convert(item, importer.fetch(item), out) == str(RELPATH / name)
        target = out / RELPATH / name
        reference = native / "Sound" / "greeting"
        assert _files(target) == _files(reference) == ["NAME.wav", "NAME.yaml", ANNOTATION_NAME]
        annotation = _yaml(target / ANNOTATION_NAME)
        native_annotation = _yaml(reference / ANNOTATION_NAME)
        assert {key: type(value) for key, value in annotation.items()} == {key: type(value) for key, value in native_annotation.items()}
        assert annotation["name"] == name
        assert annotation["path"] == f"Sound/{name}"
        assert annotation["asa"] == native_annotation["asa"] == CURRENT_ASA
        assert annotation["kind"] == kind
        assert annotation["variants"] == [f"{name}_01"]
        assert annotation["tags"][0] == "origin::fsd50k"
        assert annotation["tags"][-1] == f"kind::{kind}"
        assert native_annotation["tags"][-1] == "kind::speech"
        assert {finding.severity for finding in lint_file(target / ANNOTATION_NAME, spec)} == {finding.severity for finding in lint_file(reference / ANNOTATION_NAME, spec)} == {"info"}
        assert not (importer.cache / "stage" / item).exists()


def test_convert_indexes_the_item_like_a_native_sound(importer: FSD50K, native: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    for item in CLIPS:
        importer.convert(item, importer.fetch(item), out)
    listed = Database(out / "Common" / DATABASE_NAME).list_all("sound")
    assert sorted(listed["ids"]) == sorted(f"Sound/{asset_name(item)}" for item in CLIPS)
    native_listed = Database(native / DATABASE_NAME).list_all("sound")
    native_keys = set(native_listed["metadatas"][native_listed["ids"].index("Sound/greeting")])
    for metadata in listed["metadatas"]:
        assert set(metadata) == native_keys | {"origin"}
        assert metadata["origin"] == "fsd50k"
    by_name = {annotation.name: annotation for annotation in list_database(str(out / "Common"), AssetType.SOUND)}
    assert by_name["fsd50k_182351"] == SoundAnnotation(name="fsd50k_182351", path="Sound/fsd50k_182351", desc="Alarm clock beep", tags=["origin::fsd50k", "alarm", "beep", "clock", "kind::alarm"], asa=CURRENT_ASA, kind="alarm", loop=False, level_db=88.0, variants=["fsd50k_182351_01"])
    results = query_database(str(out / "Common"), AssetType.SOUND, "alarm clock", n=3, where={"kind": "alarm"})
    assert [annotation.path for annotation, _ in results] == ["Sound/fsd50k_182351"]
    results = query_database(str(out / "Common"), AssetType.SOUND, "steps", n=3, where={"origin": "fsd50k"})
    assert results[0][0].path == "Sound/fsd50k_386695"


@pytest.mark.usefixtures("default_sounds")
def test_reimport_replaces_the_item(importer: FSD50K, tmp_path: Path) -> None:
    first = importer.import_item("182351", tmp_path / "out")
    manifest = (first / "fsd50k_182351.yaml").read_text()
    annotation = (first / ANNOTATION_NAME).read_text()
    stale = first / "stale.txt"
    stale.write_text("x")
    assert importer.import_item("182351", tmp_path / "out") == first
    assert (first / "fsd50k_182351.yaml").read_text() == manifest
    assert (first / ANNOTATION_NAME).read_text() == annotation
    assert not stale.exists()
    assert Database(tmp_path / "out" / "Common" / DATABASE_NAME).list_all("sound")["ids"] == ["Sound/fsd50k_182351"]


@pytest.mark.usefixtures("default_sounds")
def test_relpath_matches_the_sound_identifier(importer: FSD50K, tmp_path: Path) -> None:
    relpath = Path(importer.convert("386695", importer.fetch("386695"), tmp_path / "out"))
    assert relpath == importer.relpath("386695") == SoundIdentifier(name="fsd50k_386695").relpath()
    assert SoundIdentifier.parse("fsd50k_386695").relpath() == relpath
    assert SoundIdentifier.from_relpath(relpath) == SoundIdentifier.parse("fsd50k_386695")
    assert canonical_asset_id(str(relpath)) == "fsd50k_386695"


def test_imported_clips_resolve_through_the_sound_library_with_the_upstream_audio(importer: FSD50K, library: SoundLibrary, tmp_path: Path) -> None:
    world = tmp_path / "world"
    targets = {item: importer.import_item(item, world / "assets") for item in CLIPS}
    library.use_world(world)
    listed = {identifier.shortname for identifier in SoundIdentifier.listall()}
    for item, kind in CLIPS.items():
        name = asset_name(item)
        assert targets[item] == world / "assets" / RELPATH / name
        assert f"Common/Sound/{name}" in listed
        assert local_sound_dirs()[name] == targets[item]
        hit = SoundIdentifier.parse(name).resolve_source_sync()
        assert isinstance(hit.resolver, DynamicPathResolver)
        assert hit.path == targets[item]
        assert SoundIdentifier.parse(name).resolve_sync().manifest["kind"] == kind

        asset = library.asset(f"Common/Sound/{name}")
        reference = library.default_asset(kind)
        assert asset is library.asset(name)
        assert asset.id == name
        assert asset.kind == kind
        assert (asset.level_db, asset.reference_distance_m, asset.surface) == (reference.level_db, reference.reference_distance_m, reference.surface)
        assert asset.normalize_dbfs == normalize_dbfs(FIXTURE / "clips" / "dev" / f"{item}.wav", peak_dbfs(library.kind(kind)))
        assert asset.normalize_dbfs == pytest.approx(peak_normalized_level_dbfs(targets[item] / f"{name}.wav", peak_dbfs(library.kind(kind))), abs=0.25)
        assert asset.loop is False
        assert asset.tags[0] == "origin::fsd50k"

        variant = asset.select(context={}, seed=selection_seed(name))
        assert variant.id == f"{name}_01"
        assert variant.model in WAV_MODELS
        assert variant.path == targets[item] / f"{name}.wav"
        assert library.variant(asset.id, variant.id).path.read_bytes() == (FIXTURE / "clips" / "dev" / f"{item}.wav").read_bytes()
    assert library.asset("fsd50k_182351").kind in library.kinds_of(AgentKind.ENVIRONMENT)


def test_emission_level_does_not_follow_the_recording_gain(importer: FSD50K, library: SoundLibrary, tmp_path: Path) -> None:
    clip = importer.cache / "clips" / "dev" / "182351.wav"
    world = tmp_path / "world"
    peak = peak_dbfs(library.kind("alarm"))
    target = importer.import_item("182351", world / "assets")
    loud = _yaml(target / "fsd50k_182351.yaml")
    loud_level = peak_normalized_level_dbfs(target / "fsd50k_182351.wav", peak)
    rate, data = wavfile.read(clip)
    wavfile.write(clip, rate, (data // 4).astype(np.int16))
    quiet = _yaml(importer.import_item("182351", world / "assets") / "fsd50k_182351.yaml")
    assert (target / "fsd50k_182351.wav").read_bytes() == clip.read_bytes()
    assert quiet == loud
    assert quiet["level_db"] == library.default_asset("alarm").level_db == 88.0
    assert peak_normalized_level_dbfs(target / "fsd50k_182351.wav", peak) == pytest.approx(loud_level, abs=0.1)


@pytest.mark.usefixtures("default_sounds")
def test_cc_by_attribution_stays_with_the_item_through_a_native_rebuild(importer: FSD50K, tmp_path: Path) -> None:
    target = importer.import_item("377266", tmp_path / "out")
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert attribution.source == "fsd50k"
    assert attribution.license == "CC-BY-3.0"
    assert attribution.authors == "johnnypanic"
    assert attribution.url == "https://freesound.org/s/377266/"
    assert "FSD50K" in attribution.citation
    assert read_attribution(importer.import_item("182351", tmp_path / "out") / ATTRIBUTION_FILE).license == "CC0-1.0"
    DatabaseBuilder.Builder(AssetType.SOUND)(input_path=tmp_path / "out" / "Common", output_path=tmp_path / "rebuilt" / "Common").build()
    rebuilt = tmp_path / "rebuilt" / RELPATH / "fsd50k_377266"
    assert read_attribution(rebuilt / ATTRIBUTION_FILE) == attribution
    assert _yaml(rebuilt / ANNOTATION_NAME) == _yaml(target / ANNOTATION_NAME)
    with pytest.raises(ValueError, match="not redistributable"):
        importer.attribution("327432")


def test_a_clip_outside_cc0_and_cc_by_is_refused_before_download(importer: FSD50K, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"'327432' is not redistributable: http://creativecommons.org/licenses/by-nc/3.0/"):
        importer.import_item("327432", tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not (importer.cache / "clips" / "eval").exists()
    with pytest.raises(KeyError, match="Freesound ids"):
        importer.fetch("1")


@pytest.mark.usefixtures("default_sounds")
def test_empty_wav_is_rejected(importer: FSD50K, tmp_path: Path) -> None:
    empty = tmp_path / "empty.wav"
    with wave.open(str(empty), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(44100)
    with pytest.raises(ValueError, match="empty wav"):
        importer.convert("182351", empty, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_kinds_map_covers_every_kind_offline() -> None:
    data = yaml.safe_load(KINDS_FILE.read_text())
    kinds = load_kinds()
    arena_kinds = set(yaml.safe_load(kinds_file().read_text())["kinds"])
    with open(FIXTURE / "labels" / "vocabulary.csv", newline="") as f:
        vocabulary = {row[1] for row in csv.reader(f)}
    assert set(load_spec().facets["kind"].values_for("sound")) == arena_kinds
    assert set(data) == arena_kinds
    assert set(kinds.values()) == arena_kinds
    assert set(kinds) <= vocabulary
    for kind, labels in data.items():
        for label in labels:
            assert kind_of([label], kinds) == kind
    assert kind_of(["Doorbell", "Door", "Domestic_sounds_and_home_sounds", "Alarm"], kinds) == "alarm"
    assert kind_of(["Knock", "Door", "Wood"], kinds) == "onset"
    assert kind_of(["Electric_guitar", "Guitar", "Plucked_string_instrument", "Musical_instrument", "Music"], kinds) == "music"
    assert kind_of(["Wind"], kinds) is None


def test_duplicate_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "kinds.yaml"
    path.write_text(yaml.safe_dump({"alarm": ["Siren"], "onset": ["Siren"]}))
    with pytest.raises(ValueError, match="Siren"):
        load_kinds(path)


def test_a_kind_outside_the_semantic_standard_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "kinds.yaml"
    path.write_text(yaml.safe_dump({"weather": ["Wind"]}))
    with pytest.raises(ValueError, match="'weather' is not an ASA sound kind"):
        load_kinds(path)


@pytest.mark.network
def test_one_clip_downloads_from_the_mirror_with_a_measurable_level(tmp_path: Path) -> None:
    path = FSD50K.default_provider().get("clips/dev/83479.wav", tmp_path / "83479.wav")
    with wave.open(str(path), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 44100)
    assert -60.0 < normalize_dbfs(path, PEAK_DBFS) < PEAK_DBFS
