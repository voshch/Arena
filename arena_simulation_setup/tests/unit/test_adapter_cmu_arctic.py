from __future__ import annotations

import contextlib
import functools
import http.server
import shutil
import threading
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from arena_assets.impl import ANNOTATION_NAME, DATABASE_NAME
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, read_attribution
from arena_assets.importers.providers import Http
from arena_assets.semantic import CURRENT_ASA
from arena_assets.utils.Database import Database
from arena_simulation_setup.tree.assets.Sound import SoundIdentifier
from arena_simulation_setup.tree.assets.sound_catalog import SoundLibrary, kinds_file, selection_seed

from arena_simulation_setup.adapters.cmu_arctic import BASE, LICENSE, VOICES, CMUArctic, asset_name, asset_tags, read_prompts, split_item
from arena_simulation_setup.adapters.sound import normalize_dbfs, peak_dbfs

PACKAGE = Path(__file__).resolve().parents[2]
FIXTURE = PACKAGE / "tests" / "fixtures" / "cmu_arctic"
ITEMS = {"bdl_a0507": "bdl", "awb_a0300": "awb", "rms_a0001": "rms"}
RELPATH = Path("Common") / "Sound"


def _wav(item: str) -> Path:
    voice, utterance = split_item(item)
    return FIXTURE / "cmu_arctic" / f"cmu_us_{voice}_arctic" / "wav" / f"arctic_{utterance}.wav"


def _yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def importer(tmp_path: Path) -> CMUArctic:
    cache = tmp_path / "cache"
    shutil.copytree(FIXTURE, cache, ignore=shutil.ignore_patterns("ATTRIBUTION.yaml"))
    return CMUArctic(cache=cache, provider=Http(BASE))


@pytest.fixture
def library(default_sounds: SoundLibrary) -> Iterator[SoundLibrary]:
    library = SoundLibrary([kinds_file()])
    try:
        yield library
    finally:
        library.use_world(None)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@contextlib.contextmanager
def serving(root: Path) -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_registry_loads_this_class() -> None:
    assert IMPORTERS.load("sound", "cmu_arctic") is CMUArctic
    assert CMUArctic.meta.kind == "sound"
    assert CMUArctic.meta.listing_ttl_s is None
    assert CMUArctic.default_provider().url("cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0001.wav") == "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0001.wav"


def test_item_ids_are_a_known_voice_and_an_utterance() -> None:
    assert split_item("bdl_a0001") == ("bdl", "a0001")
    assert split_item("awb_b0540") == ("awb", "b0540")
    for bad in ("xyz_a0001", "bdl_c0001", "bdl_a001", "bdl", "BDL_a0001"):
        with pytest.raises(KeyError, match="voices: awb, bdl"):
            split_item(bad)


def test_utterances_are_the_wavs_in_the_voice_directory_listing(tmp_path: Path) -> None:
    with serving(FIXTURE) as base:
        importer = CMUArctic(cache=tmp_path / "cache", provider=Http(base))
        assert {voice: importer.utterances(voice) for voice in ("bdl", "awb", "rms")} == {"bdl": ["a0507"], "awb": ["a0300"], "rms": ["a0001"]}


def test_missing_utterance_names_the_listing_to_pick_from(tmp_path: Path) -> None:
    with serving(FIXTURE) as base:
        importer = CMUArctic(cache=tmp_path / "cache", provider=Http(base))
        with pytest.raises(KeyError, match=r"no utterance 'a0001' for voice 'bdl', .*/cmu_us_bdl_arctic/wav/ lists them"):
            importer.fetch("bdl_a0001")
        assert importer.fetch("bdl_a0507").read_bytes() == _wav("bdl_a0507").read_bytes()


def test_text_is_what_the_voice_read_and_the_prompt_list_fills_gaps(importer: CMUArctic) -> None:
    prompts = read_prompts(FIXTURE / "cmuarctic.data")
    assert importer.text("awb_a0300") == "This tacit promise of continued acquaintance gave Saxon a little joy-thrill."
    assert importer.text("awb_a0300") != prompts["a0300"]
    assert importer.text("bdl_a0507") == prompts["a0507"] == "In short, my joyous individualism was dominated by the orthodox bourgeois ethics."
    assert importer.text("rms_a0300") == prompts["a0300"]
    with pytest.raises(KeyError, match="no text for 'rms_b0001'"):
        importer.text("rms_b0001")


def test_prompt_file_with_a_foreign_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "txt.done.data"
    path.write_text('( arctic_a0001 "Fine." )\n\nnot a prompt\n')
    with pytest.raises(ValueError, match=r"txt.done.data:3: not a festvox prompt line"):
        read_prompts(path)


def test_tags_are_the_origin_facet_then_the_speaker() -> None:
    assert asset_tags("ksp") == ["origin::cmu_arctic", "read_speech", "ksp", "male", "indian_english"]
    assert asset_tags("slt") == ["origin::cmu_arctic", "read_speech", "slt", "female", "us_english"]
    assert all("::" not in tag for voice in VOICES for tag in asset_tags(voice)[1:])


def test_convert_builds_a_speech_item_with_the_emission_of_the_speech_reference(importer: CMUArctic, library: SoundLibrary, tmp_path: Path) -> None:
    out = tmp_path / "out"
    reference = library.default_asset("speech")
    for item, voice in ITEMS.items():
        name = asset_name(item)
        assert importer.convert(item, importer.fetch(item), out) == str(RELPATH / name)
        target = out / RELPATH / name
        assert sorted(path.name for path in target.iterdir()) == sorted([f"{name}.wav", f"{name}.yaml", ANNOTATION_NAME])
        assert (target / f"{name}.wav").read_bytes() == _wav(item).read_bytes()
        manifest = _yaml(target / f"{name}.yaml")
        assert manifest["kind"] == "speech"
        assert manifest["desc"] == importer.text(item)
        assert manifest["tags"] == asset_tags(voice)
        assert (manifest["level_db"], manifest["reference_distance_m"]) == (reference.level_db, reference.reference_distance_m)
        assert manifest["normalize_dbfs"] == normalize_dbfs(_wav(item), peak_dbfs(library.kind("speech")))
        assert manifest["loop"] is False
        assert manifest["variants"] == [{"id": f"{name}_01", "file": f"{name}.wav"}]
        annotation = _yaml(target / ANNOTATION_NAME)
        assert annotation["asa"] == CURRENT_ASA
        assert annotation["kind"] == "speech"
        assert annotation["tags"][0] == "origin::cmu_arctic"
        assert annotation["tags"][-1] == "kind::speech"
        assert not (importer.cache / "stage" / item).exists()
    listed = Database(out / "Common" / DATABASE_NAME).list_all("sound")
    assert sorted(listed["ids"]) == sorted(f"Sound/{asset_name(item)}" for item in ITEMS)
    assert {metadata["origin"] for metadata in listed["metadatas"]} == {"cmu_arctic"}


def test_imported_utterance_resolves_through_the_sound_library_with_the_upstream_audio(importer: CMUArctic, library: SoundLibrary, tmp_path: Path) -> None:
    world = tmp_path / "world"
    target = importer.import_item("rms_a0001", world / "assets")
    library.use_world(world)
    name = asset_name("rms_a0001")
    assert target == world / "assets" / RELPATH / name
    assert SoundIdentifier.parse(name).resolve_sync().manifest["kind"] == "speech"
    asset = library.asset(name)
    assert asset.kind == "speech"
    assert asset.tags[0] == "origin::cmu_arctic"
    variant = asset.select(context={}, seed=selection_seed(name))
    assert variant.path == target / f"{name}.wav"
    assert variant.path.read_bytes() == _wav("rms_a0001").read_bytes()
    assert asset.normalize_dbfs == normalize_dbfs(_wav("rms_a0001"), peak_dbfs(library.kind("speech")))


@pytest.mark.usefixtures("default_sounds")
def test_attribution_carries_the_copyright_year_of_each_voice(importer: CMUArctic, tmp_path: Path) -> None:
    rms = read_attribution(importer.import_item("rms_a0001", tmp_path / "out") / ATTRIBUTION_FILE)
    bdl = read_attribution(importer.import_item("bdl_a0507", tmp_path / "out") / ATTRIBUTION_FILE)
    assert rms.source == "cmu_arctic"
    assert rms.license == bdl.license == LICENSE
    assert rms.authors == "Carnegie Mellon University, copyright 2004, speaker rms"
    assert bdl.authors == "Carnegie Mellon University, copyright 2003, speaker bdl"
    assert bdl.url == "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_bdl_arctic/wav/arctic_a0507.wav"
    assert "CMU-LTI-03-177" in rms.citation


@pytest.mark.network
def test_live_listing_holds_every_voice_and_one_utterance_decodes(tmp_path: Path) -> None:
    importer = CMUArctic(cache=tmp_path / "cache")
    items = list(importer.items())
    counts = {voice: sum(item.startswith(f"{voice}_") for item in items) for voice in VOICES}
    assert counts == {"awb": 1138, "bdl": 1132, "clb": 1132, "jmk": 1132, "ksp": 1132, "rms": 1132, "slt": 1132}
    path = importer.fetch("slt_a0001")
    with wave.open(str(path), "rb") as f:
        assert (f.getnchannels(), f.getsampwidth(), f.getframerate()) == (1, 2, 16000)
    assert importer.text("slt_a0001") == "Author of the danger trail, Philip Steels, etc."
