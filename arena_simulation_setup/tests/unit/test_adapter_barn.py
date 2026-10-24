"""BARN world importer: one challenge world by number, written through the world tree and resolved by name."""

from __future__ import annotations

import contextlib
import functools
import http.server
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent.parent / "fixtures" / "barn" / "world_0.world"
WORLD = "barn_0"
FILES = ["ATTRIBUTION.yaml", "kg.yaml", "map/0.png", "map/0.yaml", "scenarios/barn/scenario.yaml", "world.yaml"]


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")


def seed(cache: Path) -> Path:
    (cache / "worlds").mkdir(parents=True)
    shutil.copy(FIXTURE, cache / "worlds" / FIXTURE.name)
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from arena_simulation_setup.adapters.barn import BARNWorld

    return BARNWorld(cache=seed(tmp_path / "cache"))


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def test_registry_target_is_the_class() -> None:
    from arena_simulation_setup.adapters.barn import BARNWorld

    from arena_assets.importers import IMPORTERS

    assert IMPORTERS.load("world", "barn") is BARNWorld
    assert BARNWorld.meta.kind == "world"


def test_items_are_the_challenge_world_numbers(importer) -> None:
    items = list(importer.items())
    assert (items[0], items[-1], len(items)) == ("0", "299", 300)


@pytest.mark.parametrize("item", ["300", "-1", "barn", "1.5", ""])
def test_item_outside_the_challenge_is_refused(importer, item: str) -> None:
    with pytest.raises(ValueError, match="0 to 299"):
        importer.fetch(item)


def test_fetch_keeps_a_cached_world_file(importer) -> None:
    assert importer.fetch("0") == importer.cache / "worlds" / FIXTURE.name


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


def test_fetch_downloads_through_the_injected_provider_into_the_generator_cache_layout(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.barn import BARNWorld
    from arena_simulation_setup.generative.world.barn import WORLD_DIR, fetch_world

    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution
    from arena_assets.importers.providers import Http

    upstream = tmp_path / "upstream" / WORLD_DIR
    upstream.mkdir(parents=True)
    shutil.copy(FIXTURE, upstream / FIXTURE.name)
    with serving(tmp_path / "upstream") as url:
        fetched = BARNWorld(cache=tmp_path / "cache", provider=Http(url)).fetch("0")
        assert fetch_world(0, tmp_path / "direct", Http(url)) == tmp_path / "direct" / "worlds" / FIXTURE.name
    assert fetched == tmp_path / "cache" / "worlds" / FIXTURE.name
    assert fetched.read_bytes() == FIXTURE.read_bytes() == (tmp_path / "direct" / "worlds" / FIXTURE.name).read_bytes()
    assert read_attribution(tmp_path / "cache" / ATTRIBUTION_FILE) == read_attribution(tmp_path / "direct" / ATTRIBUTION_FILE)


def test_import_writes_a_world_the_world_tree_loads(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.tree.World import WorldView

    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    out = tmp_path / "worlds"
    target = importer.import_item("0", out)
    assert target == out / WORLD
    assert _files(target) == FILES
    view = WorldView(target)
    assert list(view.load().levels) == ["0"]
    assert all(identifier.resolve_source_sync() for identifier in view.identifiers(strict=True))
    assert [scenario.name for scenario in view.scenario.listall()] == ["barn"]
    attribution = read_attribution(target / ATTRIBUTION_FILE)
    assert (attribution.license, attribution.url) == ("MIT", "https://github.com/Daffan/the-barn-challenge")


def test_world_is_the_one_the_generator_builds_for_that_number(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.generative.world import WorldGenerator, WorldGeneratorType
    from arena_simulation_setup.generative.world.world_generator import save_generated
    from arena_simulation_setup.tree.World import WorldView, derive

    imported = importer.import_item("0", tmp_path / "worlds")
    generator = WorldGenerator(WorldGeneratorType.BARN, {"index": 0, "source": str(FIXTURE.parent)})
    built = generator.compute()
    built.kg = derive(built)
    generated = save_generated(WorldView(tmp_path / "generated" / WORLD), built, generator.files())
    for relpath in ("world.yaml", "kg.yaml", "map/0.png", "scenarios/barn/scenario.yaml"):
        assert (imported / relpath).read_bytes() == (generated / relpath).read_bytes()


def test_reimport_replaces_the_world(importer, tmp_path: Path) -> None:
    out = tmp_path / "worlds"
    stale = importer.import_item("0", out) / "stale.txt"
    stale.write_text("x")
    importer.import_item("0", out)
    assert _files(out / WORLD) == FILES


def test_cli_import_lands_where_the_world_resolver_looks(tmp_path: Path) -> None:
    cache = seed(tmp_path / "cache")
    env = {**os.environ, "ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "imported")}
    target = tmp_path / "imported" / "worlds" / WORLD
    imported = subprocess.run([sys.executable, "-m", "arena_assets", "import", "world", "barn", "0", "--cache", str(cache)], env=env, capture_output=True, text=True, check=True)
    assert imported.stdout.splitlines()[-1] == str(target)
    code = "from arena_simulation_setup.tree.World import WorldIdentifier\nview = WorldIdentifier(%r).resolve_sync()\nprint(view.path)\nprint(*sorted(scenario.name for scenario in view.scenario.listall()))" % WORLD
    resolved = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert resolved.stdout.splitlines() == [str(target), "barn"]


def test_cli_refuses_a_number_outside_the_challenge_without_a_traceback(tmp_path: Path) -> None:
    refused = subprocess.run([sys.executable, "-m", "arena_assets", "import", "world", "barn", "300", "--cache", str(tmp_path / "cache"), "--out", str(tmp_path / "out")], capture_output=True, text=True, check=False)
    assert refused.returncode == 2
    assert "0 to 299" in " ".join(refused.stderr.replace("\u2502", " ").split())
    assert "Traceback" not in refused.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.network
def test_live_fetch_of_one_world_file(tmp_path: Path) -> None:
    from arena_simulation_setup.adapters.barn import BARNWorld

    src = BARNWorld(cache=tmp_path / "cache").fetch("17")
    assert src.name == "world_17.world"
    assert "unit_cylinder" in src.read_text()
