"""The one-time patch moving the download cache out of the checkout, on a real temporary tree."""

import os
import subprocess
from pathlib import Path

PATCH = Path(__file__).parents[2] / "patches" / "2026-11-04-asset-cache-to-data"


def _file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def _run(arena: Path, data: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(PATCH)], env={**os.environ, "ARENA_DIR": str(arena), "ARENA_DATA_DIR": str(data)}, capture_output=True, text=True, check=False)


def test_caches_move_under_the_new_names_and_local_roots_stay(tmp_path):
    old = tmp_path / "Arena" / "_assets"
    for rel in ("default/Common/Object/a/a.usdz", "arena-worlds-prod-public/w/world.yaml", "arena-benchmarks-prod-public/suites/s/suite.yaml", ".import/barn/items.json", "_composed/c/world.yaml", "_imported/worlds/i/world.yaml", "_local/worlds/l/world.yaml"):
        _file(old / rel)
    done = _run(tmp_path / "Arena", tmp_path / "data")
    assert done.returncode == 0, done.stderr
    new = tmp_path / "data" / "assets"
    for rel in ("default/Common/Object/a/a.usdz", "public/worlds/w/world.yaml", "public/benchmarks/suites/s/suite.yaml", ".import/barn/items.json", "_composed/c/world.yaml"):
        assert (new / rel).is_file(), rel
    assert sorted(p.name for p in old.iterdir()) == ["_imported", "_local"]


def test_a_second_run_changes_nothing_and_an_existing_copy_wins(tmp_path):
    old = tmp_path / "Arena" / "_assets"
    _file(old / "default" / "stale.bin")
    _file(tmp_path / "data" / "assets" / "default" / "fresh.bin")
    assert _run(tmp_path / "Arena", tmp_path / "data").returncode == 0
    assert _run(tmp_path / "Arena", tmp_path / "data").returncode == 0
    assert [p.name for p in (tmp_path / "data" / "assets" / "default").iterdir()] == ["fresh.bin"]
    assert not (old / "default").exists()


def test_a_checkout_without_the_old_directory_is_done(tmp_path):
    assert _run(tmp_path / "Arena", tmp_path / "data").returncode == 0
