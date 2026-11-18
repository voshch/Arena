"""The one-time patch merging the per-bucket download caches into arena-public, on a real temporary tree."""

import os
import subprocess
from pathlib import Path

PATCH = Path(__file__).parents[2] / "patches" / "2026-11-18-asset-cache-arena-public"


def _file(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _run(data: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(PATCH)], env={**os.environ, "ARENA_DATA_DIR": str(data)}, capture_output=True, text=True, check=False)


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def test_source_caches_merge_into_arena_public_and_the_first_source_wins(tmp_path):
    root = tmp_path / "data" / "assets"
    _file(root / "default/Common/Object/a/a.usdz")
    _file(root / "default/Common/Object/a/a.ttl")
    _file(root / "default/Common/Object/c/c.usdz", "default")
    _file(root / "public/models/Common/Object/b/b.usdz")
    _file(root / "public/models/Common/Object/c/c.usdz", "public")
    _file(root / "public/worlds/w/world.yaml")
    _file(root / "public/benchmarks/suites/s/suite.yaml")
    _file(root / "default/.listing")
    _file(root / "default/.listing-children")
    _file(root / "public/worlds/.listing")
    _file(root / ".index/public.json")
    done = _run(tmp_path / "data")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == f"moved the asset cache to {root / 'arena-public'}"
    assert _tree(root) == [
        ".index/public.json",
        "arena-public/Common/Object/a/a.ttl",
        "arena-public/Common/Object/a/a.usdz",
        "arena-public/Common/Object/b/b.usdz",
        "arena-public/Common/Object/c/c.usdz",
        "arena-public/suites/s/suite.yaml",
        "arena-public/w/world.yaml",
    ]
    assert (root / "arena-public/Common/Object/c/c.usdz").read_text() == "default"
    assert sorted(p.name for p in root.iterdir()) == [".index", "arena-public"]


def test_a_second_run_changes_nothing(tmp_path):
    root = tmp_path / "data" / "assets"
    _file(root / "default/Common/Object/a/a.usdz")
    assert _run(tmp_path / "data").returncode == 0
    before = _tree(root)
    done = _run(tmp_path / "data")
    assert done.returncode == 0, done.stderr
    assert done.stdout == ""
    assert _tree(root) == before


def test_a_cache_without_old_sources_is_done(tmp_path):
    done = _run(tmp_path / "data")
    assert done.returncode == 0
    assert done.stdout == ""
