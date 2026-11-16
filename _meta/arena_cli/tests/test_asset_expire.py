"""arena asset expire against real bucket caches under a temporary ARENA_ASSETS_DIR."""

import os
import subprocess
import sys
from pathlib import Path


def _entry(bucket: Path, relpath: str, stamp: str = "1700000000") -> Path:
    entry = bucket / relpath
    entry.mkdir(parents=True)
    (entry / ".ttl").write_text(stamp)
    return entry


def _expire(assets: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "ARENA_WS_DIR": str(assets),
        "ARENA_ASSETS_DIR": str(assets),
        "ASSET_BUCKETS": "assets",
        "WORLD_BUCKETS": "worlds",
        "BENCHMARK_BUCKETS": "bench",
        "PYTHONPATH": os.pathsep.join([str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH", "")]),
    }
    return subprocess.run([sys.executable, "-m", "arena_cli", "asset", "expire", *args], env=env, capture_output=True, text=True, check=False)


def test_bare_expire_marks_every_bucket_and_drops_listings(tmp_path):
    crate = _entry(tmp_path / "assets", "Common/Object/Crate")
    world = _entry(tmp_path / "worlds", "office")
    for listing in (".listing", ".listing-children"):
        (tmp_path / "worlds" / listing).write_text("office\n")
    done = _expire(tmp_path)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["assets: expired 1 cached asset(s)", "worlds: expired 1 cached asset(s)", "bench: expired 0 cached asset(s)"]
    assert (crate / ".ttl").read_text() == "1"
    assert (world / ".ttl").read_text() == "1"
    assert sorted(p.name for p in (tmp_path / "worlds").iterdir()) == ["office"]


def test_partial_fetch_is_left_unmarked(tmp_path):
    partial = _entry(tmp_path / "assets", "Common/Object/Half", stamp="0")
    done = _expire(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "assets: expired 0 cached asset(s)" in done.stdout
    assert (partial / ".ttl").read_text() == "0"


def test_bucket_option_limits_bare_expire(tmp_path):
    crate = _entry(tmp_path / "assets", "Common/Object/Crate")
    world = _entry(tmp_path / "worlds", "office")
    done = _expire(tmp_path, "--bucket", "worlds")
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == ["worlds: expired 1 cached asset(s)"]
    assert (crate / ".ttl").read_text() == "1700000000"
    assert (world / ".ttl").read_text() == "1"


def test_one_asset_expires_alone(tmp_path):
    crate = _entry(tmp_path / "assets", "Common/Object/Crate")
    chair = _entry(tmp_path / "assets", "Common/Object/Chair")
    (tmp_path / "assets" / ".listing").write_text("Common/Object/Crate\n")
    done = _expire(tmp_path, "object", "Common/Crate")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "expired object Common/Crate in assets"
    assert (crate / ".ttl").read_text() == "1"
    assert (chair / ".ttl").read_text() == "1700000000"
    assert (tmp_path / "assets" / ".listing").exists()


def test_uncached_asset_names_the_pull(tmp_path):
    done = _expire(tmp_path, "world", "office")
    assert done.returncode == 1
    assert "world office is not cached from worlds" in done.stderr
    assert "arena asset pull world office" in done.stderr


def test_kind_without_name_lists_the_kinds(tmp_path):
    done = _expire(tmp_path, "world")
    assert done.returncode != 0
    assert "expected KIND NAME or nothing" in done.stderr
    assert "object" in done.stderr


def test_unknown_bucket_lists_the_configured_ones(tmp_path):
    done = _expire(tmp_path, "--bucket", "elsewhere")
    assert done.returncode != 0
    assert "elsewhere is not a configured bucket, expected one of assets, worlds, bench" in done.stderr
