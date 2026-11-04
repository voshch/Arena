"""arena asset push, browse, pull and rm against a loopback asset store."""

import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ARENA = Path(__file__).resolve().parents[3]
STORE_HOST = ARENA / "arena_assets" / "tests"
if not (STORE_HOST / "store_host.py").is_file():
    pytest.skip("needs the arena_assets checkout for its loopback store", allow_module_level=True)


def _environment(workspace: Path, **extra: str) -> dict[str, str]:
    inherited = {key: value for key, value in os.environ.items() if not key.startswith("ARENA_") and not key.endswith("_BUCKETS")}
    return {**inherited, "ARENA_WS_DIR": str(workspace), "ARENA_ASSETS_NET": "on", "PYTHONPATH": os.pathsep.join([str(ARENA / "_meta"), os.environ.get("PYTHONPATH", "")]), **extra}


def _asset(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "arena_cli", "asset", *args], env=env, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)


@pytest.fixture
def store(monkeypatch):
    monkeypatch.syspath_prepend(str(STORE_HOST))
    from store_host import Store

    server = Store()
    server.buckets["arena-internal"] = {}
    server.private.add("arena-internal")
    server.tokens["tok"] = "alice"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def arena(store, tmp_path):
    env = _environment(
        tmp_path,
        ARENA_ASSETS_URL=store.base,
        ARENA_ASSETS_TOKEN="tok",
        ARENA_ASSETS_DIR=str(tmp_path / "cache"),
        ARENA_ASSETS_DIR_LOCAL=str(tmp_path / "local"),
        ASSET_BUCKETS="default,arena-internal",
    )

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return _asset(env, *args)

    return run


def _clip(root: Path) -> Path:
    clip = root / "local" / "Common" / "Animation" / "wave"
    clip.mkdir(parents=True)
    for file in ("wave.npz", "wave.anim", "annotation.yaml"):
        (clip / file).write_text(file)
    return clip


def test_push_browse_pull_and_rm_round_trip(store, arena, tmp_path):
    clip = _clip(tmp_path)

    pushed = arena("push", "animation", "wave", "--bucket", "arena-internal", "--yes")
    assert pushed.returncode == 0, pushed.stderr
    assert pushed.stdout.splitlines()[-1] == "published 1 animation to arena-internal:Common/Animation/wave: sent 3 of 3 file(s), 0.0 MB, 0 already up to date"
    assert sorted(store.buckets["arena-internal"]) == ["Common/Animation/wave/annotation.yaml", "Common/Animation/wave/wave.anim", "Common/Animation/wave/wave.npz"]

    again = arena("push", "animation", "wave", "--bucket", "arena-internal", "--yes")
    assert again.returncode == 0, again.stderr
    assert again.stdout.splitlines()[-1] == "animation wave is up to date on arena-internal:Common/Animation/wave: all 3 file(s) already there"

    browsed = arena("browse", "arena-internal/Common/Animation")
    assert browsed.returncode == 0, browsed.stderr
    assert browsed.stdout.split()[0] == "wave/" and browsed.stdout.rstrip().endswith("3 file(s)")

    clip.rename(tmp_path / "moved")
    pulled = arena("pull", "animation", "wave", "--bucket", "arena-internal")
    assert pulled.returncode == 0, pulled.stderr
    assert (Path(pulled.stdout.strip()) / "wave.anim").read_text() == "wave.anim"

    removed = arena("rm", "animation", "wave", "--bucket", "arena-internal", "--yes")
    assert removed.returncode == 0, removed.stderr
    assert removed.stdout.splitlines()[-1] == "removed animation wave from arena-internal:Common/Animation/wave: 3 file(s)"
    assert store.buckets["arena-internal"] == {}
    assert not Path(pulled.stdout.strip()).exists()

    gone = arena("pull", "animation", "wave", "--bucket", "arena-internal")
    assert gone.returncode == 1
    assert "animation wave is not in arena-internal" in gone.stderr


def test_bare_browse_lists_the_sources_and_their_kinds(arena):
    done = arena("browse")
    assert done.returncode == 0, done.stderr
    rows = {line.split()[0]: line.split()[1:] for line in done.stdout.splitlines() if line.strip() and not line.startswith("`")}
    assert "animation" in rows["public/models/"]
    assert rows["public/worlds/"] == ["world"]
    assert rows["public/benchmarks/"] == ["suite", "contest", "manifest"]
    assert "animation" in rows["arena-internal/"]


def test_pull_says_why_a_fetch_failed(store, arena):
    store.tokens.clear()
    done = arena("pull", "animation", "wave", "--bucket", "arena-internal")
    assert done.returncode == 1
    assert "animation wave was not pulled from arena-internal: GET arena-internal/index.json: 401 bearer token required, the token was not accepted" in done.stderr


def test_rm_needs_confirmation_without_a_terminal(arena):
    done = arena("rm", "animation", "wave", "--bucket", "arena-internal")
    assert done.returncode != 0
    assert "needs confirmation, pass --yes" in done.stderr


def test_pull_without_a_cache_directory_says_which_variable_to_set(tmp_path):
    done = _asset(_environment(tmp_path), "pull", "world", "anything")
    assert done.returncode == 1
    assert "no asset cache directory: set ARENA_DATA_DIR or ARENA_ASSETS_DIR" in done.stderr
