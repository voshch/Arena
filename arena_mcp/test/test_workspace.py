from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("arena_cli")

from arena_mcp.core import JOBS, cli
from arena_mcp.core.errors import ToolError
from arena_mcp.domains import workspace

GIT_ENV = {
    **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "protocol.file.allow=always", *args], env=GIT_ENV, capture_output=True, text=True, check=True).stdout.strip()


def commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", name)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A metarepo clone with two submodules, one initialized and moved off its pin, one left uninitialized, and an upstream one commit ahead."""
    origin = tmp_path / "origin"
    for name in ("lib_a", "lib_b"):
        repo = tmp_path / name
        repo.mkdir()
        git(repo, "init", "-q", "-b", "main")
        commit(repo, "a.txt", "a\n")
    origin.mkdir()
    git(origin, "init", "-q", "-b", "jazzy")
    commit(origin, "readme", "r\n")
    git(origin, "submodule", "add", "-q", str(tmp_path / "lib_a"), "lib_a")
    git(origin, "submodule", "add", "-q", str(tmp_path / "lib_b"), "deps/lib_b")
    git(origin, "commit", "-q", "-m", "submodules")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    git(clone, "submodule", "update", "--init", "-q", "lib_a")
    commit(clone / "lib_a", "b.txt", "b\n")
    commit(origin, "later", "l\n")
    git(clone, "fetch", "-q")
    commit(clone, "local", "x\n")
    (clone / "readme").write_text("changed\n")
    (clone / "scratch").write_text("new\n")
    return clone


def test_tree_status_reads_branches_pins_and_dirt(tree: Path) -> None:
    status = workspace.tree_status(tree)
    top, lib_a = status["repos"]
    assert top["path"] == "." and top["branch"] == "jazzy" and top["upstream"] == "origin/jazzy"
    assert (top["ahead"], top["behind"], top["changed"], top["untracked"], top["dirty"]) == (1, 1, 1, 1, True)
    assert lib_a["path"] == "lib_a" and lib_a["pin"] == "off_pin" and not lib_a["dirty"]
    assert status["uninitialized"] == ["deps/lib_b"]
    assert not (tree / ".git" / "index.lock").exists()


def test_tree_status_of_the_real_checkout() -> None:
    root = Path(os.environ.get("ARENA_DIR", Path(__file__).resolve().parents[2]))
    if subprocess.run(["git", "-C", str(root), "rev-parse"], capture_output=True, check=False).returncode:
        pytest.skip(f"{root} is not a git checkout")
    status = workspace.tree_status(root)
    assert status["repos"][0]["path"] == "."
    assert "commit" in status["repos"][0]
    assert {repo["path"] for repo in status["repos"][1:]}.isdisjoint(status["uninitialized"])


def test_rows_split_values_warnings_and_hints() -> None:
    text = "net:       local\nsource:    default\nwarning:   the running container is stale, run 'source arena' to recreate it\nset it with 'arena settings net local|lan'\n"
    assert workspace.rows(text) == {"net": "local", "source": "default", "warnings": ["the running container is stale, run 'source arena' to recreate it"], "notes": ["set it with 'arena settings net local|lan'"]}


def test_fork_and_robot_tables_parse() -> None:
    forks = "FORK  KIND  RELEASE          DOMAIN  STATE\nmcp   kept  20261019-1700    21      running\np1    pool  20261019-1700    22      exited\n\nRELEASE          CREATED              CHANGED     COPIED  FORKS\n"
    assert workspace.fork_rows(forks) == [
        {"fork": "mcp", "kind": "kept", "release": "20261019-1700", "domain": "21", "state": "running"},
        {"fork": "p1", "kind": "pool", "release": "20261019-1700", "domain": "22", "state": "exited"},
    ]
    assert workspace.fork_rows("no forks, 'arena fork new' makes one\n") == []
    assert workspace.robot_rows("[x] jackal\n[ ] go1\n[x] arms/ur5\n") == {"ready": ["jackal", "arms/ur5"], "pending": ["go1"]}


needs_arena = pytest.mark.skipif(not os.environ.get("ARENA_DIR"), reason="needs a sourced Arena environment")
needs_cli = pytest.mark.skipif(not cli.sourced(), reason="needs a shell that sourced arena")


@needs_arena
def test_features_come_from_the_cli_registry() -> None:
    from arena_cli import features

    listed = {entry["name"]: entry for entry in workspace.workspace_features()}
    assert set(listed) <= set(features.available())
    assert listed["robots"]["installed"] and "add" in listed["robots"]["commands"]
    assert all(entry["summary"] for entry in listed.values())


@needs_arena
def test_feature_and_setting_arguments_are_validated() -> None:
    with pytest.raises(ToolError, match="no feature") as missing:
        asyncio.run(workspace.workspace_feature("no_such_feature", "install"))
    assert "robots" in missing.value.hint
    with pytest.raises(ToolError, match="no 'stop' action") as action:
        asyncio.run(workspace.workspace_feature("robots", "stop"))
    assert "update" in action.value.hint
    with pytest.raises(ToolError, match="unknown value") as net:
        workspace.workspace_settings_net("wan")
    assert net.value.hint == "values: local, lan"
    with pytest.raises(ToolError, match="unknown value") as gpu:
        workspace.workspace_settings_gpu("maybe")
    assert gpu.value.hint == "values: on, off"
    with pytest.raises(ToolError, match="unknown action"):
        asyncio.run(workspace.workspace_robots("check"))
    with pytest.raises(ToolError, match="needs robot names"):
        asyncio.run(workspace.workspace_robots("add"))
    with pytest.raises(ToolError, match="not a fork name"):
        asyncio.run(workspace.workspace_fork_down("--all"))


@needs_cli
def test_net_setting_reads_through_the_real_verb() -> None:
    read = workspace.workspace_settings_net()
    assert read["net"] in ("local", "lan") and read["values"] == ["local", "lan"]
    assert "applies" not in read


@needs_cli
def test_robots_ls_reads_submodule_state() -> None:
    listed = asyncio.run(workspace.workspace_robots("ls"))
    assert listed["ready"] or listed["pending"]


@needs_cli
def test_fork_verbs_surface_the_cli_refusal_inside_a_fork(tmp_path: Path) -> None:
    saved = {key: os.environ.get(key) for key in ("ARENA_FORK", "ARENA_DATA_DIR")}
    os.environ.update(ARENA_FORK="mcp_test_fork", ARENA_DATA_DIR=str(tmp_path))
    try:
        with pytest.raises(ToolError, match="manage forks from the dev container"):
            workspace.workspace_forks()

        async def down() -> tuple[str, str]:
            job = await workspace.workspace_fork_down("p9")
            done = await JOBS.wait(job["id"], 60)
            return done.state, JOBS.tail(done.id)

        state, log = asyncio.run(down())
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert state == "failed"
    assert "this is fork mcp_test_fork, manage forks from the dev container" in log


@needs_cli
def test_status_of_the_live_workspace() -> None:
    root = Path(os.environ["ARENA_DIR"])
    if subprocess.run(["git", "-C", str(root), "rev-parse"], capture_output=True, check=False).returncode:
        pytest.skip(f"{root} is not a git checkout")
    status = workspace.workspace_status()
    assert status["repos"][0]["path"] == "."
    assert {"planners", "robots"} <= set(status["features"])
    assert status["settings"]["net"]["net"] in ("local", "lan")
    assert "gpu" in status["settings"]["gpu"] or "unavailable" in status["settings"]["gpu"]
