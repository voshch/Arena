"""Tests for arena update's submodule branch reattach against real git repos."""

import os
import subprocess
from pathlib import Path

from arena_cli.pull import attach_branch, restore_branches

ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], env=ENV, capture_output=True, text=True, check=True).stdout.strip()


def commit(repo: Path, msg: str) -> str:
    git(repo, "commit", "-q", "--allow-empty", "-m", msg)
    return git(repo, "rev-parse", "HEAD")


def repo_at_pin(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "sub"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "jazzy")
    pin = commit(repo, "pin")
    git(repo, "update-ref", "refs/remotes/origin/jazzy", pin)
    return repo, pin


def head(repo: Path) -> tuple[str, str]:
    return git(repo, "rev-parse", "HEAD"), git(repo, "branch", "--show-current")


def test_branch_ahead_of_pin_with_unpushed_commits_is_kept(tmp_path):
    repo, pin = repo_at_pin(tmp_path)
    tip = commit(repo, "local work")
    git(repo, "switch", "-q", "--detach", pin)
    note, ok = attach_branch(str(repo), "jazzy", ENV)
    assert ok
    assert "1 unpushed" in note
    assert head(repo) == (tip, "jazzy")


def test_branch_diverged_from_pin_stays_beside_detached_pin(tmp_path):
    repo, pin = repo_at_pin(tmp_path)
    tip = commit(repo, "local work")
    git(repo, "switch", "-q", "--detach", pin)
    new_pin = commit(repo, "upstream")
    note, ok = attach_branch(str(repo), "jazzy", ENV)
    assert ok
    assert "rebase" in note
    assert head(repo) == (new_pin, "")
    assert git(repo, "rev-parse", "jazzy") == tip


def test_pushed_branch_ahead_of_pin_is_reset_onto_pin(tmp_path):
    repo, pin = repo_at_pin(tmp_path)
    tip = commit(repo, "pushed work")
    git(repo, "update-ref", "refs/remotes/origin/jazzy", tip)
    git(repo, "switch", "-q", "--detach", pin)
    assert attach_branch(str(repo), "jazzy", ENV) == ("", True)
    assert head(repo) == (pin, "jazzy")


def test_branch_behind_pin_moves_onto_pin(tmp_path):
    repo, _ = repo_at_pin(tmp_path)
    git(repo, "switch", "-q", "--detach")
    new_pin = commit(repo, "upstream")
    assert attach_branch(str(repo), "jazzy", ENV) == ("", True)
    assert head(repo) == (new_pin, "jazzy")


def test_missing_branch_is_created_at_pin(tmp_path):
    repo, pin = repo_at_pin(tmp_path)
    git(repo, "switch", "-q", "--detach")
    assert attach_branch(str(repo), "humble", ENV) == ("", True)
    assert head(repo) == (pin, "humble")


def test_restore_branches_walks_nested_submodules(tmp_path, capsys):
    leaf, _ = repo_at_pin(tmp_path)
    mid = tmp_path / "mid"
    mid.mkdir()
    git(mid, "init", "-q", "-b", "jazzy")
    git(mid, "-c", "protocol.file.allow=always", "submodule", "add", "-q", "-b", "jazzy", str(leaf), "leaf")
    commit(mid, "add leaf")
    top = tmp_path / "top"
    top.mkdir()
    git(top, "init", "-q", "-b", "jazzy")
    git(top, "-c", "protocol.file.allow=always", "submodule", "add", "-q", "-b", "jazzy", str(mid), "mid")
    commit(top, "add mid")
    git(top, "-c", "protocol.file.allow=always", "submodule", "update", "-q", "--init", "--recursive")
    nested = top / "mid" / "leaf"
    pin = git(nested, "rev-parse", "HEAD")
    git(nested, "switch", "-q", "-C", "jazzy")
    tip = commit(nested, "local work")
    git(nested, "switch", "-q", "--detach", pin)
    assert restore_branches(str(top), ENV)
    assert head(nested) == (tip, "jazzy")
    assert git(top / "mid", "branch", "--show-current") == "jazzy"
    assert "mid/leaf: kept jazzy, 1 unpushed" in capsys.readouterr().out
