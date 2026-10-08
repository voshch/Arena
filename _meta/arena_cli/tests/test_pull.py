"""Tests for arena update's submodule branch reattach and patch runner against real git repos and scripts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from arena_cli.pull import apply_patches, attach_branch, restart_command, restore_branches

ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}

PEOPLE_MSGS_PATCH = Path(__file__).parents[2] / "patches" / "2026-10-07-people-msgs-path"


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


def patch(arena: Path, name: str, body: str, mode: int = 0o755) -> Path:
    path = arena / "_meta" / "patches" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(mode)
    return path


def test_passing_patch_runs_once_and_leaves_a_marker(tmp_path):
    script = patch(tmp_path, "2026-01-01-ok", 'echo run >> "$LOG"')
    env = {**ENV, "LOG": str(tmp_path / "log")}
    assert apply_patches(str(tmp_path), env) == []
    assert Path(f"{script}.done").exists()
    assert apply_patches(str(tmp_path), env) == []
    assert (tmp_path / "log").read_text() == "run\n"


def test_failing_patch_is_reported_unmarked_and_retried(tmp_path):
    script = patch(tmp_path, "2026-01-01-fails", 'echo run >> "$LOG"; exit 3')
    env = {**ENV, "LOG": str(tmp_path / "log")}
    assert apply_patches(str(tmp_path), env) == ["2026-01-01-fails"]
    assert not Path(f"{script}.done").exists()
    assert apply_patches(str(tmp_path), env) == ["2026-01-01-fails"]
    assert (tmp_path / "log").read_text() == "run\nrun\n"


def test_patch_without_exec_bit_is_reported_unmarked(tmp_path, capsys):
    script = patch(tmp_path, "2026-01-01-noexec", "exit 0", mode=0o644)
    assert apply_patches(str(tmp_path), ENV) == ["2026-01-01-noexec"]
    assert not Path(f"{script}.done").exists()
    assert "patch 2026-01-01-noexec: " in capsys.readouterr().err


def test_dotfiles_markers_and_markdown_are_not_run(tmp_path):
    for name in (".gitignore", "README.md", "2026-01-01-gone.done"):
        patch(tmp_path, name, 'echo run >> "$LOG"')
    env = {**ENV, "LOG": str(tmp_path / "log")}
    assert apply_patches(str(tmp_path), env) == []
    assert not (tmp_path / "log").exists()
    assert sorted(p.name for p in (tmp_path / "_meta" / "patches").iterdir()) == [".gitignore", "2026-01-01-gone.done", "README.md"]


def test_patches_run_in_name_order_from_the_arena_dir(tmp_path):
    for name in ("2026-02-01-b", "2026-01-01-a", "2026-03-01-c"):
        patch(tmp_path, name, f'echo "{name} $PWD" >> "$LOG"')
    env = {**ENV, "LOG": str(tmp_path / "log")}
    assert apply_patches(str(tmp_path), env) == []
    assert (tmp_path / "log").read_text().splitlines() == [f"{name} {tmp_path}" for name in ("2026-01-01-a", "2026-02-01-b", "2026-03-01-c")]


def test_failed_patch_does_not_stop_later_patches(tmp_path):
    patch(tmp_path, "2026-01-01-fails", "exit 1")
    later = patch(tmp_path, "2026-02-01-ok", "exit 0")
    assert apply_patches(str(tmp_path), ENV) == ["2026-01-01-fails"]
    assert Path(f"{later}.done").exists()


def test_missing_patches_dir_is_not_an_error(tmp_path):
    assert apply_patches(str(tmp_path), ENV) == []


def test_directories_beside_patches_are_not_run(tmp_path):
    helper = tmp_path / "_meta" / "patches" / "2026-01-01-data"
    helper.mkdir(parents=True)
    assert apply_patches(str(tmp_path), ENV) == []
    assert not Path(f"{helper}.done").exists()


def test_patch_killed_by_a_signal_is_reported_unmarked(tmp_path):
    script = patch(tmp_path, "2026-01-01-killed", "kill -9 $$")
    assert apply_patches(str(tmp_path), ENV) == ["2026-01-01-killed"]
    assert not Path(f"{script}.done").exists()


def test_patch_without_shebang_line_is_reported_unmarked(tmp_path):
    script = tmp_path / "_meta" / "patches" / "2026-01-01-noshebang"
    script.parent.mkdir(parents=True)
    script.write_bytes(b"\x00\x01 not a program\n")
    script.chmod(0o755)
    assert apply_patches(str(tmp_path), ENV) == ["2026-01-01-noshebang"]
    assert not Path(f"{script}.done").exists()


def test_patch_reads_end_of_file_on_stdin(tmp_path):
    patch(tmp_path, "2026-01-01-stdin", "read line && exit 1\nexit 0")
    code = f"from arena_cli.pull import apply_patches; import os; print(apply_patches({str(tmp_path)!r}, dict(os.environ)))"
    env = {**ENV, "PYTHONPATH": str(Path(__file__).parents[2])}
    updater = subprocess.run([sys.executable, "-c", code], env=env, input="typed answer\n", capture_output=True, text=True, check=True)
    assert updater.stdout.splitlines()[-1] == "[]"


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes into a read-only directory")
def test_patch_whose_marker_cannot_be_written_is_reported(tmp_path, capsys):
    script = patch(tmp_path, "2026-01-01-readonly", "exit 0")
    script.parent.chmod(0o555)
    try:
        assert apply_patches(str(tmp_path), ENV) == ["2026-01-01-readonly"]
    finally:
        script.parent.chmod(0o755)
    assert not Path(f"{script}.done").exists()
    assert "2026-01-01-readonly.done" in capsys.readouterr().err


def test_people_msgs_patch_removes_stale_checkout_and_build_dir(tmp_path):
    hunav = tmp_path / "src" / "deps" / "hunav"
    for kept in (hunav / "hunav_sim" / ".git", tmp_path / "src" / "deps" / "people_msgs" / ".git", tmp_path / "build" / "hunav_sim"):
        kept.mkdir(parents=True)
    stale_people_msgs(tmp_path)
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "removed stale deps/hunav/people_msgs" in done.stdout
    assert sorted(p.name for p in hunav.iterdir()) == ["hunav_sim"]
    assert sorted(p.name for p in (tmp_path / "src" / "deps").iterdir()) == ["hunav", "people_msgs"]
    assert sorted(p.name for p in (tmp_path / "build").iterdir()) == ["hunav_sim"]


def stale_people_msgs(ws: Path) -> Path:
    stale = ws / "src" / "deps" / "hunav" / "people_msgs"
    remote = ws / "remote"
    remote.mkdir(parents=True)
    git(remote, "init", "-q", "--bare")
    git(ws, "clone", "-q", str(remote), str(stale))
    (stale / "people_msgs").mkdir()
    (stale / "people_msgs" / "package.xml").write_text("<package/>\n")
    git(stale, "add", ".")
    commit(stale, "pin")
    git(stale, "push", "-q", "origin", "HEAD:refs/heads/jazzy")
    git(stale, "fetch", "-q")
    (ws / "build" / "people_msgs").mkdir(parents=True)
    return stale


def run_people_msgs_patch(ws: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(PEOPLE_MSGS_PATCH)], env={**ENV, "ARENA_WS_DIR": str(ws)}, capture_output=True, text=True, check=False)


def test_people_msgs_patch_removes_a_clean_pushed_clone(tmp_path):
    stale = stale_people_msgs(tmp_path)
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 0, done.stderr
    assert not stale.exists()
    assert not (tmp_path / "build" / "people_msgs").exists()


def test_people_msgs_patch_keeps_a_clone_with_unpushed_commits(tmp_path):
    stale = stale_people_msgs(tmp_path)
    commit(stale, "local fix")
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 1
    assert "local changes" in done.stderr
    assert (stale / "people_msgs" / "package.xml").exists()
    assert (tmp_path / "build" / "people_msgs").is_dir()


def test_people_msgs_patch_keeps_a_clone_with_uncommitted_edits(tmp_path):
    stale = stale_people_msgs(tmp_path)
    (stale / "people_msgs" / "package.xml").write_text("<package>edited</package>\n")
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 1
    assert (stale / "people_msgs" / "package.xml").read_text() == "<package>edited</package>\n"


def test_people_msgs_patch_finishes_a_checkout_that_lost_its_git_dir(tmp_path):
    stale = tmp_path / "src" / "deps" / "hunav" / "people_msgs"
    (stale / "people_msgs").mkdir(parents=True)
    (stale / "people_msgs" / "package.xml").write_text("<package/>\n")
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 0, done.stderr
    assert not stale.exists()


def test_people_msgs_patch_keeps_a_checkout_git_cannot_read(tmp_path):
    stale = tmp_path / "src" / "deps" / "hunav" / "people_msgs"
    stale.mkdir(parents=True)
    (stale / ".git").write_text("gitdir: /nonexistent\n")
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 1
    assert stale.exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes files from read-only directories")
def test_people_msgs_patch_fails_when_the_removal_is_partial(tmp_path):
    stale = tmp_path / "src" / "deps" / "hunav" / "people_msgs"
    locked = stale / "people_msgs" / "locked"
    locked.mkdir(parents=True)
    (locked / "f").write_text("")
    locked.chmod(0o555)
    try:
        done = run_people_msgs_patch(tmp_path)
    finally:
        locked.chmod(0o755)
    assert done.returncode != 0
    assert "removed" not in done.stdout
    assert (locked / "f").exists()


def test_people_msgs_patch_does_not_read_a_parent_repo_through_a_broken_git_dir(tmp_path):
    git(tmp_path, "init", "-q")
    stale = tmp_path / "src" / "deps" / "hunav" / "people_msgs"
    (stale / ".git").mkdir(parents=True)
    (stale / "people_msgs").mkdir()
    (stale / "people_msgs" / "package.xml").write_text("<package/>\n")
    git(tmp_path, "add", "src")
    git(tmp_path, "update-ref", "refs/remotes/origin/jazzy", commit(tmp_path, "parent holds the stale tree"))
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 1
    assert (stale / "people_msgs" / "package.xml").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root removes files from read-only directories")
def test_people_msgs_patch_keeps_the_clone_while_the_build_dir_resists(tmp_path):
    stale = stale_people_msgs(tmp_path)
    locked = tmp_path / "build" / "people_msgs" / "locked"
    locked.mkdir()
    (locked / "CMakeCache.txt").write_text("")
    locked.chmod(0o555)
    try:
        done = run_people_msgs_patch(tmp_path)
    finally:
        locked.chmod(0o755)
    assert done.returncode != 0
    assert (stale / "people_msgs" / "package.xml").exists()


def test_people_msgs_patch_refuses_a_symlinked_new_path(tmp_path):
    stale = stale_people_msgs(tmp_path)
    (tmp_path / "src" / "deps" / "people_msgs").symlink_to("hunav/people_msgs")
    done = run_people_msgs_patch(tmp_path)
    assert done.returncode == 1
    assert "symlink" in done.stderr
    assert (stale / "people_msgs" / "package.xml").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists unreadable directories")
def test_unlistable_patches_dir_is_reported(tmp_path, capsys):
    patches_dir = patch(tmp_path, "2026-01-01-ok", "exit 0").parent
    patches_dir.chmod(0o000)
    try:
        assert apply_patches(str(tmp_path), ENV) == ["_meta/patches"]
    finally:
        patches_dir.chmod(0o755)
    assert "patches: " in capsys.readouterr().err


def test_people_msgs_patch_leaves_a_current_workspace_alone(tmp_path):
    for kept in (tmp_path / "src" / "deps" / "people_msgs" / ".git", tmp_path / "build" / "people_msgs"):
        kept.mkdir(parents=True)
    env = {**ENV, "ARENA_WS_DIR": str(tmp_path)}
    done = subprocess.run([str(PEOPLE_MSGS_PATCH)], env=env, capture_output=True, text=True, check=True)
    assert done.stdout == ""
    assert (tmp_path / "build" / "people_msgs").is_dir()
    assert (tmp_path / "src" / "deps" / "people_msgs" / ".git").is_dir()


def restarted_skips_in_child(env: dict[str, str]) -> list[str]:
    code = "import os; from arena_cli.pull import restarted_skips; print(restarted_skips(), os.environ.get('ARENA_UPDATE_RESTARTED'))"
    child = subprocess.run([sys.executable, "-c", code], env={**env, "PYTHONPATH": str(Path(__file__).parents[2])}, capture_output=True, text=True, check=True)
    return child.stdout.splitlines()


def test_restart_reruns_the_update_verb_with_its_arguments(tmp_path):
    cmd, _ = restart_command(["--foo", "bar"], [])
    assert cmd == [sys.executable, "-m", "arena_cli", "update", "--foo", "bar"]


def test_restart_carries_skipped_steps_and_consumes_the_marker(tmp_path):
    _, env = restart_command([], ["apt update", "Arena pull"])
    assert restarted_skips_in_child(env) == ["['apt update', 'Arena pull'] None"]


def test_restart_with_nothing_skipped_is_still_a_restart(tmp_path):
    _, env = restart_command([], [])
    assert restarted_skips_in_child(env) == ["[] None"]


def test_a_fresh_update_is_not_a_restart(tmp_path):
    env = {k: v for k, v in ENV.items() if k != "ARENA_UPDATE_RESTARTED"}
    assert restarted_skips_in_child(env) == ["None None"]
