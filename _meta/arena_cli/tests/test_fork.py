"""Tests for fork releases, pruning, domain allocation and lane arguments against real trees and git repos."""

import json
import os
import pathlib
import subprocess
from collections.abc import Iterator

import pytest

from arena_cli import fork
from arena_cli.common import CLIError

GIT_ENV = {
    **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], env=GIT_ENV, capture_output=True, text=True, check=True).stdout.strip()


def write(path: pathlib.Path, text: str) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def ws(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """A workspace with a committed repo in src, a planner venv in build, an install and a venv, set as ARENA_WS_DIR."""
    root = tmp_path / "ws"
    repo = root / "src" / "Arena"
    write(repo / "pkg" / "a.py", "a = 1\n")
    write(repo / "pkg" / "b.py", "b = 2\n")
    write(repo / "_assets" / "big.bin", "asset\n")
    write(repo / ".venv" / "ignored", "x\n")
    git(repo, "init", "-q", "-b", "jazzy")
    git(repo, "add", "pkg")
    git(repo, "commit", "-q", "-m", "init")
    write(root / "build" / "planner" / "venv" / "pyvenv.cfg", "home = /usr\n")
    write(root / "build" / "planner" / "venv" / "lib" / "torch.so", "torch\n")
    write(root / "build" / "plain" / "colcon_build.rc", "0\n")
    write(root / "build" / ".uv-cache" / "blob", "cache\n")
    write(root / "install" / "pkg" / "share" / "pkg" / "package.xml", "<package/>\n")
    write(root / "venv" / "bin" / "python3", "py\n")
    (root / ".uv-python").mkdir()
    keys = ("ARENA_WS_DIR", "ARENA_VENV_DIR", "ARENA_FORK", "ARENA_FORK_DOMAIN_BASE", "ARENA_IMAGE_TAG")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({"ARENA_WS_DIR": str(root), "ARENA_VENV_DIR": str(root / "venv")})
    os.environ.pop("ARENA_FORK", None)
    os.environ.pop("ARENA_FORK_DOMAIN_BASE", None)
    os.environ["ARENA_IMAGE_TAG"] = saved["ARENA_IMAGE_TAG"] or "arena-test"
    try:
        yield root
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def releases(ws: pathlib.Path) -> pathlib.Path:
    return ws / "build" / ".forks" / "releases"


def same_inode(a: pathlib.Path, b: pathlib.Path) -> bool:
    return os.path.samestat(os.lstat(a), os.lstat(b))


def add_fork(ws: pathlib.Path, name: str, release: str, domain: int, kept: bool = True) -> None:
    root = ws / "build" / ".forks" / name
    write(root / "release", f"{release}\n")
    write(root / "domain", f"{domain}\n")
    if kept:
        (root / "kept").touch()


def test_freeze_hardlinks_files_unchanged_since_previous_release(tmp_path: pathlib.Path) -> None:
    src, prev, dst = tmp_path / "src", tmp_path / "prev", tmp_path / "dst"
    write(src / "same.txt", "same\n")
    write(src / "edited.txt", "new content\n")
    fork._freeze(str(src), str(prev), None, fork._walk(str(src)), lambda rel: False)
    write(src / "edited.txt", "newer content, longer\n")
    part = fork._freeze(str(src), str(dst), str(prev), fork._walk(str(src)), lambda rel: False)
    assert same_inode(dst / "same.txt", prev / "same.txt")
    assert not same_inode(dst / "edited.txt", prev / "edited.txt")
    assert (dst / "edited.txt").read_text() == "newer content, longer\n"
    assert (part.files, part.changed, part.bytes) == (2, 1, len("newer content, longer\n"))


def test_freeze_detects_same_size_edit_by_content(tmp_path: pathlib.Path) -> None:
    src, prev, dst = tmp_path / "src", tmp_path / "prev", tmp_path / "dst"
    write(src / "f.txt", "aaaa\n")
    fork._freeze(str(src), str(prev), None, fork._walk(str(src)), lambda rel: False)
    write(src / "f.txt", "bbbb\n")
    os.utime(src / "f.txt", ns=(0, 0))
    part = fork._freeze(str(src), str(dst), str(prev), fork._walk(str(src)), lambda rel: False)
    assert (dst / "f.txt").read_text() == "bbbb\n"
    assert part.changed == 1


def test_freeze_links_source_where_asked_and_recreates_symlinks(tmp_path: pathlib.Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(src / "venv" / "lib.so", "lib\n")
    write(src / "plain.txt", "plain\n")
    (src / "link").symlink_to("plain.txt")
    part = fork._freeze(str(src), str(dst), None, fork._walk(str(src)), lambda rel: rel.startswith("venv"))
    assert same_inode(dst / "venv" / "lib.so", src / "venv" / "lib.so")
    assert not same_inode(dst / "plain.txt", src / "plain.txt")
    assert os.readlink(dst / "link") == "plain.txt"
    assert (part.files, part.changed, part.bytes) == (3, 3, len("plain\n"))


def test_freeze_without_destination_only_counts(tmp_path: pathlib.Path) -> None:
    src, prev = tmp_path / "src", tmp_path / "prev"
    write(src / "f.txt", "f\n")
    fork._freeze(str(src), str(prev), None, fork._walk(str(src)), lambda rel: False)
    write(src / "g.txt", "g\n")
    part = fork._freeze(str(src), None, str(prev), fork._walk(str(src)), lambda rel: False)
    assert (part.files, part.changed) == (2, 1)
    assert sorted(os.listdir(tmp_path)) == ["prev", "src"]


def test_freeze_leaves_pycache_and_git_out_of_counts(tmp_path: pathlib.Path) -> None:
    src = tmp_path / "src"
    write(src / "__pycache__" / "m.pyc", "c\n")
    write(src / ".git" / "HEAD", "ref\n")
    write(src / "m.py", "m\n")
    part = fork._freeze(str(src), str(tmp_path / "dst"), None, fork._walk(str(src)), lambda rel: False)
    assert (part.files, part.changed) == (1, 1)
    assert (tmp_path / "dst" / "__pycache__" / "m.pyc").is_file()


def test_walk_prunes_tool_dirs_and_top_level_skips(tmp_path: pathlib.Path) -> None:
    for rel in (".venv/x", ".ruff_cache/x", "Arena/_assets/x", "Arena/keep/x", "Arena/.pytest_cache/x", "deep/Arena/_assets/x"):
        write(tmp_path / rel, "x\n")
    files = sorted(rel for rel, is_dir in fork._walk(str(tmp_path), skip_top=frozenset({"Arena/_assets"})) if not is_dir)
    assert files == ["Arena/keep/x", "deep/Arena/_assets/x"]


def test_without_git_stores_drops_objects_and_worktrees_and_records_stores(tmp_path: pathlib.Path) -> None:
    repo = tmp_path / "Arena"
    sub = repo / "sub"
    git(tmp_path, "init", "-q", str(repo))
    git(tmp_path, "init", "-q", str(sub))
    write(repo / ".git" / "worktrees" / "wt" / "HEAD", "x\n")
    write(repo / "objects" / "plain.txt", "not a git store\n")
    stores: list[str] = []
    kept = [rel for rel, _ in fork._without_git_stores(str(tmp_path), fork._walk(str(tmp_path)), stores)]
    assert sorted(stores) == ["Arena/.git/objects", "Arena/sub/.git/objects"]
    assert "Arena/.git/objects" in kept
    assert not any(rel.startswith(("Arena/.git/objects/", "Arena/.git/worktrees")) for rel in kept)
    assert "Arena/objects/plain.txt" in kept
    assert "Arena/.git/HEAD" in kept


def test_release_freezes_every_part_and_reuses_a_matching_release(ws: pathlib.Path) -> None:
    first = fork._release()
    rel = releases(ws) / first
    meta = json.loads((rel / "release.json").read_text())
    assert fork._latest() == first
    assert (meta["parent"], meta["format"]) == (None, fork._FORMAT)
    assert meta["repos"] == {"Arena": {"head": git(ws / "src" / "Arena", "rev-parse", "HEAD"), "dirty": False}}
    assert (rel / "src" / "Arena" / "pkg" / "a.py").read_text() == "a = 1\n"
    assert os.listdir(rel / "src" / "Arena" / "_assets") == []
    assert not (rel / "src" / "Arena" / ".venv" / "ignored").exists()
    assert os.listdir(rel / "src" / "Arena" / ".git" / "objects") == ["info"]
    assert (rel / "src" / "Arena" / ".git" / "objects" / "info" / "alternates").read_text() == f"{fork._DEV_SRC}/Arena/.git/objects\n"
    assert same_inode(rel / "build" / "planner" / "venv" / "lib" / "torch.so", ws / "build" / "planner" / "venv" / "lib" / "torch.so")
    assert not same_inode(rel / "build" / "plain" / "colcon_build.rc", ws / "build" / "plain" / "colcon_build.rc")
    assert not (rel / "build" / ".uv-cache").exists()
    assert not (rel / "build" / ".forks").exists()
    assert (rel / "src" / "Arena" / ".venv" / "bin" / "python3").read_text() == "py\n"
    assert (rel / "install" / "pkg" / "share" / "pkg" / "package.xml").is_file()
    assert (rel / ".built").is_file()
    assert fork._release() == first
    assert fork._release_ids() == [first]


def test_release_after_edit_links_unchanged_files_and_prunes_the_unused_parent(ws: pathlib.Path) -> None:
    first = fork._release()
    write(ws / "src" / "Arena" / "pkg" / "a.py", "a = 10\n")
    second = fork._release()
    assert second != first
    rel = releases(ws) / second
    assert json.loads((rel / "release.json").read_text())["parent"] == first
    assert json.loads((rel / "release.json").read_text())["repos"]["Arena"]["dirty"]
    assert (rel / "src" / "Arena" / "pkg" / "a.py").read_text() == "a = 10\n"
    assert os.stat(rel / "src" / "Arena" / "pkg" / "b.py").st_nlink == 1
    assert fork._release_ids() == [second]
    assert fork._latest() == second


def test_release_after_edit_shares_inodes_with_a_parent_a_fork_holds(ws: pathlib.Path) -> None:
    first = fork._release()
    add_fork(ws, "mine", first, 20)
    write(ws / "src" / "Arena" / "pkg" / "a.py", "a = 10\n")
    second = fork._release()
    assert fork._release_ids() == [first, second]
    old, new = releases(ws) / first / "src" / "Arena" / "pkg", releases(ws) / second / "src" / "Arena" / "pkg"
    assert same_inode(old / "b.py", new / "b.py")
    assert not same_inode(old / "a.py", new / "a.py")
    assert (old / "a.py").read_text() == "a = 1\n"


def test_release_sees_untracked_files_and_deletions(ws: pathlib.Path) -> None:
    first = fork._release()
    add_fork(ws, "mine", first, 20)
    write(ws / "src" / "Arena" / "pkg" / "new.py", "n = 1\n")
    second = fork._release()
    assert second != first
    assert (releases(ws) / second / "src" / "Arena" / "pkg" / "new.py").is_file()
    add_fork(ws, "mine2", second, 21)
    (ws / "build" / "plain" / "colcon_build.rc").unlink()
    third = fork._release()
    assert third not in (first, second)
    assert not (releases(ws) / third / "build" / "plain" / "colcon_build.rc").exists()


def test_release_follows_a_new_commit(ws: pathlib.Path) -> None:
    first = fork._release()
    add_fork(ws, "mine", first, 20)
    git(ws / "src" / "Arena", "commit", "-q", "--allow-empty", "-m", "next")
    second = fork._release()
    assert second != first
    assert json.loads((releases(ws) / second / "release.json").read_text())["repos"]["Arena"]["head"] == git(ws / "src" / "Arena", "rev-parse", "HEAD")


def test_release_clears_an_abandoned_freeze(ws: pathlib.Path) -> None:
    first = fork._release()
    write(releases(ws) / ".tmp-20200101-000000" / "src" / "half.txt", "half\n")
    write(ws / "src" / "Arena" / "pkg" / "a.py", "a = 10\n")
    second = fork._release()
    assert second != first
    assert sorted(os.listdir(releases(ws))) == [".lock", second, "latest"]


def test_prune_keeps_latest_and_releases_forks_use(ws: pathlib.Path) -> None:
    for rid in ("20200101-000000", "20200101-000001", "20200101-000002"):
        (releases(ws) / rid).mkdir(parents=True)
    os.symlink("20200101-000002", releases(ws) / "latest")
    add_fork(ws, "old", "20200101-000000", 20)
    fork._prune()
    assert fork._release_ids() == ["20200101-000000", "20200101-000002"]


def test_free_domain_skips_domains_forks_hold(ws: pathlib.Path) -> None:
    add_fork(ws, "a", "r", 20)
    add_fork(ws, "b", "r", 21, kept=False)
    add_fork(ws, "c", "r", 23)
    used = {int(d) for d in fork._used_domains() if d.isdigit()}
    domain = fork._free_domain()
    assert domain >= 22
    assert domain not in {20, 21, 23} | used
    assert all(d in {20, 21, 23} | used for d in range(20, domain))


def test_free_domain_starts_at_the_configured_base_and_stops_at_232(ws: pathlib.Path) -> None:
    os.environ["ARENA_FORK_DOMAIN_BASE"] = "200"
    used = {int(d) for d in fork._used_domains() if d.isdigit()}
    assert fork._free_domain() == min(set(range(200, 233)) - used)
    os.environ["ARENA_FORK_DOMAIN_BASE"] = "233"
    with pytest.raises(CLIError, match="no free ROS domain"):
        fork._free_domain()


def test_forks_lists_complete_fork_dirs_by_kind(ws: pathlib.Path) -> None:
    add_fork(ws, "kept1", "r1", 20)
    add_fork(ws, "p1", "r1", 21, kept=False)
    write(ws / "build" / ".forks" / "half" / "release", "r1\n")
    add_fork(ws, "Bad", "r1", 22)
    (releases(ws)).mkdir(parents=True)
    forks = fork._forks()
    assert list(forks) == ["kept1", "p1"]
    assert {n: (f["kind"], f["release"], f["domain"]) for n, f in forks.items()} == {"kept1": ("kept", "r1", "20"), "p1": ("pool", "r1", "21")}


def test_fork_verbs_refuse_inside_a_fork(ws: pathlib.Path) -> None:
    os.environ["ARENA_FORK"] = "p1"
    for verb in (fork.new, fork.ls, fork.down, fork.exec_):
        with pytest.raises(CLIError, match="this is fork p1"):
            verb([])


@pytest.mark.parametrize("name", ["Upper", "-dash", "releases", "a" * 64, "sp ace"])
def test_new_rejects_bad_fork_names(ws: pathlib.Path, name: str) -> None:
    with pytest.raises(CLIError, match="must be 1-63 lowercase"):
        fork.new([name])


def test_down_rejects_flags_and_unknown_forks(ws: pathlib.Path) -> None:
    add_fork(ws, "kept1", "r1", 20)
    with pytest.raises(CLIError, match="unexpected argument --all"):
        fork.down(["--all"])
    with pytest.raises(CLIError, match="no fork nope, forks: kept1"):
        fork.down(["nope"])


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--suite", "s", "--lanes", "3", "--x"], (3, ["--suite", "s", "--x"])),
        (["--lanes=2", "--suite", "s"], (2, ["--suite", "s"])),
        (["--suite", "s"], (None, ["--suite", "s"])),
    ],
)
def test_split_lanes_takes_the_lane_count_out(argv: list[str], expected: tuple[int | None, list[str]]) -> None:
    assert fork.split_lanes(argv) == expected


@pytest.mark.parametrize("argv", [["--lanes"], ["--lanes", "0"], ["--lanes=x"], ["--lanes", "-1"]])
def test_split_lanes_rejects_bad_counts(argv: list[str]) -> None:
    with pytest.raises(CLIError, match="--lanes takes a lane count"):
        fork.split_lanes(argv)


def test_flag_reads_both_spellings_and_needs_a_value() -> None:
    assert fork._flag(["--run-id", "r1"], "--run-id") == "r1"
    assert fork._flag(["--run-id=r2"], "--run-id") == "r2"
    assert fork._flag(["--suite", "s"], "--run-id") is None
    with pytest.raises(CLIError, match="--resume needs a value"):
        fork._flag(["--resume", "--x"], "--resume")


def test_run_shared_refuses_explicit_shared_and_a_bare_resume(ws: pathlib.Path) -> None:
    with pytest.raises(CLIError, match="--lanes sets --shared itself"):
        fork.run_shared(2, ["--shared", "tok"])
    with pytest.raises(CLIError, match="--resume needs a value"):
        fork.run_shared(2, ["--resume"])
