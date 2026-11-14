"""Garbage collection of the planner venv store in $ARENA_WS_DIR/.uv/venvs."""

import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

GRACE_S = 86400


def _dirs(path: Path) -> Iterator[Path]:
    if path.is_dir():
        yield from (Path(e.path) for e in os.scandir(path) if e.is_dir(follow_symlinks=False))


def _builds(ws: Path) -> Iterator[Path]:
    build = ws / "build"
    yield build
    forks = build / ".forks"
    for release in _dirs(forks / "releases"):
        yield release / "build"
    for fork in _dirs(forks):
        if fork.name != "releases":
            yield fork / "upper" / "build"


def references(ws: Path) -> set[str]:
    """Store entries named by a venv symlink in the dev build, a release or a fork's upper layer."""
    refs: set[str] = set()
    for build in _builds(ws):
        for pkg in _dirs(build):
            venv = pkg / "venv"
            if venv.is_symlink():
                refs.add(os.path.basename(os.readlink(venv)))
    return refs


def gc(ws: Path) -> list[str]:
    """Stamp unreferenced store entries, delete those stamped over a day ago and still unreferenced, and return their names."""
    store = ws / ".uv" / "venvs"
    if os.environ.get("ARENA_FORK") or not store.is_dir():
        return []
    try:
        refs = references(ws)
    except OSError as e:
        print(f"venv store: gc skipped, {e}", flush=True)
        return []
    stamps = store / ".gc"
    stamps.mkdir(exist_ok=True)
    old = time.time() - GRACE_S
    removed: list[str] = []
    for entry in _dirs(store):
        name = entry.name
        if name.startswith(".tmp-") and entry.stat().st_mtime < old:
            shutil.rmtree(entry)
        if name.startswith("."):
            continue
        stamp = stamps / name
        if name in refs:
            stamp.unlink(missing_ok=True)
        elif not stamp.exists():
            stamp.touch()
        elif stamp.stat().st_mtime < old:
            shutil.rmtree(entry)
            stamp.unlink()
            (store / ".locks" / name).unlink(missing_ok=True)
            removed.append(name)
    for stamp in stamps.iterdir():
        if not (store / stamp.name).exists():
            stamp.unlink()
    return removed
