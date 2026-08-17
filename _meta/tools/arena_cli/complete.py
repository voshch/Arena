"""Shell completion endpoint: words in, candidate lines out. Pure stdlib, never imports ROS."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from common import Verb

SCHEMA = 1
LOCK_MAX_AGE_S = 120.0


def _table(values: object) -> dict[str, str]:
    if isinstance(values, Mapping):
        return {str(k): str(v) for k, v in values.items()}
    return {str(v): "" for v in values}


@dataclasses.dataclass
class Nothing:
    """No candidates."""


@dataclasses.dataclass
class Files:
    """Native filename completion."""


@dataclasses.dataclass
class Static:
    """Fixed values, optionally with descriptions. A callable is resolved on demand."""

    values: dict[str, str] | Callable[[], object]

    def __init__(self, values: object) -> None:
        self.values = values if callable(values) else _table(values)

    def table(self) -> dict[str, str]:
        return _table(self.values()) if callable(self.values) else self.values


@dataclasses.dataclass
class Manifest:
    """Values under manifest['values'][key], produced by manifest.py."""

    key: str


@dataclasses.dataclass
class Flags:
    """--flags with descriptions. `valued` flags take a free-form next word."""

    flags: dict[str, str]
    valued: tuple[str, ...] = ()

    def __init__(self, flags: object, valued: tuple[str, ...] = ()) -> None:
        self.flags = _table(flags)
        self.valued = valued


@dataclasses.dataclass
class Kv:
    """key:=value tokens. Each key maps to a value source, None for free-form."""

    keys: dict[str, Source | None]


@dataclasses.dataclass
class LaunchArgs:
    """Every argument of a launch file (via the manifest). `values` overrides per-arg value sources."""

    package: str
    file: str
    values: dict[str, Source] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Union:
    """Any of the given specs, position-independent."""

    specs: tuple[Spec, ...]

    def __init__(self, *specs: Spec) -> None:
        self.specs = specs


@dataclasses.dataclass
class Sub:
    """First word selects a subverb (Spec or Verb), the rest is delegated. `extra` adds first-position candidates, `fallback` handles an unknown first word."""

    subs: Mapping[str, Spec | Verb] | Callable[[], Mapping[str, Spec | Verb]]
    extra: Spec | None = None
    fallback: Spec | None = None

    def table(self) -> Mapping[str, Spec | Verb]:
        return self.subs() if callable(self.subs) else self.subs


@dataclasses.dataclass
class Packages:
    """Workspace package names from a src/**/package.xml walk."""


Source = Static | Manifest | Files | Nothing
Spec = Source | Flags | Kv | LaunchArgs | Union | Sub | Packages


def walk(spec: Spec | None) -> Iterator[Spec]:
    """Every spec nested under `spec`, itself included."""
    if spec is None:
        return
    yield spec
    if isinstance(spec, Union):
        for s in spec.specs:
            yield from walk(s)
    elif isinstance(spec, Kv):
        for s in spec.keys.values():
            yield from walk(s)
    elif isinstance(spec, LaunchArgs):
        for s in spec.values.values():
            yield from walk(s)
    elif isinstance(spec, Sub):
        for target in spec.table().values():
            yield from walk(_spec_of(target))
        yield from walk(spec.extra)
        yield from walk(spec.fallback)


def _spec_of(target: Spec | Verb) -> Spec | None:
    from common import Verb

    return target.complete if isinstance(target, Verb) else target



@dataclasses.dataclass
class Result:
    items: list[tuple[str, str, str]] = dataclasses.field(default_factory=list)
    prefix: str = ""
    files: bool = False
    stop: bool = False

    def add(self, values: Mapping[str, str], group: str = "") -> None:
        self.items += [(v, d, group) for v, d in values.items()]

    def merge(self, other: Result) -> None:
        self.items += other.items
        self.prefix = self.prefix or other.prefix
        self.files = self.files or other.files
        self.stop = self.stop or other.stop


class Context:
    def __init__(self, ws: str, manifest: dict) -> None:
        self.ws = ws
        self.manifest = manifest

    def values(self, key: str) -> dict[str, str] | None:
        found = self.manifest.get("values", {}).get(key)
        return None if found is None else _table(found)

    def launch_args(self, package: str, file: str) -> dict[str, dict[str, str]]:
        return self.manifest.get("launch", {}).get(f"{package}/{file}", {})


def _is_bool(default: str) -> bool:
    return default.lower() in ("true", "false")


def _expand_value(source: Source | None, key: str, val: str, ctx: Context) -> Result:
    r = Result(prefix=key + ":=")
    if source is not None:
        r.merge(_expand(source, [], val, ctx))
    return r


def _expand(spec: Spec, words: list[str], cur: str, ctx: Context) -> Result:
    if isinstance(spec, Kv | LaunchArgs):
        return _expand_kv(spec, cur, ctx)
    if isinstance(spec, Union):
        r = Result()
        for s in spec.specs:
            r.merge(_expand(s, words, cur, ctx))
        return r
    if isinstance(spec, Sub):
        return _expand_sub(spec, words, cur, ctx)
    return _expand_flat(spec, words, cur, ctx)


def _expand_flat(spec: Source | Flags | Packages, words: list[str], cur: str, ctx: Context) -> Result:
    r = Result()
    if isinstance(spec, Nothing) or ":=" in cur:
        return r
    if isinstance(spec, Files):
        r.files = True
    elif isinstance(spec, Static):
        r.add(spec.table())
    elif isinstance(spec, Manifest):
        r.add(ctx.values(spec.key) or {})
    elif isinstance(spec, Flags):
        if words and words[-1] in spec.valued:
            r.stop = True
        else:
            r.add(spec.flags)
    elif isinstance(spec, Packages):
        r.add({p: "" for p in packages(ctx.ws)})
    return r


def _expand_kv(spec: Kv | LaunchArgs, cur: str, ctx: Context) -> Result:
    if isinstance(spec, Kv):
        keys: dict[str, str] = {k: "" for k in spec.keys}
        sources: Mapping[str, Source | None] = spec.keys
    else:
        args = ctx.launch_args(spec.package, spec.file) or {k: {} for k in spec.values}
        keys = {k: a.get("desc", "") for k, a in args.items()}
        sources = {k: _launch_source(spec, k, a, ctx) for k, a in args.items()}
    if ":=" not in cur:
        r = Result()
        r.add({k + ":=": d for k, d in keys.items()})
        return r
    key, _, val = cur.partition(":=")
    if key not in sources:
        return Result()
    return _expand_value(sources[key], key, val, ctx)


def _expand_sub(spec: Sub, words: list[str], cur: str, ctx: Context) -> Result:
    r = Result()
    table = spec.table()
    if not words:
        if ":=" not in cur:
            r.add(_sub_names(table))
        if spec.extra is not None:
            r.merge(_expand(spec.extra, [], cur, ctx))
    elif words[0] in table:
        r.merge(_expand(_spec_of(table[words[0]]) or Nothing(), words[1:], cur, ctx))
    elif spec.fallback is not None:
        r.merge(_expand(spec.fallback, words[1:], cur, ctx))
    return r


def _launch_source(spec: LaunchArgs, key: str, arg: dict[str, str], ctx: Context) -> Source | None:
    if key in spec.values:
        return spec.values[key]
    if ctx.values(key) is not None:
        return Manifest(key)
    if _is_bool(arg.get("default", "")):
        return Static(["true", "false"])
    return None


def _sub_names(table: Mapping[str, Spec | Verb]) -> dict[str, str]:
    from common import Verb

    out: dict[str, str] = {}
    for name, target in table.items():
        if isinstance(target, Verb):
            if target.hidden:
                continue
            out[name] = target.short
        else:
            out[name] = ""
    return out



def packages(ws: str) -> list[str]:
    root = os.path.join(ws, "src")
    names: list[str] = []
    seen: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in seen or "COLCON_IGNORE" in filenames or "AMENT_IGNORE" in filenames:
            dirnames[:] = []
            continue
        seen.add(real)
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if "package.xml" not in filenames:
            continue
        dirnames[:] = []
        try:
            with open(os.path.join(dirpath, "package.xml")) as f:
                m = re.search(r"<name>\s*([^<\s]+)\s*</name>", f.read())
        except OSError:
            continue
        if m:
            names.append(m.group(1))
    return sorted(set(names))



def cache_path(ws: str) -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    digest = hashlib.sha1(ws.encode()).hexdigest()[:12]
    return os.path.join(base, "arena", f"completion-{digest}.json")


def stamp(ws: str, deps: list[str]) -> float:
    best = 0.0
    for p in (os.path.join(ws, "install"), os.path.join(ws, "install", "local_setup.sh"), *deps):
        try:
            best = max(best, os.stat(p).st_mtime)
        except OSError:
            continue
    return best


def load_manifest(ws: str) -> dict:
    path = cache_path(ws)
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    if isinstance(data, dict) and data.get("schema") == SCHEMA and stamp(ws, data.get("deps", [])) == data.get("stamp"):
        return data
    _spawn_generator(path)
    return {}


def _spawn_generator(path: str) -> None:
    import subprocess

    lock = path + ".lock"
    try:
        if time.time() - os.stat(lock).st_mtime < LOCK_MAX_AGE_S:
            return
    except OSError:
        pass
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(lock, "w"):
            pass
    except OSError:
        return
    subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.py"), "--out", path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def invalidate(ws: str) -> None:
    with contextlib.suppress(OSError):
        os.remove(cache_path(ws))


def refresh(ws: str) -> int:
    import subprocess

    return subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.py"), "--out", cache_path(ws)], check=False).returncode



def complete(words: list[str], verbs: Mapping[str, Verb], sections: Mapping[str, list[str]], ws: str) -> list[str]:
    """Candidate lines for `arena <words>`, the last word being the one under the cursor (empty words means the verb position)."""
    ctx = Context(ws, load_manifest(ws))
    cur = words[-1] if words else ""
    r = Result()
    if len(words) <= 1:
        placed: set[str] = set()
        for title, names in sections.items():
            r.add({n: verbs[n].short for n in names if n in verbs and not verbs[n].hidden}, title)
            placed.update(names)
        r.add({n: v.short for n, v in verbs.items() if n not in placed and not v.hidden}, "Other")
    else:
        v = verbs.get(words[0])
        if v is not None and v.complete is not None:
            r = _expand(v.complete, words[1:-1], cur, ctx)
    return render(r, cur)


def _fold_key(value: str, desc: str, typed: str) -> tuple[str, str, bool]:
    if not value.endswith(":=") or ":" in typed:
        return value, desc, False
    parts = value[:-2].split(".")
    depth = typed.count(".") + 1
    if ".".join(parts[:depth]) == typed:
        depth += 1
    if len(parts) <= depth:
        return value, desc, False
    return ".".join(parts[:depth]), "", True


def render(r: Result, cur: str) -> list[str]:
    if r.stop:
        return []
    local = cur[len(r.prefix):] if cur.startswith(r.prefix) else cur
    seen: set[str] = set()
    items = []
    folded: set[str] = set()
    for value, desc, group in r.items:
        value, desc, fold = _fold_key(value, desc, local)
        if value.startswith(local) and value not in seen:
            seen.add(value)
            items.append((value, desc, group))
            if fold:
                folded.add(value)
    items = [i for i in items if not (i[0].endswith(":=") and i[0][:-2] in folded)]
    out: list[str] = []
    if r.prefix:
        out.append(f"@prefix={r.prefix}")
    if items and all(v.endswith(":=") or v in folded for v, _, _ in items):
        out.append("@nospace")
    if r.files:
        out.append("@files")
    group = None
    for value, desc, g in items:
        if g != group:
            group = g
            out.append(f"@group={g or 'arena'}")
        out.append(f"{value}\t{desc}" if desc else value)
    return out
