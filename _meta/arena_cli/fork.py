"""Forks: isolated containers on frozen releases of the dev tree."""

import dataclasses
import datetime
import fcntl
import filecmp
import json
import os
import pathlib
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator

from arena_cli import features
from arena_cli.common import CLIError, Verb, _env, make_verb

DESCRIPTION = """Isolated containers on snapshots of the dev tree, each on its own ROS domain.

`source arena --fork [<fork>]` enters one, forking it first if it does not exist."""

_SKIP_DIRS = {".venv", ".ruff_cache", ".pytest_cache", ".hypothesis", ".mypy_cache"}
_RCFILE = "/opt/arena_ws/src/Arena/_meta/docker/features/docker/rcfile"
_FORK_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,62}")
_OVERLAYS = ("src", "build", "install", "venv", "uv-python")
_FORMAT = 2
_DEV_SRC = "/opt/arena_ws/.dev-src"
_SOURCED = 'source /opt/arena_ws/source >/dev/null; eval "$1"'


@dataclasses.dataclass
class _Part:
    """File, change and byte counts of one frozen part."""

    files: int = 0
    changed: int = 0
    bytes: int = 0


def _ws() -> str:
    return _env("ARENA_WS_DIR")


def _root() -> str:
    return os.path.join(_ws(), "build", ".forks")


def _releases() -> str:
    return os.path.join(_root(), "releases")


def _latest() -> str | None:
    link = os.path.join(_releases(), "latest")
    return os.readlink(link) if os.path.islink(link) else None


def _release_ids() -> list[str]:
    root = _releases()
    if not os.path.isdir(root):
        return []
    return sorted(e for e in os.listdir(root) if not e.startswith(".") and e != "latest" and os.path.isdir(os.path.join(root, e)))


def _meta(release: str) -> dict:
    path = os.path.join(_releases(), release, "release.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _walk(root: str, rel: str = "", skip_top: frozenset[str] = frozenset()) -> Iterator[tuple[str, bool]]:
    """Yield (relpath, is_dir) under root without following symlinks, pruning _SKIP_DIRS."""
    with os.scandir(os.path.join(root, rel)) as it:
        for e in it:
            r = os.path.join(rel, e.name)
            if e.is_dir(follow_symlinks=False):
                if e.name in _SKIP_DIRS or (not rel and e.name in skip_top) or r in skip_top:
                    continue
                yield r, True
                yield from _walk(root, r, skip_top)
            else:
                yield r, False


def _freeze(src: str, dst: str | None, prev: str | None, entries: Iterable[tuple[str, bool]], link_source: Callable[[str], bool]) -> _Part:
    """Copy src into dst, hardlinking files unchanged since prev or, where link_source says so, from src. A dst of None only counts."""
    part = _Part()
    if dst is not None:
        os.makedirs(dst, exist_ok=True)
    for rel, is_dir in entries:
        s = os.path.join(src, rel)
        d = os.path.join(dst, rel) if dst is not None else ""
        try:
            ss = os.lstat(s)
        except FileNotFoundError:
            continue
        if is_dir:
            if dst is not None:
                os.makedirs(d, exist_ok=True)
                os.chmod(d, stat.S_IMODE(ss.st_mode))
            continue
        parts = rel.split(os.sep)
        counted = "__pycache__" not in parts and ".git" not in parts
        part.files += counted
        p = os.path.join(prev, rel) if prev else None
        try:
            ps = os.lstat(p) if p else None
        except FileNotFoundError:
            ps = None
        if stat.S_ISLNK(ss.st_mode):
            target = os.readlink(s)
            if dst is not None:
                os.symlink(target, d)
            part.changed += counted and not (ps is not None and stat.S_ISLNK(ps.st_mode) and os.readlink(p) == target)
        elif not stat.S_ISREG(ss.st_mode):
            part.files -= counted
        elif link_source(rel):
            if dst is not None:
                os.link(s, d)
            part.changed += counted and not (ps is not None and os.path.samestat(ps, ss))
        elif ps is not None and stat.S_ISREG(ps.st_mode) and ps.st_size == ss.st_size and (ps.st_mtime_ns == ss.st_mtime_ns or filecmp.cmp(p, s, shallow=False)):
            if dst is not None:
                os.link(p, d)
        else:
            if dst is not None:
                shutil.copy2(s, d)
            part.changed += counted
            part.bytes += ss.st_size
    return part


def _repos(src: str) -> dict[str, dict[str, object]]:
    """HEAD and dirty state of every git repo under src."""
    out: dict[str, dict[str, object]] = {}
    for rel, _ in _walk_repos(src):
        root = os.path.join(src, rel)
        head = subprocess.run(["git", "--no-optional-locks", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
        dirty = subprocess.run(["git", "--no-optional-locks", "-C", root, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False).stdout.strip()
        out[rel or "."] = {"head": head, "dirty": bool(dirty)}
    return out


def _without_git_stores(src: str, entries: Iterable[tuple[str, bool]], stores: list[str]) -> Iterator[tuple[str, bool]]:
    """Drop git object stores (keeping the empty dir, appended to stores) and worktree dirs from a walk of src."""
    pruned = None
    for rel, is_dir in entries:
        if pruned is not None and rel.startswith(pruned):
            continue
        pruned = None
        name, parent = os.path.basename(rel), os.path.dirname(rel)
        if is_dir and name in ("objects", "worktrees") and ".git" in rel.split(os.sep) and os.path.isfile(os.path.join(src, parent, "HEAD")):
            pruned = rel + os.sep
            if name == "worktrees":
                continue
            stores.append(rel)
        yield rel, is_dir


def _walk_repos(src: str) -> Iterator[tuple[str, bool]]:
    for dirpath, dirnames, filenames in os.walk(src):
        if ".git" in dirnames or ".git" in filenames:
            yield os.path.relpath(dirpath, src) if dirpath != src else "", True
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and d not in ("_assets", ".git")]


def _release() -> str:
    """Build stale packages, then freeze the dev tree into a release unless the latest one matches it, returning its id."""
    ws = _ws()
    t0 = time.monotonic()
    print("release: arena build (stale packages only)", flush=True)
    if subprocess.run([sys.executable, "-m", "arena_cli", "build"], cwd=ws, check=False).returncode:
        raise CLIError("arena build failed, a fork needs a dev tree that builds")
    t_build = time.monotonic() - t0

    root = _releases()
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        latest = _latest()
        prev = os.path.join(root, latest) if latest else None
        venvs = {pkg for pkg in os.listdir(os.path.join(ws, "build")) if os.path.isfile(os.path.join(ws, "build", pkg, "venv", "pyvenv.cfg"))}

        def planner_venv(rel: str) -> bool:
            head = rel.split(os.sep, 2)
            return len(head) > 2 and head[1] == "venv" and head[0] in venvs

        src = os.path.join(ws, "src")
        stores: list[str] = []
        venv = os.environ.get("ARENA_VENV_DIR", "/opt/venv")
        sources = {
            "src": ("src", src, lambda: _without_git_stores(src, _walk(src, skip_top=frozenset({"Arena/_assets"})), stores), lambda rel: False),
            "build": ("build", os.path.join(ws, "build"), lambda: _walk(os.path.join(ws, "build"), skip_top=frozenset({".forks", ".uv-cache"})), planner_venv),
            "install": ("install", os.path.join(ws, "install"), lambda: _walk(os.path.join(ws, "install")), lambda rel: False),
            "venv": (os.path.join("src", "Arena", ".venv"), venv, lambda: _walk(venv), lambda rel: False),
            "uv-python": ("uv-python", os.path.join(ws, ".uv-python"), lambda: _walk(os.path.join(ws, ".uv-python")), lambda rel: False),
        }
        t1 = time.monotonic()
        repos = _repos(src)
        if prev is not None and _meta(latest).get("format") == _FORMAT and repos == _meta(latest).get("repos"):
            prev_parts = _meta(latest).get("parts", {})
            for name, (dest, path, entries, link_source) in sources.items():
                p = _freeze(path, None, os.path.join(prev, dest), entries(), link_source)
                if p.changed or p.files != prev_parts.get(name, {}).get("files"):
                    break
            else:
                print(f"release: {latest} matches the dev tree (build {t_build:.0f}s, compare {time.monotonic() - t1:.0f}s)", flush=True)
                return latest
        for name in os.listdir(root):
            if name.startswith(".tmp-"):
                shutil.rmtree(os.path.join(root, name))
        release = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        while os.path.exists(os.path.join(root, release)):
            time.sleep(0.2)
            release = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tmp = os.path.join(root, f".tmp-{release}")
        stores.clear()
        parts: dict[str, _Part] = {}
        for name, (dest, path, entries, link_source) in sources.items():
            ts = time.monotonic()
            parts[name] = _freeze(path, os.path.join(tmp, dest), os.path.join(prev, dest) if prev else None, entries(), link_source)
            p = parts[name]
            print(f"release: {name:<9} {p.files:>7} files, {p.changed:>6} changed, {p.bytes / 1e6:>9.1f} MB copied, {time.monotonic() - ts:5.1f}s", flush=True)
        for store in stores:
            os.makedirs(os.path.join(tmp, "src", store, "info"), exist_ok=True)
            pathlib.Path(tmp, "src", store, "info", "alternates").write_text(f"{_DEV_SRC}/{store}\n")
        os.makedirs(os.path.join(tmp, "src", "Arena", "_assets"), exist_ok=True)
        pathlib.Path(tmp, ".built").touch()
        meta = {
            "id": release,
            "format": _FORMAT,
            "parent": latest,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "parts": {n: dataclasses.asdict(p) for n, p in parts.items()},
            "repos": repos,
        }
        with open(os.path.join(tmp, "release.json"), "w") as f:
            json.dump(meta, f, indent=1)
        os.rename(tmp, os.path.join(root, release))
        link = os.path.join(root, ".latest")
        os.symlink(release, link)
        os.replace(link, os.path.join(root, "latest"))
        _prune()
    copied = sum(p.bytes for p in parts.values())
    print(f"release: {release}, {copied / 1e9:.2f} GB new, parent {latest or 'none'} (build {t_build:.0f}s, freeze {time.monotonic() - t1:.0f}s)", flush=True)
    return release


def _prune() -> None:
    """Delete releases neither latest nor used by a fork."""
    keep = {_latest()} | {pathlib.Path(_root(), name, "release").read_text().strip() for name in os.listdir(_root()) if os.path.isfile(os.path.join(_root(), name, "release"))}
    for rid in _release_ids():
        if rid not in keep:
            shutil.rmtree(os.path.join(_releases(), rid))
            print(f"release: {rid} pruned, no fork uses it", flush=True)


def _require_dev() -> None:
    if os.environ.get("ARENA_FORK"):
        raise CLIError(f"this is fork {os.environ['ARENA_FORK']}, manage forks from the dev container")


def _fork_env(fork: str, release: str | None = None, domain: int | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["ARENA_FORK"] = fork
    env["ARENA_PROJECT_NAME"] = f"{_env('ARENA_IMAGE_TAG')}-{fork}"
    if release is not None:
        env["ARENA_RELEASE"] = release
    if domain is not None:
        env["ROS_DOMAIN_ID"] = str(domain)
    return env


def _domain(fork: str) -> int:
    return int(pathlib.Path(_root(), fork, "domain").read_text())


def _free_domain() -> int:
    """Lowest ROS domain from ARENA_FORK_DOMAIN_BASE up that no fork holds and no running container uses."""
    root = _root()
    held = {_domain(name) for name in os.listdir(root) if os.path.isfile(os.path.join(root, name, "domain"))} if os.path.isdir(root) else set()
    held |= {int(d) for d in _used_domains() if d.isdigit()}
    domain = int(os.environ.get("ARENA_FORK_DOMAIN_BASE", "20"))
    while domain in held:
        domain += 1
    if domain > 232:
        raise CLIError("no free ROS domain up to 232, delete a fork or lower ARENA_FORK_DOMAIN_BASE in .env")
    return domain


def _containers() -> dict[str, dict[str, str]]:
    """Fork name -> arena container id and state, for this workspace."""
    tag = _env("ARENA_IMAGE_TAG")
    out = features.engine_output(["ps", "-a", "--filter", "label=com.docker.compose.service=arena", "--format", '{{.ID}}\t{{.Label "com.docker.compose.project"}}\t{{.State}}'])
    found: dict[str, dict[str, str]] = {}
    for line in (out or "").splitlines():
        cid, project, state = line.split("\t")
        name = project[len(tag) + 1 :] if project.startswith(f"{tag}-") else ""
        if _FORK_RE.fullmatch(name):
            found[name] = {"id": cid, "state": state}
    return found


def _forks() -> dict[str, dict[str, str]]:
    """Fork name -> release, kind (kept or pool), ROS domain and container state, one entry per fork dir."""
    root = _root()
    containers = _containers()
    forks: dict[str, dict[str, str]] = {}
    for name in os.listdir(root) if os.path.isdir(root) else []:
        path = os.path.join(root, name)
        if not _FORK_RE.fullmatch(name) or not os.path.isfile(os.path.join(path, "release")) or not os.path.isfile(os.path.join(path, "domain")):
            continue
        forks[name] = {
            "release": pathlib.Path(path, "release").read_text().strip(),
            "kind": "kept" if os.path.exists(os.path.join(path, "kept")) else "pool",
            "domain": str(_domain(name)),
            "state": containers.get(name, {}).get("state", "no container"),
        }
    return dict(sorted(forks.items()))


def _used_domains() -> dict[str, str]:
    """ROS_DOMAIN_ID -> container name for every running container this engine shows."""
    ids = (features.engine_output(["ps", "-q"]) or "").split()
    if not ids:
        return {}
    out = features.engine_output(["inspect", "--format", '{{.Name}}\t{{json .Config.Env}}', *ids]) or ""
    used: dict[str, str] = {}
    for line in out.splitlines():
        name, envs = line.split("\t", 1)
        for e in json.loads(envs):
            if e.startswith("ROS_DOMAIN_ID="):
                used[e.split("=", 1)[1]] = name.lstrip("/")
    return used


def _create(fork: str, release: str, kind: str) -> None:
    root = os.path.join(_root(), fork)
    for part in _OVERLAYS:
        os.makedirs(os.path.join(root, "upper", part), exist_ok=True)
        os.makedirs(os.path.join(root, "work", part), exist_ok=True)
    if kind == "kept":
        pathlib.Path(root, "kept").touch()
    pathlib.Path(root, "domain").write_text(f"{_free_domain()}\n")
    pathlib.Path(root, "release").write_text(f"{release}\n")


def _start(fork: str, release: str) -> None:
    if not os.path.isdir(os.path.join(_releases(), release)):
        raise CLIError(f"{fork} sits on release {release}, which is gone, 'arena fork down {fork}' deletes the fork")
    domain = _domain(fork)
    used = _used_domains()
    if str(domain) in used:
        raise CLIError(f"domain {domain} of {fork} is in use by {used[str(domain)]}, stop that container or 'arena fork down {fork}'")
    t0 = time.monotonic()
    env = _fork_env(fork, release, domain)
    if features.compose(["--progress", "quiet", "up", "-d", "arena"], env=env) or features.wait_healthy("arena", env=env):
        raise CLIError(f"{fork} failed to start, see 'arena fork exec {fork}' or the engine logs")
    print(f"{fork}: up on release {release}, domain {domain} ({time.monotonic() - t0:.1f}s)", flush=True)


def _delete(fork: str, forks: dict[str, dict[str, str]]) -> None:
    t0 = time.monotonic()
    if forks[fork]["state"] != "no container" and features.compose(["--progress", "quiet", "down", "-v", "--timeout", "0"], env=_fork_env(fork, forks[fork]["release"])):
        raise CLIError(f"could not take down {fork}")
    if features.engine(["run", "--rm", "--user", "0", "--entrypoint", "rm", "-v", f"{_env('HOST_ARENA_WS_DIR')}/build/.forks:/forks", _env("ARENA_IMAGE"), "-rf", f"/forks/{fork}"]):
        raise CLIError(f"could not delete build/.forks/{fork}")
    print(f"{fork}: {forks[fork]['kind']} deleted ({time.monotonic() - t0:.1f}s)", flush=True)
    _prune()


def _pool(n: int) -> list[str]:
    """Run N pool forks on a release of the dev tree as it is now, recreating pool forks on older releases."""
    release = _release()
    forks = _forks()
    pool: list[str] = []
    i = 0
    while len(pool) < n:
        i += 1
        fork = f"p{i}"
        if fork in forks and forks[fork]["kind"] == "kept":
            continue
        pool.append(fork)
        if fork in forks and forks[fork]["release"] != release:
            _delete(fork, forks)
            del forks[fork]
        if fork in forks and forks[fork]["state"] == "running":
            print(f"{fork}: up on release {release}, domain {forks[fork]['domain']}", flush=True)
            continue
        if fork not in forks:
            _create(fork, release, "pool")
        _start(fork, release)
    return pool


def new(argv: list[str]) -> None:
    """Fork the dev tree into a new fork, or start an existing one.

    `arena fork new [<fork>]`, the lowest free pN when no name is given. A
    fork is the dev image over an overlay of a hardlinked snapshot of the built
    dev tree, so its edits, builds and commits land in build/.forks/<fork>
    without touching the dev tree. It has its own hostname, gz partition and
    the lowest ROS domain free from ARENA_FORK_DOMAIN_BASE (default 20) up, and
    ARENA_FORK_CPUS and ARENA_FORK_MEM in .env cap it. It stays until
    `arena fork down <fork>` names it.
    """
    _require_dev()
    if len(argv) > 1:
        raise CLIError("usage: arena fork new [<fork>]")
    name = argv[0] if argv else None
    if name is not None and (not _FORK_RE.fullmatch(name) or name == "releases"):
        raise CLIError(f"fork name '{name}' must be 1-63 lowercase letters, digits, '-' or '_', starting with a letter or digit, and not 'releases'")
    forks = _forks()
    fork = name if name is not None else next(f"p{i}" for i in range(1, len(forks) + 2) if f"p{i}" not in forks)
    if fork in forks:
        if forks[fork]["state"] == "running":
            print(f"{fork}: up on release {forks[fork]['release']}, domain {forks[fork]['domain']}", flush=True)
            return
        _start(fork, forks[fork]["release"])
        return
    release = _release()
    _create(fork, release, "kept")
    _start(fork, release)


def ls(argv: list[str]) -> None:
    """List forks and the releases they run on.

    `arena fork ls`. Forks show whether they are kept (until named) or pool
    forks, their release, ROS domain and container state. Releases show
    what each one changed against its parent and which forks use it.
    """
    _require_dev()
    if argv:
        raise CLIError("unexpected arguments, usage: arena fork ls")
    latest = _latest()
    forks = _forks()
    if forks:
        w = max(6, *(len(name) + 2 for name in forks))
        print(f"{'FORK':<{w}}{'KIND':<6}{'RELEASE':<17}{'DOMAIN':<8}STATE")
        for name, fork in forks.items():
            print(f"{name:<{w}}{fork['kind']:<6}{fork['release']:<17}{fork['domain']:<8}{fork['state']}")
        print()
    else:
        print("no forks, 'arena fork new' or 'source arena --fork' makes one, 'arena evaluation benchmark --lanes N' a pool\n")
    print(f"{'RELEASE':<17}{'CREATED':<21}{'CHANGED':>9}{'COPIED':>11}  FORKS")
    for rid in _release_ids():
        meta = _meta(rid)
        parts = meta.get("parts", {})
        changed = sum(p["changed"] for p in parts.values())
        copied = sum(p["bytes"] for p in parts.values())
        using = " ".join(n for n, fork in forks.items() if fork["release"] == rid)
        tag = " (latest)" if rid == latest else ""
        print(f"{rid:<17}{meta.get('created', '?'):<21}{changed:>9}{copied / 1e9:>9.2f}GB  {using or '-'}{tag}")


def exec_(argv: list[str]) -> None:
    """Open a shell in a fork, or run a command there.

    `arena fork exec <fork> [command...]`, starting the fork if it is stopped, e.g. `arena fork exec p1 arena launch sim:=gazebo headless:=true`.
    The command runs in the fork's sourced arena environment, on the fork's own ROS domain.
    """
    _require_dev()
    if not argv:
        raise CLIError("usage: arena fork exec <fork> [command...]")
    fork, cmd = argv[0], argv[1:]
    forks = _forks()
    if fork not in forks:
        raise CLIError(f"no fork '{fork}', see 'arena fork ls'")
    if forks[fork]["state"] != "running":
        _start(fork, forks[fork]["release"])
    env = _fork_env(fork)
    if not cmd and sys.stdin.isatty():
        sys.exit(features.compose(["exec", "arena", "/entrypoint.sh", "bash", "--rcfile", _RCFILE, "-i"], env=env))
    sys.exit(features.compose(["exec", "-T", "arena", "/entrypoint.sh", "bash", "--norc", "-c", _SOURCED, "bash", shlex.join(cmd) if cmd else "exec bash -s"], env=env))


def _flag(args: list[str], name: str) -> str | None:
    """Value of `--name value` or `--name=value` in args, None when absent."""
    for i, a in enumerate(args):
        if a == name:
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise CLIError(f"{name} needs a value")
            return args[i + 1]
        if a.startswith(f"{name}="):
            return a.split("=", 1)[1]
    return None


def split_lanes(argv: list[str]) -> tuple[int | None, list[str]]:
    """Take `--lanes N` or `--lanes=N` out of argv, returning (N or None, the rest)."""
    for i, a in enumerate(argv):
        if a == "--lanes" or a.startswith("--lanes="):
            value = a.split("=", 1)[1] if "=" in a else (argv[i + 1] if i + 1 < len(argv) else "")
            if not value.isdigit() or int(value) < 1:
                raise CLIError(f"--lanes takes a lane count >= 1, got '{value}'")
            return int(value), argv[:i] + argv[i + (1 if "=" in a else 2) :]
    return None, argv


def run_shared(n: int, args: list[str]) -> None:
    """Run one `arena evaluation benchmark` across N lanes, one pool fork each, on the dev tree as it is now, as one run dir."""
    _require_dev()
    if _flag(args, "--shared") is not None:
        raise CLIError("--lanes sets --shared itself")
    run_id = _flag(args, "--resume") or _flag(args, "--run-id")
    if run_id is None:
        suite = _flag(args, "--suite") or "basic"
        stem = "inline" if suite.lstrip().startswith(("[", "{")) else pathlib.Path(suite.removesuffix(".yaml")).stem
        run_id = f"{datetime.datetime.now(tz=datetime.UTC).strftime('%Y%m%d-%H%M%S')}-{stem}-x{n}"
        args = [*args, "--run-id", run_id]
    data_root = _flag(args, "--data-root") or "$ARENA_DATA_DIR/benchmarks"
    pool = _pool(n)

    token = secrets.token_hex(4)
    cmd = shlex.join(["arena", "evaluation", "benchmark", *args, "--shared", token])
    procs: dict[str, subprocess.Popen[str]] = {}

    def pump(fork: str, proc: subprocess.Popen[str]) -> None:
        for line in proc.stdout or ():
            print(f"[{fork}] {line.rstrip()}", flush=True)

    print(f"lanes: run {run_id} on {' '.join(pool)}, token {token}", flush=True)
    for fork in pool:
        compose = ["bash", "-c", 'arena_docker_compose "$@"', "arena_docker_compose", "exec", "-T", "arena", "/entrypoint.sh", "bash", "--norc", "-c", _SOURCED, "bash", cmd]
        procs[fork] = subprocess.Popen(compose, env=_fork_env(fork), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    pumps = [threading.Thread(target=pump, args=(fork, proc), daemon=True) for fork, proc in procs.items()]
    for th in pumps:
        th.start()

    def signal_lanes(sig: str) -> None:
        for fork in procs:
            pid = f'"{data_root}/{run_id}/runner.{fork}.pid"'
            features.compose(["exec", "-T", "arena", "bash", "-c", f"[ -f {pid} ] && kill -{sig} $(cat {pid})"], env=_fork_env(fork))

    try:
        for proc in procs.values():
            proc.wait()
    except KeyboardInterrupt:
        print("lanes: stopping (Ctrl-C again forces)", flush=True)
        signal_lanes("INT")
        try:
            for proc in procs.values():
                proc.wait()
        except KeyboardInterrupt:
            signal_lanes("TERM")
            for proc in procs.values():
                proc.wait()
    for th in pumps:
        th.join()
    rcs = {fork: proc.returncode for fork, proc in procs.items()}
    print(f"lanes: run {run_id} done, " + ", ".join(f"{fork} rc={rc}" for fork, rc in rcs.items()), flush=True)
    print(f"lanes: results in {os.path.expandvars(data_root)}/{run_id}", flush=True)
    subprocess.run(["ros2", "run", "arena_evaluation", "evaluation_cli", "status", *(["--data-root", data_root] if data_root != "$ARENA_DATA_DIR/benchmarks" else []), run_id], check=False)
    sys.exit(max(rcs.values()))


def down(argv: list[str]) -> None:
    """Delete forks. Bare, the pool forks of --lanes runs.

    `arena fork down [<fork>...]` deletes the named forks, container and edits
    both. Bare it deletes the pool forks p1, p2, ... that `arena evaluation
    benchmark --lanes N` made, kept forks go only when named.
    """
    _require_dev()
    forks = _forks()
    names = list(argv)
    bad = [a for a in names if a.startswith("-")]
    if bad:
        raise CLIError(f"unexpected argument {bad[0]}, usage: arena fork down [<fork>...]")
    targets = names or [name for name, fork in forks.items() if fork["kind"] == "pool"]
    if not targets:
        kept = [name for name, fork in forks.items() if fork["kind"] == "kept"]
        print(f"no pool forks, kept forks go only when named: arena fork down {' '.join(kept)}" if kept else "no forks")
        return
    unknown = [fork for fork in targets if fork not in forks]
    if unknown:
        raise CLIError(f"no fork {', '.join(unknown)}, forks: {', '.join(forks) or 'none'}")
    for fork in targets:
        _delete(fork, forks)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("new", new),
        make_verb("ls", ls),
        make_verb("exec", exec_, passthrough=True),
        make_verb("down", down),
    ]
}
