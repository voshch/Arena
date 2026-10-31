"""Inspect and move resolvable arena data. KINDS below is the list of what is movable."""

from __future__ import annotations

import dataclasses
import importlib
import json
import os
import subprocess
import sys
import time
import typing

from arena_cli.common import CLIError, make_verb
from arena_cli.complete import Context, Dynamic, Flags, Manifest, Static, Sub, Union

if typing.TYPE_CHECKING:
    from pathlib import Path

    from arena_simulation_setup.tree import AssetIdentifier, NetResolver
    from arena_simulation_setup.tree.World import WorldView

    from arena_evaluation.benchmark.config import Suite

_TREE = "arena_simulation_setup.tree"
_BENCH = "arena_evaluation.benchmark.tree"


@dataclasses.dataclass(frozen=True)
class Kind:
    """One resolvable data kind: which identifier answers for it, and how it publishes."""

    description: str
    module: str
    identifier: str
    providers: str  # bucket-list symbol on arena_simulation_setup.tree
    parse: bool = False  # built via .parse() rather than the constructor
    bundle: bool = False  # resolves to a yaml, so publishing wraps it into a directory
    closure: str | None = None  # key into _PREFLIGHTS, checked before publishing
    repo_subdir: str | None = None  # ships in the repo under ASS_DIR/<subdir>


KINDS = {
    "object": Kind("3D object models", f"{_TREE}.assets.Object", "ObjectIdentifier", "NETWORK_PROVIDERS", parse=True),
    "human": Kind("human models", f"{_TREE}.assets.Human", "HumanIdentifier", "NETWORK_PROVIDERS", parse=True),
    "material": Kind("surface materials", f"{_TREE}.assets.Material", "MaterialIdentifier", "NETWORK_PROVIDERS", parse=True),
    "wall": Kind("wall kinds", f"{_TREE}.Wall", "WallIdentifier", "NETWORK_PROVIDERS", parse=True),
    "stairs": Kind("stair and escalator kinds", f"{_TREE}.Stairs", "StairsIdentifier", "NETWORK_PROVIDERS", parse=True),
    "sound": Kind("sound assets", f"{_TREE}.assets.Sound", "SoundIdentifier", "NETWORK_PROVIDERS", parse=True),
    "animation": Kind("pedestrian animation clips", f"{_TREE}.assets.Animation", "AnimationIdentifier", "NETWORK_PROVIDERS", parse=True),
    "world": Kind("worlds", f"{_TREE}.World", "WorldIdentifier", "WORLD_PROVIDERS", closure="world", repo_subdir="worlds"),
    "suite": Kind("benchmark suites", _BENCH, "SuiteIdentifier", "BENCHMARK_PROVIDERS", bundle=True, closure="suite"),
    "contest": Kind("benchmark contests", _BENCH, "ContestIdentifier", "BENCHMARK_PROVIDERS", bundle=True),
    "manifest": Kind("report manifests", _BENCH, "ManifestIdentifier", "BENCHMARK_PROVIDERS", bundle=True),
}

DESCRIPTION = """Inspect and move resolvable arena data.

`ls` alone lists the kinds. Reading is anonymous, pushing needs GCS_ACCESS_TOKEN in the
environment."""


def _parse(argv: list[str], positionals: tuple[str, ...], flags: tuple[str, ...] = (), valued: tuple[str, ...] = (), optional: int = 0, variadic: bool = False) -> tuple[list[str], dict[str, str | bool]]:
    """Split argv into the declared positionals (the last `optional` may be absent, `variadic` allows extras) and options."""
    pos: list[str] = []
    opts: dict[str, str | bool] = {}
    it = iter(argv)
    for arg in it:
        if arg in valued:
            value = next(it, None)
            if value is None:
                raise CLIError(f"{arg} needs a value")
            opts[arg] = value
        elif arg in flags:
            opts[arg] = True
        elif arg.startswith("-"):
            raise CLIError(f"No such option '{arg}'. Options: {', '.join((*flags, *valued))}")
        else:
            pos.append(arg)
    if len(pos) < len(positionals) - optional or (len(pos) > len(positionals) and not variadic):
        raise CLIError(f"expected {' '.join(positionals)}, got {' '.join(pos) or 'nothing'}")
    return pos, opts


def _kind(kind: str) -> Kind:
    try:
        return KINDS[kind]
    except KeyError:
        raise CLIError(f"unknown kind {kind!r}, expected one of {', '.join(sorted(KINDS))}") from None


def _identifier_type(kind: str) -> type[AssetIdentifier]:
    """The identifier class for one kind. Imports the ROS-side package lazily."""
    spec = _kind(kind)
    return getattr(importlib.import_module(spec.module), spec.identifier)


def _identifier(kind: str, name: str) -> AssetIdentifier:
    identifier_t = _identifier_type(kind)
    return identifier_t.parse(name) if _kind(kind).parse else identifier_t(name)


def find(argv: list[str]) -> int:
    """Show every resolver's verdict for one asset, without downloading it.

    `arena asset find KIND NAME [--json]`. Marks which source won and which were shadowed.
    """
    (kind, name), opts = _parse(argv, ("KIND", "NAME"), flags=("--json",))
    as_json = "--json" in opts

    from arena_simulation_setup.tree import Verdict

    labels = {Verdict.HIT: "HIT", Verdict.MISS: "miss", Verdict.SHADOWED: "shadowed", Verdict.STALE: "STALE"}
    verdicts = _identifier(kind, name).probe_sync()

    if as_json:
        import json

        print(
            json.dumps(
                [{"resolver": repr(v.resolver), "verdict": v.verdict.value, "path": str(v.path) if v.path else None} for v in verdicts],
                indent=2,
            )
        )
        return 0

    print(f"{kind} {name}")
    width = len(str(len(verdicts)))
    for n, verdict in enumerate(verdicts, start=1):
        label = labels[verdict.verdict]
        suffix = f"  {verdict.path}" if verdict.path is not None else ""
        print(f"  {n:>{width}d}  {repr(verdict.resolver):<64s} {label}{suffix}")
    if not any(v.verdict is Verdict.HIT for v in verdicts):
        return 1
    return 0


def ls(argv: list[str]) -> int:
    """List the kinds, or every available asset of one kind.

    `arena asset ls [KIND] [--network] [--json]`. With --network, bucket-only entries are
    listed too and names that a local copy shadows are marked.
    """
    pos, opts = _parse(argv, ("KIND",), flags=("--network", "--json"), optional=1)
    as_json = "--json" in opts
    if not pos:
        if as_json:
            import json

            print(json.dumps({kind: spec.description for kind, spec in KINDS.items()}, indent=2))
        else:
            width = max(len(kind) for kind in KINDS)
            for kind, spec in KINDS.items():
                print(f"{kind.ljust(width)}  {spec.description}")
        return 0
    kind = pos[0]
    network = "--network" in opts

    from arena_simulation_setup.tree import NetResolver

    identifier_t = _identifier_type(kind)

    if not network:
        names = sorted({identifier.shortname for identifier in identifier_t.listall()})
        _emit(names, as_json)
        return 0

    # a NetResolver's own cache is not a competing source, so it must not count as local
    local = {identifier.shortname for resolver in identifier_t._resolvers if not isinstance(resolver, NetResolver) for identifier in resolver.listall()}
    remote = {identifier.shortname for resolver in identifier_t._resolvers if isinstance(resolver, NetResolver) for identifier in resolver.listall(network=True)}
    _emit(sorted(local | remote), as_json, shadowed=local & remote)
    return 0


def _emit(names: list[str], as_json: bool, shadowed: set[str] | None = None) -> None:
    if as_json:
        import json

        print(json.dumps(names if shadowed is None else [{"name": n, "shadowed": n in shadowed} for n in names], indent=2))
        return
    shadowed = shadowed or set()
    for name in names:
        print(f"{name}  shadowed" if name in shadowed else name)


def _buckets(kind: str) -> list[str]:
    from arena_simulation_setup import tree

    return list(getattr(tree, _kind(kind).providers))


def _bucket_of(opts: dict[str, str | bool], kind: str) -> str:
    if "--bucket" in opts:
        return str(opts["--bucket"])
    candidates = _buckets(kind)
    if not candidates:
        raise CLIError(f"no bucket configured for {kind}, pass --bucket")
    return candidates[0]


def _net_resolver(kind: str, bucket: str) -> NetResolver:
    """The kind's resolver for one bucket, which owns that bucket's cache and freshness stamps."""
    from arena_simulation_setup.tree import NetResolver

    for resolver in _identifier_type(kind)._resolvers:
        if isinstance(resolver, NetResolver) and resolver._provider == bucket:
            return resolver
    raise CLIError(f"{bucket} is not a configured bucket for {kind}, expected one of {', '.join(_buckets(kind))}")


def _assets(*args: str) -> int:
    return subprocess.call(["ros2", "run", "arena_assets", "arena_assets", *args])


def _net(bucket: str, *args: str) -> int:
    return _assets("-s", "net", bucket, *args)


def pull(argv: list[str]) -> int:
    """Download an asset from its bucket into the local cache.

    `arena asset pull KIND NAME [--bucket B]`. Prints the cache path.
    """
    (kind, name), opts = _parse(argv, ("KIND", "NAME"), valued=("--bucket",))
    bucket = _bucket_of(opts, kind)
    identifier = _identifier(kind, name)
    resolver = _net_resolver(kind, bucket)
    path = identifier._run_sync(resolver.resolve(identifier))
    if path is None:
        print(f"{kind} {name} is not in {bucket}", file=sys.stderr)
        return 1
    print(path)
    return 0


def _world_dangling(view: WorldView, source: Path) -> list[str]:
    """World references that would dangle once published: resolvable only from a local-only source."""
    from arena_simulation_setup import ASS_DIR
    from arena_simulation_setup.tree import DynamicPathListResolver, DynamicPathResolver, DynamicPaths, NetResolver
    from arena_simulation_setup.tree.World import WorldIdentifier

    # world-local assets only resolve once WORLD points at the world being published
    DynamicPaths.WORLD.path = source
    world = view.load(validate=False)
    DynamicPaths.LINKED.paths = list(world.sources)

    dangling: list[str] = []
    for level in world.all_levels:
        link = level.source.partition(":")[0]
        if link.startswith(("./", "../")) and not (source / link).resolve().is_relative_to(source.resolve()):
            dangling.append(f"level {level.id} (links {level.source}, outside the world)")
    for identifier in view.identifiers(strict=True):
        # only the winning resolver matters here, and resolve_source stops at it rather
        # than probing every remaining one over the network
        try:
            hit = identifier.resolve_source_sync()
        except FileNotFoundError:
            dangling.append(f"{identifier.shortname} (unresolvable)")
            continue
        assert hit.path is not None
        if isinstance(hit.resolver, NetResolver):
            continue
        if isinstance(hit.resolver, DynamicPathResolver) and hit.path.is_relative_to(source):
            continue
        if isinstance(hit.resolver, DynamicPathListResolver):
            continue
        if isinstance(identifier, WorldIdentifier) and hit.path.is_relative_to(ASS_DIR / "worlds"):
            continue
        dangling.append(f"{identifier.shortname} (only at {hit.path})")
    return dangling


def _suite_dangling(suite: Suite, source: Path) -> list[str]:
    """Worlds a suite stages that would not resolve for someone else. One bundled under the
    suite counts, since the runner exports it via ARENA_WORLD_PATH.

    Task modes are deliberately not checked: they are code rather than data shipped with the
    suite, so the publisher's registry says nothing about the consumer's, and PROMPT is not
    registered until a human simulator constructs it.
    """
    from arena_simulation_setup.tree.World import WorldIdentifier

    dangling: list[str] = []
    for world in dict.fromkeys(stage.map for stage in suite.stages):
        if (source / "worlds" / world).is_dir():
            continue
        try:
            WorldIdentifier(world).resolve_source_sync()
        except FileNotFoundError:
            dangling.append(f"world {world} (unresolvable)")
    return dangling


_PREFLIGHTS = {"world": _world_dangling, "suite": _suite_dangling}


def _config_source(identifier: AssetIdentifier) -> Path:
    """Where a bundled kind lives. It resolves to its yaml, so a directory bundle is
    reported as the directory holding it and a flat config as the file itself."""
    resolved = identifier.resolve_source_sync().path
    assert resolved is not None
    if resolved.is_file() and resolved.parent.name == identifier.name:
        return resolved.parent
    return resolved


def expire(argv: list[str]) -> int:
    """Mark cached assets expired, so each is downloaded again on its next use.

    `arena asset expire [KIND NAME] [--bucket B]`. Bare, every cached asset of every bucket
    (or of B) and the bucket listings, else one asset. An expired asset is still served
    while its bucket is unreachable. A running runtime keeps what it already resolved.
    """
    pos, opts = _parse(argv, ("KIND", "NAME"), valued=("--bucket",), optional=2)
    if len(pos) == 1:
        raise CLIError(f"expected KIND NAME or nothing, got only {pos[0]!r}. Kinds: {', '.join(KINDS)}")
    if pos:
        kind, name = pos
        bucket = _bucket_of(opts, kind)
        if not _net_resolver(kind, bucket).expire(_identifier(kind, name)):
            print(f"{kind} {name} is not cached from {bucket}, nothing to expire. `arena asset pull {kind} {name}` fetches it", file=sys.stderr)
            return 1
        print(f"expired {kind} {name} in {bucket}")
        return 0

    from arena_simulation_setup.tree import Identifier, NetResolver

    buckets = list(dict.fromkeys(bucket for kind in KINDS for bucket in _buckets(kind)))
    if "--bucket" in opts:
        if opts["--bucket"] not in buckets:
            raise CLIError(f"{opts['--bucket']} is not a configured bucket, expected one of {', '.join(buckets)}")
        buckets = [str(opts["--bucket"])]
    for bucket in buckets:
        print(f"{bucket}: expired {NetResolver(Identifier, provider=bucket).expire()} cached asset(s)")
    return 0


def _drop_listing(kind: str, bucket: str) -> None:
    """A publish makes the cached bucket listing stale, so the next ls re-fetches it."""
    from arena_simulation_setup.tree import NetResolver

    for resolver in _identifier_type(kind)._resolvers:
        if isinstance(resolver, NetResolver) and resolver._provider == bucket:
            resolver._listing_path.unlink(missing_ok=True)


def _publishable(kind: str, name: str, identifier: AssetIdentifier, force: bool) -> Path | None:
    """The source of one asset, None when its preflight refuses it."""
    spec = _kind(kind)
    if spec.repo_subdir is not None and not force:
        from arena_simulation_setup import ASS_DIR

        if (ASS_DIR / spec.repo_subdir / name).is_dir():
            print(f"{name} ships in the repo, so a published copy would never be read. Use --force to publish anyway.", file=sys.stderr)
            return None

    source = _config_source(identifier) if spec.bundle else identifier.resolve_source_sync().path
    assert source is not None

    verdict = "publishing anyway" if force else f"refusing to publish {name}"
    check = _PREFLIGHTS.get(spec.closure) if spec.closure is not None else None
    dangling: list[str] = []
    if check is not None:
        try:
            dangling = check(identifier.resolve_sync(), source)
        except Exception as exc:
            # a strict walk raises on a reference it cannot read, and that must never be
            # mistaken for an asset that simply has no references
            reason = str(exc).splitlines()[0] or type(exc).__name__
            print(f"{verdict}: references could not be enumerated ({reason})", file=sys.stderr)
            if not force:
                return None
    if dangling:
        print(f"{verdict}: {len(dangling)} reference(s) would not resolve for anyone else", file=sys.stderr)
        for entry in dangling:
            print(f"  MISS  {entry}", file=sys.stderr)
        if not force:
            print("bundle them under the asset directory, or pass --force", file=sys.stderr)
            return None
    return source


def _stage(kind: str, source: Path, staged: Path, only: tuple[str, ...]) -> None:
    """Copy one asset into staged, without cache bookkeeping and, given `only`, just the files matching a glob."""
    import fnmatch
    import shutil

    def ignored(directory: str, names: list[str]) -> set[str]:
        unmatched = {name for name in names if only and not os.path.isdir(os.path.join(directory, name)) and not any(fnmatch.fnmatch(name, glob) for glob in only)}
        return unmatched | {".ttl"}

    if source.is_dir():
        # .ttl is cache bookkeeping; publishing it would ship a stale freshness stamp
        shutil.copytree(source, staged, ignore=ignored)
    else:
        staged.mkdir(parents=True)
        shutil.copy(source, staged / f"{kind}.yaml")


def _local_identifiers(kind: str) -> list[AssetIdentifier]:
    from arena_simulation_setup.tree import NetResolver

    found = {identifier.shortname: identifier for resolver in _identifier_type(kind)._resolvers if not isinstance(resolver, NetResolver) for identifier in resolver.listall()}
    return [found[name] for name in sorted(found)]


def push(argv: list[str]) -> int:
    """Publish assets to their bucket, adding and replacing files, never deleting any.

    `arena asset push KIND NAME|--all [--only GLOB[,GLOB]] [--dry-run] [--bucket B] [--force] [--yes]`.
    Kinds carrying a closure are checked first, and publishing is refused if any reference
    would not resolve for someone else. A flat benchmark config is wrapped into the
    directory bundle the bucket requires. --all publishes every local asset of the kind in
    one upload. --only keeps the files matching a glob, so the bucket's other files stay as
    they are. --dry-run lists the files and uploads nothing.
    """
    import tempfile
    from pathlib import Path

    pos, opts = _parse(argv, ("KIND", "NAME"), flags=("--all", "--dry-run", "--force", "--yes"), valued=("--bucket", "--only"), optional=1)
    kind = pos[0]
    spec = _kind(kind)
    if ("--all" in opts) == (len(pos) == 2):
        raise CLIError(f"pass one {kind} name or --all, run 'arena asset ls {kind}' to list them")
    only = tuple(str(opts["--only"]).split(",")) if "--only" in opts else ()
    if only and spec.bundle:
        raise CLIError(f"--only filters directory assets, a {kind} publishes one file")
    force = "--force" in opts
    bucket = _bucket_of(opts, kind)

    identifiers = _local_identifiers(kind) if "--all" in opts else [_identifier(kind, pos[1])]
    if not identifiers:
        raise CLIError(f"no local {kind} to publish, {kind} assets resolved from a bucket have nothing to send")
    names = [identifier.shortname for identifier in identifiers] if "--all" in opts else [pos[1]]
    sources = [_publishable(kind, name, identifier, force) for name, identifier in zip(names, identifiers, strict=True)]
    if None in sources:
        return 1

    relpaths = [str(identifier.relpath()) for identifier in identifiers]
    prefix = os.path.commonpath(relpaths)
    target = f"{bucket}:{prefix}"
    with tempfile.TemporaryDirectory() as staging:
        root = Path(staging) / "publish"
        for source, relpath in zip(sources, relpaths, strict=True):
            assert source is not None
            _stage(kind, source, root / os.path.relpath(relpath, prefix), only)
        files = sorted(f for f in root.rglob("*") if f.is_file())
        size = sum(f.stat().st_size for f in files)
        if not files:
            print(f"nothing to publish: no file of {' '.join(names)} matches {opts.get('--only')}", file=sys.stderr)
            return 1
        if "--dry-run" in opts:
            for file in files:
                print(f"{prefix}/{file.relative_to(root)}")
            print(f"would publish {len(files)} file(s), {size / 1e6:.1f} MB to {bucket}")
            return 0
        if "--yes" not in opts:
            if not sys.stdin.isatty():
                raise CLIError(f"publishing {target} needs confirmation, pass --yes")
            print(f"publish {len(files)} file(s) of {kind} {' '.join(names[:3])}{' ...' if len(names) > 3 else ''} to {target} ? [y/N] ", end="", flush=True)
            if input().strip().lower() not in ("y", "yes"):
                return 1
        result = _net(bucket, "author", str(root), "-d", prefix)
    if result == 0:
        _drop_listing(kind, bucket)
        print(f"published {len(names)} {kind} to {target}: {len(files)} file(s), {size / 1e6:.1f} MB")
    return result


def _import_args(pos: list[str], opts: dict[str, str | bool]) -> list[str]:
    """argv for `arena_assets import` from the parsed verb line."""
    if len(pos) < 2 and set(opts) - {"--json"}:
        raise CLIError("options need a KIND and a SOURCE, run 'arena asset import' to list them")
    if len(pos) > 2 and "--all" in opts:
        raise CLIError("pass ITEM... or --all, not both")
    args = list(pos)
    if "--json" in opts:
        if len(pos) > 2 or "--all" in opts:
            raise CLIError("--json lists sources or items, it takes no ITEM or --all")
        args.append("--json")
    if "--all" in opts:
        args.append("--all")
    if "--out" in opts:
        args += ["--out", str(opts["--out"])]
    return args


def import_(argv: list[str]) -> int:
    """Import items of one kind from an upstream dataset as local items the resolvers read.

    `arena asset import [KIND] [SOURCE] [ITEM...] [--all] [--out DIR] [--json]`. Bare, it lists the
    registered sources by kind with their license verdict (yes: public bucket with attribution,
    nc: NC prefix only, sa: SA prefix only, no: local use only, unclear), host and granularity
    (item, chunk, bulk). `import KIND` lists the sources of that kind, `import KIND SOURCE` lists
    that source's items and its default target, cached for tab completion (a day for live hosts,
    for good for pinned sources). ITEM... or --all converts those items into the imported root
    ($ARENA_ASSETS_DIR_IMPORTED, worlds under its worlds/), which every resolver reads after the
    curated local root, or --out DIR, each with an ATTRIBUTION.yaml beside it, and prints every
    item's path. One source can serve several kinds: `import world barn 17` is one challenge
    world, `import suite barn barn` the whole benchmark. --json prints a source or item listing
    as JSON.
    """
    pos, opts = _parse(argv, ("KIND", "SOURCE"), flags=("--all", "--json"), valued=("--out",), optional=2, variadic=True)
    return _assets("import", *_import_args(pos, opts))


IMPORT_KINDS = ("animation", "episode", "material", "object", "scenario", "sound", "suite", "world")

_KIND = Sub({kind: Manifest(f"asset.{kind}") for kind in KINDS})
_IMPORT_FLAGS = Flags({"--all": "import every item of the source", "--out": "target root instead of $ARENA_ASSETS_DIR_IMPORTED", "--json": "machine-readable source or item listing"}, valued=("--out",))
ITEM_CANDIDATES = 500
REFRESH_LOCK_MAX_AGE_S = 300.0


def import_listing(kind: str, source: str) -> str | None:
    """Item list the bare `import KIND SOURCE` caches for that source, None without ARENA_ASSETS_DIR."""
    root = os.environ.get("ARENA_ASSETS_DIR")
    return os.path.join(root, ".import", source, f"items.{kind}.json") if root else None


def listed_items(path: str, now: float) -> tuple[list[str], bool]:
    """Cached item ids and whether the list needs fetching again (missing, unreadable or expired)."""
    try:
        with open(path) as f:
            data = json.load(f)
        items, expires = [str(item) for item in data["items"]], data["expires_at"]
    except (OSError, ValueError, KeyError, TypeError):
        return [], True
    return items, expires is not None and now >= float(expires)


def _refresh_listing(kind: str, source: str, path: str, now: float) -> None:
    lock = path + ".lock"
    try:
        if now - os.stat(lock).st_mtime < REFRESH_LOCK_MAX_AGE_S:
            return
    except OSError:
        pass
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(lock, "w"):
            pass
        subprocess.Popen(["ros2", "run", "arena_assets", "arena_assets", "-s", "import", kind, source], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return


def import_candidates(kind: str, words: list[str], cur: str, ctx: Context) -> dict[str, str]:
    """Sources of the kind before the SOURCE word, then that source's cached items, a stale or missing list refreshed in the background."""
    positional = [word for index, word in enumerate(words) if not word.startswith("--") and not (index and words[index - 1] == "--out")]
    sources = ctx.values(f"asset.import.{kind}") or {}
    if not positional:
        return sources
    if positional[0] not in sources:
        return {}
    path = import_listing(kind, positional[0])
    if path is None:
        return {}
    now = time.time()
    items, stale = listed_items(path, now)
    if stale:
        _refresh_listing(kind, positional[0], path, now)
    taken = set(positional[1:])
    matches = [item for item in items if item.startswith(cur) and item not in taken]
    return dict.fromkeys(matches[:ITEM_CANDIDATES], "")


_IMPORT = Sub({kind: Union(Dynamic(lambda words, cur, ctx, kind=kind: import_candidates(kind, words, cur, ctx)), _IMPORT_FLAGS) for kind in IMPORT_KINDS})
_KINDS = Static({kind: spec.description for kind, spec in KINDS.items()})

COMMANDS = {
    v.name: v
    for v in (
        make_verb("find", find, complete=Sub(_KIND.subs, extra=Flags({"--json": "machine-readable verdicts"}))),
        make_verb("ls", ls, complete=Union(_KINDS, Flags({"--network": "include bucket-only entries", "--json": "machine-readable"}))),
        make_verb("pull", pull, complete=_KIND),
        make_verb("expire", expire, complete=_KIND),
        make_verb("push", push, complete=Sub(_KIND.subs, extra=Flags({"--all": "every local asset of the kind, in one upload", "--only": "only files matching GLOB[,GLOB]", "--dry-run": "list the files, upload nothing", "--force": "publish despite a failed preflight", "--yes": "skip the confirmation"}))),
        make_verb("import", import_, complete=_IMPORT),
    )
}
