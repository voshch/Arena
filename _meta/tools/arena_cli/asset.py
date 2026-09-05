"""Inspect and move resolvable arena data. KINDS below is the list of what is movable."""

from __future__ import annotations

import dataclasses
import importlib
import sys

from common import CLIError, make_verb
from complete import Flags, Manifest, Static, Sub, Union

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
    "world": Kind("worlds", f"{_TREE}.World", "WorldIdentifier", "WORLD_PROVIDERS", closure="world", repo_subdir="worlds"),
    "suite": Kind("benchmark suites", _BENCH, "SuiteIdentifier", "BENCHMARK_PROVIDERS", bundle=True, closure="suite"),
    "contest": Kind("benchmark contests", _BENCH, "ContestIdentifier", "BENCHMARK_PROVIDERS", bundle=True),
    "manifest": Kind("report manifests", _BENCH, "ManifestIdentifier", "BENCHMARK_PROVIDERS", bundle=True),
}

DESCRIPTION = """Inspect and move resolvable arena data.

`ls` alone lists the kinds. Reading is anonymous, pushing needs GCS_ACCESS_TOKEN in the
environment."""


def _parse(argv: list[str], positionals: tuple[str, ...], flags: tuple[str, ...] = (), valued: tuple[str, ...] = (), optional: int = 0) -> tuple[list[str], dict[str, str | bool]]:
    """Split argv into the declared positionals (the last `optional` may be absent) and options."""
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
    if not len(positionals) - optional <= len(pos) <= len(positionals):
        raise CLIError(f"expected {' '.join(positionals)}, got {' '.join(pos) or 'nothing'}")
    return pos, opts


def _kind(kind: str) -> Kind:
    try:
        return KINDS[kind]
    except KeyError:
        raise CLIError(f"unknown kind {kind!r}, expected one of {', '.join(sorted(KINDS))}") from None


def _identifier_type(kind: str):
    """The identifier class for one kind. Imports the ROS-side package lazily."""
    spec = _kind(kind)
    return getattr(importlib.import_module(spec.module), spec.identifier)


def _identifier(kind: str, name: str):
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
                [
                    {"resolver": repr(v.resolver), "verdict": v.verdict.value, "path": str(v.path) if v.path else None}
                    for v in verdicts
                ],
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
    local = {
        identifier.shortname
        for resolver in identifier_t._resolvers
        if not isinstance(resolver, NetResolver)
        for identifier in resolver.listall()
    }
    remote = {
        identifier.shortname
        for resolver in identifier_t._resolvers
        if isinstance(resolver, NetResolver)
        for identifier in resolver.listall(network=True)
    }
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


def _net_resolver(kind: str, bucket: str):
    """The kind's resolver for one bucket, which owns that bucket's cache and freshness stamps."""
    from arena_simulation_setup.tree import NetResolver

    for resolver in _identifier_type(kind)._resolvers:
        if isinstance(resolver, NetResolver) and resolver._provider == bucket:
            return resolver
    raise CLIError(f"{bucket} is not a configured bucket for {kind}, expected one of {', '.join(_buckets(kind))}")


def _net(bucket: str, *args: str) -> int:
    import subprocess

    return subprocess.call(["ros2", "run", "arena_models", "arena_models", "-s", "net", bucket, *args])


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


def _world_dangling(view, source) -> list[str]:
    """World references that would dangle once published: resolvable only from a local-only source."""
    from arena_simulation_setup.tree import DynamicPaths, DynamicPathResolver, NetResolver

    # world-local assets only resolve once WORLD points at the world being published
    DynamicPaths.WORLD.path = source

    dangling: list[str] = []
    for identifier in view.identifiers(strict=True):
        # only the winning resolver matters here, and resolve_source stops at it rather
        # than probing every remaining one over the network
        try:
            hit = identifier.resolve_source_sync()
        except FileNotFoundError:
            dangling.append(f"{identifier.shortname} (unresolvable)")
            continue
        if isinstance(hit.resolver, NetResolver):
            continue
        if isinstance(hit.resolver, DynamicPathResolver) and hit.path.is_relative_to(source):
            continue
        dangling.append(f"{identifier.shortname} (only at {hit.path})")
    return dangling


def _suite_dangling(suite, source) -> list[str]:
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


def _config_source(identifier):
    """Where a bundled kind lives. It resolves to its yaml, so a directory bundle is
    reported as the directory holding it and a flat config as the file itself."""
    resolved = identifier.resolve_source_sync().path
    if resolved.is_file() and resolved.parent.name == identifier.name:
        return resolved.parent
    return resolved


def _drop_listing(kind: str, bucket: str) -> None:
    """A publish makes the cached bucket listing stale, so the next ls re-fetches it."""
    from arena_simulation_setup.tree import NetResolver

    for resolver in _identifier_type(kind)._resolvers:
        if isinstance(resolver, NetResolver) and resolver._provider == bucket:
            resolver._listing_path.unlink(missing_ok=True)


def push(argv: list[str]) -> int:
    """Publish an asset to its bucket.

    `arena asset push KIND NAME [--bucket B] [--force] [--yes]`. Kinds carrying a closure
    are checked first, and publishing is refused if any reference would not resolve for
    someone else. A flat benchmark config is wrapped into the directory bundle the bucket
    requires.
    """
    import shutil
    import tempfile
    from pathlib import Path

    (kind, name), opts = _parse(argv, ("KIND", "NAME"), flags=("--force", "--yes"), valued=("--bucket",))
    spec = _kind(kind)
    force = "--force" in opts
    bucket = _bucket_of(opts, kind)

    identifier = _identifier(kind, name)

    if spec.repo_subdir is not None and not force:
        from arena_simulation_setup import ASS_DIR

        if (ASS_DIR / spec.repo_subdir / name).is_dir():
            print(f"{name} ships in the repo, so a published copy would never be read. Use --force to publish anyway.", file=sys.stderr)
            return 1

    source = _config_source(identifier) if spec.bundle else identifier.resolve_source_sync().path

    verdict = "publishing anyway" if force else f"refusing to publish {name}"
    check = _PREFLIGHTS.get(spec.closure)
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
                return 1
    if dangling:
        print(f"{verdict}: {len(dangling)} reference(s) would not resolve for anyone else", file=sys.stderr)
        for entry in dangling:
            print(f"  MISS  {entry}", file=sys.stderr)
        if not force:
            print("bundle them under the asset directory, or pass --force", file=sys.stderr)
            return 1

    target = f"{bucket}:{identifier.relpath()}"
    if "--yes" not in opts:
        if not sys.stdin.isatty():
            raise CLIError(f"publishing {target} needs confirmation, pass --yes")
        print(f"publish {kind} {name} to {target} ? [y/N] ", end="", flush=True)
        if input().strip().lower() not in ("y", "yes"):
            return 1

    with tempfile.TemporaryDirectory() as staging:
        staged = Path(staging) / name
        if source.is_dir():
            # .ttl is cache bookkeeping; publishing it would ship a stale freshness stamp
            shutil.copytree(source, staged, ignore=shutil.ignore_patterns(".ttl"))
        else:
            staged.mkdir(parents=True)
            shutil.copy(source, staged / f"{kind}.yaml")
        files = [f for f in staged.rglob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        result = _net(bucket, "author", str(staged), "-d", str(identifier.relpath()))
    if result == 0:
        _drop_listing(kind, bucket)
        print(f"published {kind} {name} to {target}: {len(files)} file(s), {size / 1e6:.1f} MB")
    return result


_KIND = Sub({kind: Manifest(f"asset.{kind}") for kind in KINDS})
_KINDS = Static({kind: spec.description for kind, spec in KINDS.items()})

COMMANDS = {
    v.name: v
    for v in (
        make_verb("find", find, complete=Sub(_KIND.subs, extra=Flags({"--json": "machine-readable verdicts"}))),
        make_verb("ls", ls, complete=Union(_KINDS, Flags({"--network": "include bucket-only entries", "--json": "machine-readable"}))),
        make_verb("pull", pull, complete=_KIND),
        make_verb("push", push, complete=Sub(_KIND.subs, extra=Flags({"--force": "publish despite a failed preflight", "--yes": "skip the confirmation"}))),
    )
}
