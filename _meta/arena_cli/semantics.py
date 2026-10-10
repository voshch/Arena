"""Read and set the semantic state of running envs: lights, doors, signals, sounds, zones."""

from __future__ import annotations

import collections
import dataclasses
import fnmatch
import os
import random
import time
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from arena_cli.common import CLIError
from arena_cli.complete import Flags

if TYPE_CHECKING:
    import rclpy.node

SNAPSHOT_SUFFIX = "/state/semantics"
SET_SUFFIX = "/semantics/set"
DISCOVERY_S = 6.0
READ_S = 5.0

HELP = """Read and set the semantic state of running envs: lights, doors, signals, sounds, zones.

Usage:
  arena env [ENV] semantics [--kind KIND]                       list entities, live values, writable fields marked *
  arena env [ENV] semantics ENTITY                              one entity in full
  arena env [ENV] semantics ENTITY FIELD=VALUE [FIELD=VALUE]    set fields, ENTITY may be a glob
  arena env [ENV] semantics --kind KIND FIELD=VALUE             set a field on every entity of a kind
  arena env [ENV] semantics --watch [--kind KIND]               print changes as they happen, Ctrl-C stops

ENV (optional while one env runs):
  0, env_0          env by id or name
  --ns NS           env by node namespace
  --all             every running env

Names and values:
  ENTITY is the authored name (hall), or name/level (hall/0) where a world has levels.
  A zone and its ceiling lights share the zone's name: a field goes to the matched
  entities that have it, --kind narrows the match.
  VALUE is true|false for a predicate, a number or token otherwise. lo..hi draws a
  uniform number in that range and random one in 0..1, --seed N makes the draw repeat.
  A single = sets a field, := stays for launch arguments.

Examples:
  arena env semantics --kind light
  arena env 0 semantics hall level=0.3
  arena env 0 semantics hall lit=false dead_fraction=0.5
  arena env --all semantics --kind light lit=false
  arena env semantics 'desk_*' level=0.2..0.6 --seed 7

A write lasts until the next episode reset restores the authored state, and a lit
set by hand also yields when a regime the light follows (light_on) changes."""

FLAGS = Flags(
    {"--kind": "only entities of this kind", "--watch": "print changes until Ctrl-C", "--seed": "seed for lo..hi and random values", "-h": "this help"},
    valued=("--kind", "--seed"),
)


@dataclasses.dataclass(frozen=True)
class Entity:
    """One semantic entity of one env as its snapshot reports it."""

    env: str
    realized: str
    kind: str
    values: dict[str, str]
    roles: dict[str, str]

    @property
    def name(self) -> str:
        return self.realized.removeprefix(f"{self.env}/")

    @property
    def base(self) -> str:
        return self.name.partition("/")[0]

    @property
    def key(self) -> tuple[str, str]:
        return (self.kind, self.realized)

    def matches(self, pattern: str) -> bool:
        return fnmatch.fnmatchcase(self.name, pattern) or fnmatch.fnmatchcase(self.base, pattern)


def index(entities: Iterable[Entity]) -> dict[tuple[str, str], Entity]:
    """Entities by kind and realized name, the pair that identifies one."""
    return {e.key: e for e in entities}


def labels(entities: Iterable[Entity]) -> dict[tuple[str, str], str]:
    """Display name per entity: the authored name, with its level only where several levels of one kind share it."""
    entities = list(entities)
    counts = collections.Counter((e.kind, e.base) for e in entities)
    return {e.key: e.base if counts[e.kind, e.base] == 1 else e.name for e in entities}


@dataclasses.dataclass(frozen=True)
class Kind:
    """What the runtime declares about a semantic kind."""

    writable: frozenset[str] | None
    ranges: Mapping[str, tuple[float, float]]

    def writable_of(self, entity: Entity) -> frozenset[str]:
        return frozenset(entity.values) if self.writable is None else self.writable & frozenset(entity.values)


@dataclasses.dataclass
class Query:
    """A parsed `arena env [ENV] semantics ...` command line."""

    env: str | None = None
    ns: str | None = None
    all_envs: bool = False
    kind: str | None = None
    watch: bool = False
    seed: int | None = None
    entity: str | None = None
    assignments: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    help: bool = False


def split_env(argv: list[str]) -> tuple[list[str], list[str]] | None:
    """The env selector tokens and the semantics arguments, or None when argv is not a semantics command."""
    if "semantics" not in argv:
        return None
    index = argv.index("semantics")
    selector = argv[:index]
    if any(":=" in token for token in selector):
        return None
    return selector, argv[index + 1 :]


def parse(selector: list[str], argv: list[str]) -> Query:
    """Parse the env selector and the semantics arguments, CLIError naming what is accepted."""
    query = Query()
    tokens = iter(selector)
    for token in tokens:
        if token == "--all":
            query.all_envs = True
        elif token == "--ns":
            query.ns = next(tokens, None)
            if not query.ns:
                raise CLIError("arena env --ns takes a namespace, e.g. --ns /arena/env_0")
        elif token.startswith("-"):
            raise CLIError(f"arena env: unknown option {token!r} before 'semantics', expected ENV, --ns NS or --all")
        elif query.env is None:
            query.env = f"env_{token}" if token.isdigit() else token
        else:
            raise CLIError(f"arena env: one ENV before 'semantics', got {selector!r}")
    if sum((query.env is not None, query.ns is not None, query.all_envs)) > 1:
        raise CLIError("arena env: pass one of ENV, --ns NS or --all")
    tokens = iter(argv)
    for token in tokens:
        if token in ("-h", "--help"):
            query.help = True
        elif token == "--watch":
            query.watch = True
        elif token in ("--kind", "--seed"):
            value = next(tokens, None)
            if value is None:
                raise CLIError(f"arena env semantics: {token} takes a value")
            if token == "--kind":
                query.kind = value
            elif value.lstrip("-").isdigit():
                query.seed = int(value)
            else:
                raise CLIError(f"arena env semantics: --seed takes an integer, got {value!r}")
        elif token.startswith("-"):
            raise CLIError(f"arena env semantics: unknown option {token!r}, options are --kind KIND, --watch, --seed N")
        elif ":=" in token:
            raise CLIError(f"arena env semantics: {token!r} uses :=, set a field with a single =, e.g. {token.replace(':=', '=')}")
        elif "=" in token:
            field, _, value = token.partition("=")
            if not field or not value:
                raise CLIError(f"arena env semantics: {token!r} needs FIELD=VALUE")
            query.assignments.append((field, value))
        elif query.entity is None and not query.assignments:
            query.entity = token
        else:
            raise CLIError(f"arena env semantics: unexpected {token!r}, the form is ENTITY FIELD=VALUE [FIELD=VALUE ...]")
    if query.watch and (query.entity is not None or query.assignments):
        raise CLIError("arena env semantics: --watch takes only --kind")
    if query.assignments and query.entity is None and query.kind is None:
        raise CLIError("arena env semantics: name an ENTITY or pass --kind KIND to set fields")
    return query


def entity(env: str, realized: str, kind: str, discrete: Iterable[tuple[str, str]], continuous: Iterable[tuple[str, float]], predicates: Iterable[tuple[str, bool]]) -> Entity:
    """An entity from the name and value pairs of one snapshot entry."""
    values: dict[str, str] = {}
    roles: dict[str, str] = {}
    for name, value in discrete:
        values[name], roles[name] = str(value), "token"
    for name, value in continuous:
        values[name], roles[name] = f"{value:g}", "number"
    for name, value in predicates:
        values[name], roles[name] = "true" if value else "false", "true|false"
    return Entity(env, realized, kind, values, roles)


def select(entities: Iterable[Entity], pattern: str | None, kind: str | None) -> list[Entity]:
    """The entities matching a name or glob and a kind, CLIError listing what exists when none does."""
    entities = list(entities)
    kinds = sorted({e.kind for e in entities})
    if kind is not None and kind not in kinds:
        raise CLIError(f"no entity of kind {kind!r}, kinds present: {', '.join(kinds) or '(none)'}")
    chosen = [e for e in entities if (kind is None or e.kind == kind) and (pattern is None or e.matches(pattern))]
    if not chosen:
        names = sorted({label for e, label in zip(entities, labels(entities).values(), strict=True) if kind is None or e.kind == kind})
        raise CLIError(f"no entity matches {pattern!r}{f' of kind {kind}' if kind else ''}, entities: {', '.join(names) or '(none)'}")
    return chosen


def fitting(chosen: Iterable[Entity], assignments: list[tuple[str, str]]) -> list[Entity]:
    """The matched entities that carry every assigned field, all of them when none does."""
    chosen = list(chosen)
    return [e for e in chosen if all(field in e.values for field, _ in assignments)] or chosen


def draw(value: str, rng: random.Random) -> str:
    """A literal value as given, or a uniform draw for 'random' and 'lo..hi'."""
    if value == "random":
        return f"{rng.random():.4g}"
    low, sep, high = value.partition("..")
    if not sep:
        return value
    try:
        lo, hi = float(low), float(high)
    except ValueError:
        raise CLIError(f"{value!r} is not a range, write lo..hi with two numbers, e.g. 0.3..1.0") from None
    return f"{rng.uniform(lo, hi):.4g}"


def plan(chosen: Iterable[Entity], assignments: list[tuple[str, str]], kinds: Mapping[str, Kind], rng: random.Random) -> list[tuple[Entity, str, str]]:
    """The writes to send, one value drawn per entity and field, CLIError when a field or value cannot apply."""
    writes: list[tuple[Entity, str, str]] = []
    chosen = list(chosen)
    names = labels(chosen)
    for target in chosen:
        name = names[target.key]
        kind = kinds.get(target.kind, Kind(None, {}))
        writable = kind.writable_of(target)
        for field, raw in assignments:
            if field not in writable:
                listed = ", ".join(f"{f} ({target.roles[f]})" for f in sorted(writable)) or "none"
                what = "is read-only" if field in target.values else "does not exist"
                raise CLIError(f"{name}.{field} {what} on this {target.kind}, writable fields: {listed}")
            value = draw(raw, rng)
            role = target.roles[field]
            if role == "true|false" and value.lower() not in ("true", "false"):
                raise CLIError(f"{name}.{field} takes true or false, got {value!r}")
            if role == "number":
                try:
                    number = float(value)
                except ValueError:
                    raise CLIError(f"{name}.{field} takes a number, got {value!r}") from None
                low, high = kind.ranges.get(field, (-float("inf"), float("inf")))
                if not low <= number <= high:
                    raise CLIError(f"{name}.{field} takes a number in {low:g}..{high:g}, got {value}")
            writes.append((target, field, value.lower() if role == "true|false" else value))
    return writes


def render(entities: Iterable[Entity], kinds: Mapping[str, Kind]) -> str:
    """One line per entity: kind, name and every field, writable ones marked *."""
    rows = sorted(entities, key=lambda e: (e.kind, e.name))
    if not rows:
        return "  (no semantic entities)"
    names = labels(rows)
    kind_w = max(len(e.kind) for e in rows)
    name_w = max(len(name) for name in names.values())
    lines = []
    for e in rows:
        writable = kinds.get(e.kind, Kind(None, {})).writable_of(e)
        fields = " ".join(f"{f}={v or chr(34) * 2}{'*' if f in writable else ''}" for f, v in e.values.items())
        lines.append(f"  {e.kind.ljust(kind_w)}  {names[e.key].ljust(name_w)}  {fields}")
    return "\n".join(lines)


def describe(target: Entity, kinds: Mapping[str, Kind], name: str) -> str:
    """Every field of one entity with its value type, range and whether it is writable."""
    kind = kinds.get(target.kind, Kind(None, {}))
    writable = kind.writable_of(target)
    lines = [f"{name}  ({target.kind}, {target.realized})"]
    for field, value in target.values.items():
        low, high = kind.ranges.get(field, (None, None))
        span = f" in {low:g}..{high:g}" if low is not None else ""
        lines.append(f"  {field} = {value}    {target.roles[field]}{span}, {'writable' if field in writable else 'read-only'}")
    return "\n".join(lines)


def changes(previous: Mapping[tuple[str, str], Entity], current: Mapping[tuple[str, str], Entity]) -> list[str]:
    """One line per field that changed between two snapshots of an env."""
    lines = []
    names = labels(current.values())
    for key, now in current.items():
        before = previous.get(key)
        for field, value in now.values.items():
            old = before.values.get(field) if before is not None else None
            if old != value:
                lines.append(f"{now.env} {now.kind} {names[key]} {field}: {old if old is not None else '-'} -> {value}")
    return lines


def _kinds() -> dict[str, Kind]:
    from arena_runtime.sim._semantics import _SEMANTIC_KINDS

    return {name: Kind(frozenset(cls.WRITABLE), dict(cls.RANGES)) for name, cls in _SEMANTIC_KINDS.items()}


def _discover() -> list[str]:
    import subprocess

    try:
        out = subprocess.check_output(["ros2", "topic", "list"], stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return sorted(line.strip()[: -len(SNAPSHOT_SUFFIX)] for line in out.splitlines() if line.strip().endswith(SNAPSHOT_SUFFIX))


def _env_name(node_ns: str) -> str:
    return os.path.basename(os.path.dirname(node_ns))


def _targets(query: Query) -> list[str]:
    deadline = time.monotonic() + DISCOVERY_S
    nodes = _discover()
    while not nodes and time.monotonic() < deadline:
        time.sleep(0.5)
        nodes = _discover()
    if not nodes:
        raise CLIError("no env is running (no */state/semantics topic), start one with arena launch or arena env")
    listing = ", ".join(f"{_env_name(n)} ({os.path.dirname(n)})" for n in nodes)
    if query.all_envs:
        return nodes
    if query.ns is not None:
        chosen = [n for n in nodes if query.ns.rstrip("/") in (n, os.path.dirname(n))]
    elif query.env is not None:
        chosen = [n for n in nodes if _env_name(n) == query.env]
    elif len(nodes) == 1:
        chosen = nodes
    else:
        raise CLIError(f"several envs are running, name one before 'semantics' (arena env 0 semantics ...) or pass --all: {listing}")
    if not chosen:
        raise CLIError(f"no env {query.env or query.ns!r}, running: {listing}")
    return chosen


def _ros_node() -> object:
    import contextlib

    @contextlib.contextmanager
    def _cm() -> object:
        import rclpy

        rclpy.init(args=[])
        node = rclpy.create_node("arena_semantics_cli")
        try:
            yield node
        finally:
            node.destroy_node()
            rclpy.try_shutdown()

    return _cm()


def _entities(msg: object, env: str) -> dict[tuple[str, str], Entity]:
    return index(
        entity(
            env,
            e.entity,
            e.kind,
            zip(e.discrete_names, e.discrete_values, strict=True),
            zip(e.continuous_names, e.continuous_values, strict=True),
            zip(e.predicate_names, e.predicate_values, strict=True),
        )
        for e in msg.entities
    )


def _subscribe(node: rclpy.node.Node, node_ns: str, callback: object) -> object:
    import rclpy.qos
    from task_generator_msgs.msg import SemanticSnapshot

    qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
    return node.create_subscription(SemanticSnapshot, node_ns + SNAPSHOT_SUFFIX, callback, qos)


def _read(node: rclpy.node.Node, node_ns: str) -> dict[tuple[str, str], Entity]:
    import rclpy

    latest: list = []
    sub = _subscribe(node, node_ns, latest.append)
    deadline = time.monotonic() + READ_S
    while rclpy.ok() and not latest and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    if not latest:
        raise CLIError(f"{_env_name(node_ns)} published no semantic snapshot within {READ_S:g} s, is its world loaded?")
    return _entities(latest[-1], _env_name(node_ns))


def _set(node: rclpy.node.Node, node_ns: str, target: Entity, field: str, value: str) -> None:
    import rclpy
    from task_generator_msgs.srv import SetSemantic

    client = node.create_client(SetSemantic, node_ns + SET_SUFFIX)
    if not client.wait_for_service(timeout_sec=READ_S):
        raise CLIError(f"service {node_ns + SET_SUFFIX} is not available")
    future = client.call_async(SetSemantic.Request(entity=target.realized, field=field, value=value))
    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_client(client)
    result = future.result()
    if result is None or not result.success:
        raise CLIError(f"{target.env} rejected {target.realized}.{field}={value}: {result.error_msg if result else 'no response'}")


def _watch(node: rclpy.node.Node, nodes: list[str], kind: str | None) -> int:
    import rclpy
    from rclpy.executors import ExternalShutdownException

    last: dict[str, dict[tuple[str, str], Entity]] = {}

    def on_snapshot(node_ns: str, msg: object) -> None:
        current = {k: e for k, e in _entities(msg, _env_name(node_ns)).items() if kind is None or e.kind == kind}
        if node_ns in last:
            for line in changes(last[node_ns], current):
                print(line, flush=True)
        else:
            print(f"{_env_name(node_ns)}: watching {len(current)} entities", flush=True)
        last[node_ns] = current

    subs = [_subscribe(node, n, lambda msg, n=n: on_snapshot(n, msg)) for n in nodes]
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except (ExternalShutdownException, KeyboardInterrupt):
        return 0
    finally:
        for sub in subs:
            node.destroy_subscription(sub)
    return 0


def run(selector: list[str], argv: list[str]) -> int:
    """Entry point of `arena env [ENV] semantics ...`."""
    query = parse(selector, argv)
    if query.help:
        print(HELP)
        return 0
    nodes = _targets(query)
    kinds = _kinds()
    rng = random.Random(query.seed)
    with _ros_node() as node:
        if query.watch:
            return _watch(node, nodes, query.kind)
        for node_ns in nodes:
            env = _env_name(node_ns)
            entities = _read(node, node_ns)
            names = labels(entities.values())
            if not query.assignments and query.entity is None:
                shown = select(entities.values(), None, query.kind) if query.kind else list(entities.values())
                print(f"{env}  ({len(shown)} entities, * writable)")
                print(render(shown, kinds))
                continue
            chosen = fitting(select(entities.values(), query.entity, query.kind), query.assignments)
            if not query.assignments:
                print("\n".join(describe(e, kinds, names[e.key]) for e in chosen))
                continue
            for target, field, value in plan(chosen, query.assignments, kinds, rng):
                _set(node, node_ns, target, field, value)
                print(f"{env} {names[target.key]}.{field} = {value}")
    if not query.assignments and query.entity is None:
        print("\nset a field: arena env [ENV] semantics ENTITY FIELD=VALUE, all forms: arena env semantics --help")
    return 0
