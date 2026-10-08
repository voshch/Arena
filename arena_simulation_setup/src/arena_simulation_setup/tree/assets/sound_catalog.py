"""Sound catalog: the kinds table, sound assets with variants and deterministic variant selection."""

from __future__ import annotations

import enum
import fnmatch
import hashlib
import math
import re
import threading
import typing
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import attrs
import yaml

from arena_simulation_setup import ASS_DIR, DOMAIN_DEFAULT
from arena_simulation_setup.tree import DynamicPaths, NetResolver, PathResolverBase
from arena_simulation_setup.tree.assets.Sound import SoundIdentifier, SoundView
from arena_simulation_setup.tree.World import WorldIdentifier


class AgentKind(enum.StrEnum):
    PEDESTRIAN = "pedestrian"
    ROBOT = "robot"
    ENVIRONMENT = "environment"
    EXTERNAL = "external"


MANIFEST_VERSION = 2
WAV_MODELS = frozenset({"wav", "wav_loop"})
SURFACES = ("", "floor")
SAMPLE_KEY_SEPARATOR = "#"

type Stem = typing.Literal["pedestrian", "ambient", "motor"]
STEMS: tuple[Stem, ...] = ("pedestrian", "ambient", "motor")

_KIND_KEYS = frozenset({"height_m", "agent", "stem", "detect", "marker", "color", "default_asset"})
_ASSET_KEYS = frozenset({"version", "kind", "desc", "tags", "level_db", "reference_distance_m", "normalize_dbfs", "loop", "model", "surface", "variants", "kinds"})
_VARIANT_KEYS = frozenset({"id", "file", "model", "match", "default", "tags", "params"})


def _unknown_keys(data: Mapping[str, object], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys {unknown}, expected some of {sorted(allowed)}")


def _color(value: object) -> tuple[float, float, float]:
    r, g, b = (float(v) for v in typing.cast(Iterable[float], value))
    return (r, g, b)


def _agent(value: object) -> AgentKind | None:
    return None if value is None else AgentKind(str(value))


@attrs.frozen(kw_only=True)
class Kind:
    """One row of the kinds table."""

    name: str
    height_m: float = attrs.field(default=0.0, converter=float)
    agent: AgentKind | None = attrs.field(default=None, converter=_agent)
    stem: Stem = attrs.field(validator=attrs.validators.in_(STEMS))
    detect: bool = attrs.field(default=False, converter=bool)
    marker: bool = attrs.field(default=False, converter=bool)
    color: tuple[float, float, float] = attrs.field(default=(0.8, 0.8, 0.8), converter=_color)
    default_asset: str = attrs.field(default="", converter=str)

    @classmethod
    def from_dict(cls, name: str, data: Mapping[str, typing.Any], *, where: str) -> Kind:
        """Raises ValueError on unknown keys or values."""
        if not isinstance(data, Mapping):
            raise ValueError(f"{where}: kind {name!r} must be a mapping")
        _unknown_keys(data, _KIND_KEYS, f"{where}: kind {name!r}")
        try:
            return cls(name=str(name), **data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where}: kind {name!r}: {exc}") from exc


def _match_value(value: str) -> tuple[str, frozenset[str]]:
    leaf = str(value).strip().lower().rsplit("/", 1)[-1]
    return leaf, frozenset(re.split(r"[^a-z0-9]+", leaf))


def pattern_matches(pattern: str, value: str) -> bool:
    """Whole-word match on the lowercased leaf after the last '/', fnmatch on the leaf for patterns with '*' or '?'."""
    leaf, words = _match_value(value)
    if not leaf:
        return False
    if "*" in pattern or "?" in pattern:
        return fnmatch.fnmatchcase(leaf, pattern)
    return pattern in words


@attrs.frozen(kw_only=True)
class Variant:
    """One playable alternative of an asset. The id is unique within its asset."""

    id: str
    model: str
    path: Path | None = None
    match: Mapping[str, tuple[str, ...]] = attrs.field(factory=dict)
    default: bool = False
    tags: tuple[str, ...] = attrs.field(default=(), converter=tuple)
    params: Mapping[str, object] = attrs.field(factory=dict)

    def matches(self, context: Mapping[str, str]) -> bool:
        """Every match key has a context value that one of its patterns matches."""
        return all(any(pattern_matches(pattern, context.get(key, "")) for pattern in patterns) for key, patterns in self.match.items())


@attrs.frozen(kw_only=True)
class SoundAsset:
    """A sound asset, [domain/]name, and its variants."""

    id: str
    kind: str
    desc: str = ""
    tags: tuple[str, ...] = attrs.field(default=(), converter=tuple)
    level_db: float
    reference_distance_m: float = 1.0
    loop: bool = False
    surface: str = ""
    normalize_dbfs: float
    variants: tuple[Variant, ...] = attrs.field(converter=tuple)

    def variant(self, variant_id: str) -> Variant:
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        raise KeyError(f"sound asset {self.id!r} has no variant {variant_id!r}, expected one of {[v.id for v in self.variants]}")

    def select(self, *, context: Mapping[str, str], seed: int, models: frozenset[str] | None = None) -> Variant:
        """Matching non-default variants, else the defaults, else all, picked by seed. Raises LookupError when models excludes every variant."""
        pool = [variant for variant in self.variants if models is None or variant.model in models]
        if not pool:
            raise LookupError(f"sound asset {self.id!r} has no variant with a model in {sorted(models or ())}")
        candidates = [v for v in pool if not v.default and v.matches(context)] or [v for v in pool if v.default] or pool
        return candidates[int(seed) % len(candidates)]


def selection_seed(*parts: object) -> int:
    """blake2b-64 of the '|'-joined parts, stable across processes."""
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def sample_key(asset_id: str, variant_id: str) -> str:
    """The key of one variant's decoded sample, '<asset id>#<variant id>'."""
    return f"{asset_id}{SAMPLE_KEY_SEPARATOR}{variant_id}"


def split_sample_key(key: str) -> tuple[str, str]:
    """Asset id and variant id of a sample key. Raises KeyError on a key that lacks either."""
    asset_id, separator, variant_id = str(key).rpartition(SAMPLE_KEY_SEPARATOR)
    if not (separator and asset_id and variant_id):
        raise KeyError(f"sample key {key!r} is not '<asset id>{SAMPLE_KEY_SEPARATOR}<variant id>'")
    return asset_id, variant_id


def _canonical(identifier: SoundIdentifier) -> str:
    return identifier.name if identifier.domain == DOMAIN_DEFAULT else f"{identifier.domain}/{identifier.name}"


def canonical_asset_id(asset_id: str) -> str:
    """[domain/]name with the default domain and the Sound type segment dropped."""
    return _canonical(SoundIdentifier.parse(str(asset_id).strip()))


def parse_kinds(data: object, *, where: str) -> dict[str, Kind]:
    """A kinds: mapping. Raises ValueError on a malformed table."""
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: kinds must be a mapping")
    return {str(name): Kind.from_dict(str(name), entry, where=where) for name, entry in data.items()}


def _strings(value: object, what: str, where: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError(f"{where}: {what} must be a string or a list of strings, got {value!r}")


def _number(data: Mapping[str, object], key: str, where: str, default: float | None = None) -> float:
    value = data.get(key, default)
    try:
        return float(typing.cast(float, value))
    except (TypeError, ValueError):
        raise ValueError(f"{where}: {key} must be a number, got {value!r}") from None


def _parse_match(raw: object, where: str) -> dict[str, tuple[str, ...]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: match must be a mapping of context key to patterns")
    match: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        patterns = tuple(pattern.strip().lower() for pattern in _strings(value, f"match {key!r}", where))
        if not patterns or not all(patterns):
            raise ValueError(f"{where}: match {key!r} needs non-empty patterns")
        match[str(key)] = patterns
    return match


def parse_manifest(asset_id: str, directory: Path, data: object) -> tuple[SoundAsset, dict[str, Kind]]:
    """A v2 manifest and its kinds fragment. Raises ValueError on a malformed manifest, FileNotFoundError on a missing wav."""
    where = f"sound asset {asset_id!r} ({directory})"
    if not isinstance(data, Mapping):
        raise ValueError(f"{where}: manifest must be a mapping")
    _unknown_keys(data, _ASSET_KEYS, where)
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{where}: manifest version must be {MANIFEST_VERSION}, got {data.get('version')!r}")
    for key in ("kind", "level_db", "normalize_dbfs", "variants"):
        if key not in data:
            raise ValueError(f"{where}: missing {key!r}")
    loop = bool(data.get("loop", False))
    asset_model = str(data.get("model", "wav_loop" if loop else "wav"))
    surface = str(data.get("surface", ""))
    if surface not in SURFACES:
        raise ValueError(f"{where}: surface must be one of {SURFACES}, got {surface!r}")
    desc = data.get("desc", "")
    if not isinstance(desc, str):
        raise ValueError(f"{where}: desc must be a string, got {desc!r}")
    reference_distance_m = _number(data, "reference_distance_m", where, 1.0)
    if not reference_distance_m > 0.0:
        raise ValueError(f"{where}: reference_distance_m must be positive")
    level_db = _number(data, "level_db", where)
    normalize_dbfs = _number(data, "normalize_dbfs", where)
    if not (math.isfinite(level_db) and math.isfinite(normalize_dbfs)):
        raise ValueError(f"{where}: level_db and normalize_dbfs must be finite")
    raw_variants = data["variants"]
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError(f"{where}: variants must be a non-empty list")
    variants: list[Variant] = []
    for raw in raw_variants:
        if not isinstance(raw, Mapping) or "id" not in raw:
            raise ValueError(f"{where}: every variant is a mapping with an id")
        variant_where = f"{where} variant {raw['id']!r}"
        _unknown_keys(raw, _VARIANT_KEYS, variant_where)
        if not str(raw["id"]) or SAMPLE_KEY_SEPARATOR in str(raw["id"]):
            raise ValueError(f"{variant_where}: id must be non-empty and free of {SAMPLE_KEY_SEPARATOR!r}")
        model = str(raw.get("model", asset_model))
        path = None
        if raw.get("file"):
            path = directory / str(raw["file"])
            if not path.is_file():
                raise FileNotFoundError(f"{variant_where}: missing wav {path}")
        elif model in WAV_MODELS:
            raise ValueError(f"{variant_where}: model {model!r} needs a file")
        params = raw.get("params", {})
        if not isinstance(params, Mapping):
            raise ValueError(f"{variant_where}: params must be a mapping")
        variants.append(
            Variant(
                id=str(raw["id"]),
                model=model,
                path=path,
                match=_parse_match(raw.get("match"), variant_where),
                default=bool(raw.get("default", False)),
                tags=_strings(raw.get("tags"), "tags", variant_where),
                params=dict(params),
            )
        )
    ids = [variant.id for variant in variants]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{where}: duplicate variant ids {ids}")
    asset = SoundAsset(
        id=asset_id,
        kind=str(data["kind"]),
        desc=desc,
        tags=_strings(data.get("tags"), "tags", where),
        level_db=level_db,
        reference_distance_m=reference_distance_m,
        loop=loop,
        surface=surface,
        normalize_dbfs=normalize_dbfs,
        variants=tuple(variants),
    )
    return asset, parse_kinds(data.get("kinds"), where=where)


def _load_manifest(asset_id: str, view: SoundView) -> tuple[SoundAsset, dict[str, Kind]]:
    manifest_path = view.path / f"{view.path.name}.yaml"
    try:
        manifest = view.manifest
    except FileNotFoundError as exc:
        raise ValueError(f"sound asset {asset_id!r} at {view.path} has no manifest {manifest_path.name}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"sound asset {asset_id!r}: unreadable manifest {manifest_path}: {exc}") from exc
    try:
        return parse_manifest(asset_id, view.path, manifest)
    except TypeError as exc:
        raise ValueError(f"sound asset {asset_id!r}: malformed manifest {manifest_path}: {exc}") from exc


def _load_asset(key: str) -> tuple[SoundAsset, dict[str, Kind]]:
    try:
        view = SoundIdentifier.parse(key).resolve_sync()
    except FileNotFoundError:
        raise KeyError(f"unknown sound asset {key!r}") from None
    return _load_manifest(key, view)


def local_sound_dirs() -> dict[str, Path]:
    """Directory of every sound asset in the world and shared local trees by canonical id, the first resolver holding an id winning."""
    found: dict[str, Path] = {}
    for resolver in SoundIdentifier._resolvers:
        if isinstance(resolver, NetResolver) or not isinstance(resolver, PathResolverBase) or not resolver.path.is_dir():
            continue
        for directory in sorted(resolver.path.glob(f"*/{SoundIdentifier.label()}/*")):
            if directory.is_dir():
                found.setdefault(canonical_asset_id(f"{directory.parent.parent.name}/{directory.name}"), directory)
    return found


def _manifest_kinds(asset_id: str, directory: Path) -> dict[str, Kind]:
    try:
        data = SoundView(directory).manifest
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, Mapping):
        return {}
    return parse_kinds(data.get("kinds"), where=f"sound asset {asset_id!r} ({directory})")


def kinds_file() -> Path:
    return ASS_DIR / "configs" / "sounds" / "kinds.yaml"


def _load_kinds_file(path: Path) -> dict[str, Kind]:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: kinds file must be a mapping")
    if data.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{path}: version must be {MANIFEST_VERSION}, got {data.get('version')!r}")
    _unknown_keys(data, frozenset({"version", "kinds"}), str(path))
    return parse_kinds(data.get("kinds"), where=str(path))


class SoundLibrary:
    """Kinds table plus lazily resolved sound assets. Thread-safe."""

    _default: typing.ClassVar[SoundLibrary | None] = None
    _default_lock: typing.ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, kinds_files: Sequence[Path]) -> None:
        self._lock = threading.RLock()
        self._file_kinds: dict[str, Kind] = {}
        for path in kinds_files:
            self._merge(self._file_kinds, _load_kinds_file(path), str(path))
        self._base_kinds = self._local_kinds()
        self._kinds = dict(self._base_kinds)
        self._assets: dict[str, SoundAsset] = {}
        self._missing: set[str] = set()
        self._world: Path | None = None

    @classmethod
    def default(cls) -> SoundLibrary:
        """The process-wide library over kinds_file()."""
        with cls._default_lock:
            if cls._default is None:
                cls._default = cls([kinds_file()])
            return cls._default

    @staticmethod
    def _merge(into: dict[str, Kind], fragment: Mapping[str, Kind], where: str) -> None:
        for name, kind in fragment.items():
            existing = into.get(name)
            if existing is not None and existing != kind:
                raise ValueError(f"{where}: redefines kind {name!r} with different values")
            into[name] = kind

    def _local_kinds(self) -> dict[str, Kind]:
        kinds = dict(self._file_kinds)
        for asset_id, directory in local_sound_dirs().items():
            self._merge(kinds, _manifest_kinds(asset_id, directory), f"sound asset {asset_id!r}")
        return kinds

    def use_world(self, world_path: Path | None) -> None:
        """Point world-local sound resolution at world_path, drop every cached asset and reread the local kinds when it changes. Raises ValueError on bad local kinds, keeping the previous world."""
        path = Path(world_path) if world_path is not None else None
        with self._lock:
            if path == self._world:
                return
            self._point_at(path)
            try:
                base_kinds = self._local_kinds()
            except ValueError:
                self._point_at(self._world)
                raise
            self._base_kinds = base_kinds
            self._kinds = dict(base_kinds)
            self._world = path

    def _point_at(self, path: Path | None) -> None:
        DynamicPaths.WORLD.path = path if path is not None else Path("/dev/null")
        for resolver in SoundIdentifier._resolvers:
            resolver.invalidate()
        self._assets.clear()
        self._missing.clear()

    def use_world_named(self, world: str) -> None:
        """use_world on the named world's resolved path, a level filter ignored. Raises the world resolver's errors."""
        self.use_world(Path(WorldIdentifier(WorldIdentifier.parse(world)[0]).resolve_sync().path))

    def kinds(self) -> Mapping[str, Kind]:
        with self._lock:
            return dict(self._kinds)

    def kind(self, name: str) -> Kind:
        with self._lock:
            try:
                return self._kinds[name]
            except KeyError:
                raise KeyError(f"unknown sound kind {name!r}, expected one of {sorted(self._kinds)}") from None

    def asset(self, asset_id: str) -> SoundAsset:
        """Resolve and parse a sound asset. Raises KeyError when no resolver has it, ValueError on a malformed manifest."""
        key = canonical_asset_id(asset_id)
        with self._lock:
            cached = self._assets.get(key)
            if cached is not None:
                return cached
            if key in self._missing:
                raise KeyError(f"unknown sound asset {asset_id!r}")
            try:
                asset, fragment = _load_asset(key)
            except KeyError:
                self._missing.add(key)
                raise
            return self._admit(key, asset, fragment)

    def _admit(self, key: str, asset: SoundAsset, fragment: Mapping[str, Kind]) -> SoundAsset:
        with self._lock:
            cached = self._assets.get(key)
            if cached is not None:
                return cached
            kinds = dict(self._kinds)
            self._merge(kinds, fragment, f"sound asset {key!r}")
            if asset.kind not in kinds:
                raise ValueError(f"sound asset {key!r}: unknown kind {asset.kind!r}, expected one of {sorted(kinds)}")
            self._kinds = kinds
            self._assets[key] = asset
            return asset

    def known_asset_ids(self) -> tuple[str, ...]:
        """Ids of the sound assets on disk (world, local tree, bucket caches) and of every kind's default asset."""
        defaults = {canonical_asset_id(kind.default_asset) for kind in self.kinds().values() if kind.default_asset}
        return tuple(sorted(defaults.union(_canonical(identifier) for identifier in SoundIdentifier.listall())))

    def default_asset(self, kind: str) -> SoundAsset:
        """Raises KeyError for an unknown kind or one without a default asset."""
        default = self.kind(kind).default_asset
        if not default:
            raise KeyError(f"sound kind {kind!r} has no default asset")
        return self.asset(default)

    def variant(self, asset_id: str, variant_id: str) -> Variant:
        return self.asset(asset_id).variant(variant_id)

    def kinds_of(self, agent: AgentKind) -> tuple[str, ...]:
        """Names of the kinds emitted by agent, in table order."""
        return tuple(name for name, kind in self.kinds().items() if kind.agent is agent)
