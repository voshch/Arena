"""Pose profiles: harmonic joint signals per gait, cadence and gain laws, and the yaml library and agent-type sections resolving them."""

from __future__ import annotations

import cmath
import functools
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import attrs
import yaml

JOINT_NAMES: tuple[str, ...] = (
    "r_waist",
    "y_waist",
    "waist",
    "r_spine",
    "y_spine",
    "spine",
    "r_chest",
    "y_chest",
    "chest",
    "r_head",
    "y_head",
    "p_head",
    "l_y_collar",
    "l_p_collar",
    "l_y_shoulder",
    "l_p_shoulder",
    "l_r_shoulder",
    "l_elbow",
    "r_y_collar",
    "r_p_collar",
    "r_y_shoulder",
    "r_p_shoulder",
    "r_r_shoulder",
    "r_elbow",
    "l_y_hip",
    "l_p_hip",
    "l_r_hip",
    "l_knee",
    "r_y_hip",
    "r_p_hip",
    "r_r_hip",
    "r_knee",
    "l_y_ankle",
    "l_ankle",
    "r_y_ankle",
    "r_ankle",
)

LIBRARY_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_WALK = "walk_cmu_12_01"
DEFAULT_RUN = "run"
IDLE = "idle"
GAITS = ("walk", "run", "idle")
OPS = ("set", "scale", "offset")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@attrs.frozen
class Signal:
    """Mean plus integer harmonics (amp, phase) in radians, value = mean + sum amp_k * sin(k * phi + phase_k)."""

    mean: float
    harmonics: tuple[tuple[float, float], ...] = ()

    def evaluate(self, phi: float, shift: float, gain: float) -> float:
        v = self.mean
        for k, (amp, ph) in enumerate(self.harmonics, start=1):
            v += amp * math.sin(k * (phi + shift) + ph)
        return gain * v

    def scaled(self, gain: float) -> Signal:
        return Signal(gain * self.mean, tuple((gain * amp, ph) for amp, ph in self.harmonics))

    def plus(self, other: Signal) -> Signal:
        """Sum of two signals, harmonics of the same order added as phasors."""
        n = max(len(self.harmonics), len(other.harmonics))
        mine = self.harmonics + ((0.0, 0.0),) * (n - len(self.harmonics))
        theirs = other.harmonics + ((0.0, 0.0),) * (n - len(other.harmonics))
        harmonics: list[tuple[float, float]] = []
        for (a1, p1), (a2, p2) in zip(mine, theirs, strict=True):
            if a2 == 0.0:
                harmonics.append((a1, p1))
            elif a1 == 0.0:
                harmonics.append((a2, p2))
            else:
                z = a1 * cmath.exp(1j * p1) + a2 * cmath.exp(1j * p2)
                harmonics.append((abs(z), cmath.phase(z)))
        return Signal(self.mean + other.mean, tuple(harmonics))

    @classmethod
    def parse(cls, raw: float | Mapping[str, Any]) -> Signal:
        """A bare number is a constant, a mapping takes mean and harmonics: [[amp, phase], ...]."""
        if isinstance(raw, int | float):
            return cls(float(raw))
        if not isinstance(raw, Mapping) or not set(raw) <= {"mean", "harmonics"}:
            raise ValueError(f"signal must be a number or {{mean, harmonics}}, got {raw!r}")
        harmonics = tuple((float(amp), float(ph)) for amp, ph in raw.get("harmonics", ()))
        return cls(float(raw.get("mean", 0.0)), harmonics)


@attrs.frozen
class CadenceLaw:
    """Cycle rate in Hz from speed: clamp(base + per_speed * |v|, lo, hi)."""

    base: float = 0.4
    per_speed: float = 0.55
    lo: float = 0.4
    hi: float = 2.2

    def __call__(self, speed_abs: float) -> float:
        return _clamp(self.base + self.per_speed * speed_abs, self.lo, self.hi)


@attrs.frozen
class GainLaw:
    """Amplitude gain from speed: factor * clamp(|v| / per_speed, lo, hi)."""

    per_speed: float = 1.2
    lo: float = 0.2
    hi: float = 1.0
    factor: float = 1.0

    def __call__(self, speed_abs: float) -> float:
        return self.factor * _clamp(speed_abs / self.per_speed, self.lo, self.hi)


@attrs.frozen
class PhaseWarp:
    """Piecewise-linear remap of phi mod 2pi so the first half-cycle takes `split` of the period, identity at 0.5."""

    split: float = 0.5

    def __attrs_post_init__(self) -> None:
        if not 0.0 < self.split < 1.0:
            raise ValueError(f"phase_warp.split must lie in (0, 1), got {self.split}")

    def __call__(self, phi: float) -> float:
        if self.split == 0.5:
            return phi
        cycle = 2.0 * math.pi
        u = phi % cycle
        base = phi - u
        knee = cycle * self.split
        if u < knee:
            return base + u / (2.0 * self.split)
        return base + math.pi + (u - knee) / (2.0 * (1.0 - self.split))


@attrs.frozen
class JointSignal:
    """One joint's binding: the canonical signal key, the phase shift it is read at, and the signal itself."""

    key: str
    shift: float
    signal: Signal


_LAW_FIELDS = {
    "phase_warp": {"split": "split"},
    "cadence": {"base": "base", "per_speed": "per_speed", "min": "lo", "max": "hi"},
    "gain": {"per_speed": "per_speed", "min": "lo", "max": "hi", "factor": "factor"},
}


def _parse_limits(name: str, raw: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for joint, pair in raw.items():
        if joint not in JOINT_NAMES or not isinstance(pair, list | tuple) or len(pair) != 2 or float(pair[0]) > float(pair[1]):
            raise ValueError(f"{name}: limits take {{<joint from JOINT_NAMES>: [lo, hi]}}, got {joint!r}: {pair!r}")
        limits[joint] = (float(pair[0]), float(pair[1]))
    return limits


@attrs.frozen
class GaitProfile:
    """One cyclic gait: joint signals read at a shared phase, with its cadence, gain and warp laws."""

    name: str
    joints: Mapping[str, JointSignal]
    symmetric: bool = True
    phase_warp: PhaseWarp = PhaseWarp()
    cadence: CadenceLaw = CadenceLaw()
    gain: GainLaw = GainLaw()
    limits: Mapping[str, tuple[float, float]] = attrs.field(factory=dict)

    def evaluate(self, phi: float, gain: float) -> dict[str, float]:
        phi = self.phase_warp(phi)
        return {joint: js.signal.evaluate(phi, js.shift, gain) for joint, js in self.joints.items()}

    def targets(self, name: str) -> dict[str, JointSignal]:
        """Joints an op named `name` applies to: a signal key, a bare joint, or (asymmetric only) one side of a pair."""
        matches = {joint: js for joint, js in self.joints.items() if js.key == name}
        if matches:
            return matches
        if f"l_{name}" in JOINT_NAMES and f"r_{name}" in JOINT_NAMES:
            return {joint: self.joints.get(joint, JointSignal(name, shift, Signal(0.0))) for joint, shift in ((f"l_{name}", 0.0), (f"r_{name}", math.pi))}
        if name not in JOINT_NAMES:
            raise ValueError(f"{self.name}: unknown joint {name!r}, expected a signal key {sorted({js.key for js in self.joints.values()})} or a joint from JOINT_NAMES")
        sided = name[:2] in ("l_", "r_") and f"l_{name[2:]}" in JOINT_NAMES and f"r_{name[2:]}" in JOINT_NAMES
        if sided and self.symmetric:
            key = self.joints[name].key if name in self.joints else name[2:]
            raise ValueError(f"{self.name}: {name!r} is one side of a symmetric gait, name its key {key!r} or set symmetric: false")
        return {name: self.joints.get(name, JointSignal(name, 0.0, Signal(0.0)))}

    def with_ops(self, ops: Mapping[str, Mapping[str, Any]]) -> GaitProfile:
        """Apply the `joints:` ops of an agent-type section: set replaces, scale multiplies, offset adds a signal."""
        joints = dict(self.joints)
        for name, spec in ops.items():
            if not isinstance(spec, Mapping) or not spec or not set(spec) <= set(OPS):
                raise ValueError(f"{self.name}: joint {name!r} takes {OPS}, got {spec!r}")
            targets = attrs.evolve(self, joints=joints).targets(name)
            for op, arg in spec.items():
                fn = _OP[op](Signal.parse(arg) if op != "scale" else float(arg))
                if self.symmetric:
                    shared = fn(next(iter(targets.values())).signal)
                    for joint, js in targets.items():
                        joints[joint] = attrs.evolve(js, signal=shared)
                else:
                    for joint, js in targets.items():
                        joints[joint] = attrs.evolve(js, signal=fn(js.signal))
                targets = {joint: joints[joint] for joint in targets}
        return attrs.evolve(self, joints=joints)

    def with_laws(self, spec: Mapping[str, Any]) -> GaitProfile:
        """Override symmetric, phase_warp, cadence and gain field by field and limits joint by joint."""
        changes: dict[str, Any] = {}
        if "symmetric" in spec:
            changes["symmetric"] = bool(spec["symmetric"])
        if "limits" in spec:
            if not isinstance(spec["limits"], Mapping):
                raise ValueError(f"{self.name}: limits take a mapping, got {spec['limits']!r}")
            changes["limits"] = {**self.limits, **_parse_limits(self.name, spec["limits"])}
        for law, fields in _LAW_FIELDS.items():
            if law not in spec:
                continue
            raw = spec[law]
            if not isinstance(raw, Mapping) or not set(raw) <= set(fields):
                raise ValueError(f"{self.name}: {law} takes {sorted(fields)}, got {raw!r}")
            changes[law] = attrs.evolve(getattr(self, law), **{fields[k]: float(v) for k, v in raw.items()})
        return attrs.evolve(self, **changes)


_OP: dict[str, Callable[[Any], Callable[[Signal], Signal]]] = {
    "set": lambda new: lambda _old: new,
    "scale": lambda gain: lambda old: old.scaled(gain),
    "offset": lambda add: lambda old: old.plus(add),
}


@attrs.frozen
class PoseProfile:
    """An agent type's locomotion: the walk and run gaits plus the idle gait, None for the hand-written idle sway."""

    walk: GaitProfile
    run: GaitProfile
    idle: GaitProfile | None = None


def dump_gait_profile(profile: GaitProfile) -> str:
    """The library yaml text of a gait profile, floats as repr so they load back to identical doubles."""
    signals: dict[str, Signal] = {}
    for js in profile.joints.values():
        if signals.setdefault(js.key, js.signal) != js.signal:
            raise ValueError(f"{profile.name}: joints bound to {js.key!r} carry different signals, the library form takes one signal per key")
    lines = [f"name: {profile.name}", f"symmetric: {'true' if profile.symmetric else 'false'}"]
    for law, fields in _LAW_FIELDS.items():
        value = getattr(profile, law)
        lines.append(f"{law}: {{" + ", ".join(f"{key}: {getattr(value, field)!r}" for key, field in fields.items()) + "}")
    if profile.limits:
        lines.append("limits:")
        lines += [f"  {joint}: [{lo!r}, {hi!r}]" for joint, (lo, hi) in profile.limits.items()]
    lines.append("signals:")
    for key, signal in signals.items():
        if signal.harmonics:
            lines += [f"  {key}:", f"    mean: {signal.mean!r}", "    harmonics: [" + ", ".join(f"[{amp!r}, {ph!r}]" for amp, ph in signal.harmonics) + "]"]
        else:
            lines.append(f"  {key}: {{mean: {signal.mean!r}}}")
    lines.append("joints:")
    lines += [f"  {joint}: {{signal: {js.key}, shift: {js.shift!r}}}" for joint, js in profile.joints.items()]
    return "\n".join(lines) + "\n"


_LIBRARY_KEYS = {"name", "same_as", "symmetric", "phase_warp", "cadence", "gain", "limits", "signals", "joints"}


def _library_entry(name: str, raw_by_name: Mapping[str, Mapping[str, Any]], resolved: dict[str, GaitProfile], chain: tuple[str, ...]) -> GaitProfile:
    if name in resolved:
        return resolved[name]
    if name not in raw_by_name:
        raise ValueError(f"profile library has no {name!r}, available: {sorted(raw_by_name)}")
    if name in chain:
        raise ValueError(f"profile library same_as cycle: {' -> '.join((*chain, name))}")
    raw = raw_by_name[name]
    if not set(raw) <= _LIBRARY_KEYS:
        raise ValueError(f"profile {name!r} has unknown keys {sorted(set(raw) - _LIBRARY_KEYS)}")
    if "same_as" in raw:
        base = _library_entry(raw["same_as"], raw_by_name, resolved, (*chain, name))
        if "signals" in raw or "joints" in raw:
            raise ValueError(f"profile {name!r}: same_as takes law overrides only, not signals or joints")
        profile = attrs.evolve(base.with_laws(raw), name=name)
    else:
        signals = {key: Signal.parse(spec) for key, spec in raw.get("signals", {}).items()}
        joints: dict[str, JointSignal] = {}
        for joint, binding in raw.get("joints", {}).items():
            if joint not in JOINT_NAMES:
                raise ValueError(f"profile {name!r} binds unknown joint {joint!r}")
            if not isinstance(binding, Mapping) or set(binding) != {"signal", "shift"} or binding["signal"] not in signals:
                raise ValueError(f"profile {name!r}: joint {joint!r} needs {{signal: <key>, shift: <rad>}} with a declared signal, got {binding!r}")
            joints[joint] = JointSignal(binding["signal"], float(binding["shift"]), signals[binding["signal"]])
        profile = GaitProfile(name=name, joints=joints).with_laws(raw)
    resolved[name] = profile
    return profile


@functools.cache
def load_library(library_dir: Path = LIBRARY_DIR) -> Mapping[str, GaitProfile]:
    """Every gait profile yaml in the directory, keyed by file stem, same_as references resolved."""
    raw_by_name: dict[str, Mapping[str, Any]] = {}
    for path in sorted(Path(library_dir).glob("*.yaml")):
        with open(path) as fh:
            raw_by_name[path.stem] = yaml.safe_load(fh) or {}
    resolved: dict[str, GaitProfile] = {}
    for name in raw_by_name:
        _library_entry(name, raw_by_name, resolved, ())
    return resolved


_SECTION_KEYS = {"base", "same_as", "symmetric", "phase_warp", "cadence", "gain", "limits", "joints"}
_SAME_AS_EXCLUDES = {"base", "symmetric", "joints"}


def _resolve_gait(gait: str, spec: Mapping[str, Any] | None, library: Mapping[str, GaitProfile], walk: GaitProfile | None) -> GaitProfile:
    if spec is None:
        spec = {}
    if not isinstance(spec, Mapping) or not set(spec) <= _SECTION_KEYS:
        raise ValueError(f"pose.{gait} takes {sorted(_SECTION_KEYS)}, got {spec!r}")
    default = DEFAULT_RUN if gait == "run" else DEFAULT_WALK
    if spec.get("same_as") is not None:
        if gait != "run" or spec["same_as"] != "walk" or walk is None:
            raise ValueError(f"pose.{gait}: same_as is only valid as pose.run: {{same_as: walk}}")
        if set(spec) & _SAME_AS_EXCLUDES:
            raise ValueError(f"pose.run: same_as takes law overrides only ({sorted(_SECTION_KEYS - _SAME_AS_EXCLUDES - {'same_as'})}), not {sorted(set(spec) & _SAME_AS_EXCLUDES)}")
        return attrs.evolve(walk, name=gait, gain=attrs.evolve(walk.gain, factor=library[default].gain.factor)).with_laws(spec)
    name = spec.get("base", default)
    if name not in library:
        raise ValueError(f"pose.{gait}: unknown base {name!r}, available: {sorted(library)}")
    return library[name].with_laws(spec).with_ops(spec.get("joints") or {})


def resolve_pose_profile(pose_section: Mapping[str, Any] | None, library_dir: Path = LIBRARY_DIR) -> PoseProfile:
    """Build an agent type's PoseProfile from its `pose:` section (walk, run, idle), missing gaits take the library defaults."""
    section = dict(pose_section or {})
    if not set(section) <= set(GAITS):
        raise ValueError(f"pose takes {GAITS}, got {sorted(section)}")
    library = load_library(Path(library_dir))
    walk = _resolve_gait("walk", section.get("walk"), library, None)
    run = _resolve_gait("run", section.get("run"), library, walk)
    idle_spec = section.get("idle") or {}
    if not isinstance(idle_spec, Mapping):
        raise ValueError(f"pose.idle takes {{base: {IDLE} | <library name>, ...}}, got {idle_spec!r}")
    if idle_spec.get("base", IDLE) == IDLE:
        if set(idle_spec) - {"base"}:
            raise ValueError(f"pose.idle: the builtin {IDLE} takes no overrides, got {sorted(set(idle_spec) - {'base'})}")
        return PoseProfile(walk=walk, run=run, idle=None)
    return PoseProfile(walk=walk, run=run, idle=_resolve_gait("idle", idle_spec, library, None))


@functools.cache
def default_profile() -> PoseProfile:
    """The library defaults: walk_cmu_12_01 walking, the same clip at the run gain, the hand-written idle sway."""
    return resolve_pose_profile({})


def _merge_pose_sections(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {gait: dict(spec or {}) for gait, spec in parent.items()}
    for gait, spec in child.items():
        target = merged.setdefault(gait, {})
        for key, value in (spec or {}).items():
            if key in ("joints", "limits"):
                target[key] = {**target.get(key, {}), **(value or {})}
            else:
                target[key] = value
    return merged


def _locomotion_split(agent_type: str, raw: Mapping[str, Any]) -> float | None:
    """The type's own locomotion.phase_warp.split, a bare number or the mean of a distribution."""
    locomotion = raw.get("locomotion") or {}
    if not isinstance(locomotion, Mapping) or not isinstance(locomotion.get("phase_warp") or {}, Mapping):
        raise ValueError(f"agent type {agent_type!r}: locomotion and locomotion.phase_warp must be mappings")
    split = (locomotion.get("phase_warp") or {}).get("split")
    if isinstance(split, Mapping):
        split = split.get("mean")
    if split is None:
        return None
    if isinstance(split, bool) or not isinstance(split, int | float):
        raise ValueError(f"agent type {agent_type!r}: locomotion.phase_warp.split must be a number or {{mean, ...}}, got {split!r}")
    return float(split)


def _agent_type_pose(agent_type: str, builtin_dir: Path | None, chain: tuple[str, ...]) -> tuple[dict[str, Any], float | None] | None:
    if "/" in agent_type or agent_type.endswith(".yaml"):
        path = Path(agent_type)
    elif builtin_dir is not None:
        path = Path(builtin_dir) / f"{agent_type}.yaml"
    else:
        return None
    if not path.is_file():
        return None
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    section = raw.get("pose") or {}
    split = _locomotion_split(agent_type, raw)
    parent = raw.get("extends")
    if parent is None:
        return dict(section), split
    if parent in chain:
        raise ValueError(f"agent type extends cycle: {' -> '.join((*chain, agent_type, parent))}")
    inherited, inherited_split = _agent_type_pose(str(parent), builtin_dir, (*chain, agent_type)) or ({}, None)
    return _merge_pose_sections(inherited, section), inherited_split if split is None else split


def agent_type_pose_section(agent_type: str, builtin_dir: Path | None) -> dict[str, Any] | None:
    """The `pose:` section of an agent type yaml (a path or a builtin name) merged along `extends:`, walk's phase_warp defaulting to locomotion.phase_warp.split, None when the type cannot be found."""
    resolved = _agent_type_pose(agent_type, builtin_dir, ())
    if resolved is None:
        return None
    section, split = resolved
    walk = section.get("walk") or {}
    if split is not None and "phase_warp" not in walk:
        section["walk"] = {**walk, "phase_warp": {"split": split}}
    return section
