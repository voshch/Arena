"""Expansion of camera specs to atomic actions.

A camera spec is a `params:` defaults block plus a `timeline:` of steps, where each
step `{name: params}` is a verb (`PRIMITIVES`) or another camera asset of the bundle,
or a `cameras:` map of named cameras (a rig). A bundle is `{name: spec}` for every
asset a root reaches, built by `arena_simulation_setup.tree.assets.Camera.collect`.
`resolve` expands a name eagerly to a flat list of atomic actions, substituting
`${param}` references and guarding against cycles.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import typing
from collections.abc import Mapping

from .curves import FOV_DEFAULT
from .mounts import Mount
from .registry import PRIMITIVES

if typing.TYPE_CHECKING:
    from .camera import _Action

_LOGGER = logging.getLogger("arena_cam")
_TOKEN = re.compile(r"\$\{([^}]+)\}")
_CONTEXT = re.compile(r"env|robot|(robot|human|face)\..+")
_CAMERA_KEYS = frozenset({"source", "reference", "timeline", "mount"})
_SOURCES = ("sim", "viz")


@dataclasses.dataclass
class RigCamera:
    """One camera of a take: its name, empty for a lone camera, its timeline or mount, and the surfaces it films through."""

    name: str = ""
    actions: list[_Action] = dataclasses.field(default_factory=list)
    source: str | None = None
    mount: Mount | None = None


@dataclasses.dataclass
class Rig:
    """The cameras of a `cameras:` spec, their shared projection and the camera live playback plays."""

    cameras: list[RigCamera]
    projection: str | None
    live: str


def _param(key: str, args: dict) -> object:
    if key not in args:
        raise ValueError(f"shot parameter ${{{key}}} not provided")
    return args[key]


def substitute(raw: object, args: dict) -> object:
    """Resolve `${param}` references through a step spec, preserving value types.

    A string that is exactly `${key}` becomes the parameter's value (list, number,
    string), a string with `${key}` embedded in text is interpolated as text. A
    context token nobody passed stays text, the cam fills it per surface.
    """
    if isinstance(raw, dict):
        return {k: substitute(v, args) for k, v in raw.items()}
    if isinstance(raw, list):
        return [substitute(v, args) for v in raw]
    if isinstance(raw, str):
        whole = _TOKEN.fullmatch(raw)
        if whole and (whole.group(1) in args or not _CONTEXT.fullmatch(whole.group(1))):
            return _param(whole.group(1), args)
        return _TOKEN.sub(lambda m: m.group(0) if m.group(1) not in args and _CONTEXT.fullmatch(m.group(1)) else str(_param(m.group(1), args)), raw)
    return raw


def _reference_steps(ref: dict) -> list[dict]:
    if ref.get("entity"):
        return [{"track": {"entity": ref["entity"], "mode": ref.get("mode", "full")}}]
    if ref.get("latch"):
        return [{"latch": {}}]
    if ref.get("pose"):
        return [{"reference": {"pose": ref["pose"], "mode": ref.get("mode", "full")}}]
    return []


def desugar(spec: dict) -> list[dict]:
    """Flatten a shot's projection/reference header into leading timeline steps.

    Folding the header into ordinary steps is what lets a shot compose when
    included: an included shot that sets a reference emits a `track`/`reference`
    step, sticky exactly as if authored inline.
    """
    steps: list[dict] = []
    projection = spec.get("projection")
    if projection:
        steps.append({"projection": projection})
    ref = spec.get("reference")
    if ref:
        steps.extend(_reference_steps(ref))
    steps.extend(spec.get("timeline") or [])
    return steps


class _Params:
    """A params dict that records which keys a verb's `from_params` reads, so
    `resolve` can warn about the rest. A mistyped param is otherwise silent."""

    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self._seen: set[str] = set()

    def __contains__(self, key: str) -> bool:
        self._seen.add(key)
        return key in self._raw

    def __getitem__(self, key: str) -> object:
        self._seen.add(key)
        return self._raw[key]

    def get(self, key: str, default: object = None) -> object:
        self._seen.add(key)
        return self._raw.get(key, default)

    def unknown(self) -> set[str]:
        return set(self._raw) - self._seen


def _from_params(name: str, params: object) -> _Action:
    try:
        return PRIMITIVES[name].from_params(params)
    except KeyError as e:
        raise ValueError(f"cam verb {name!r}: missing required param {e.args[0]!r}") from None


def check_bundle(bundle: Mapping[str, dict]) -> None:
    """Refuse a bundle with a name that shadows a verb."""
    shadowing = sorted(set(bundle) & set(PRIMITIVES))
    if shadowing:
        raise ValueError(f"camera assets {shadowing} shadow a camera verb of the same name")


def _bounds(raw: object) -> tuple[float, float | None]:
    if not isinstance(raw, list | tuple) or len(raw) != 2:
        raise ValueError(f"slice takes [from, to] in seconds, got {raw!r}")
    start = float(raw[0])
    end = None if raw[1] is None else float(raw[1])
    if start < 0.0 or (end is not None and start >= end):
        raise ValueError(f"slice needs 0 <= from < to, got {raw!r}")
    return start, end


def resolve(name: str, spec: object = None, stack: tuple[str, ...] = (), bundle: Mapping[str, dict] | None = None) -> list[_Action]:
    """Expand a verb-or-asset name + params to a flat list of atomic actions.

    Verbs win over assets, an unknown name lists both vocabularies. Recursion into
    assets is eager, so cycles surface here as a load-time error.
    """
    bundle = bundle or {}
    if name in PRIMITIVES:
        if spec is None:
            return [_from_params(name, {})]
        if isinstance(spec, str) and name == "projection":
            return [_from_params(name, spec)]
        if not isinstance(spec, dict):
            raise ValueError(f"cam verb {name!r}: params must be a mapping, got {type(spec).__name__}")
        if "slice" in spec:
            raise ValueError(f"cam verb {name!r}: slice applies to a step that names a camera asset")
        params = _Params(spec)
        action = _from_params(name, params)
        unknown = params.unknown()
        if unknown:
            _LOGGER.warning("cam: verb %r ignores unknown param(s) %s", name, sorted(unknown))
        return [action]
    if name in bundle:
        if name in stack:
            raise ValueError("shot cycle: " + " -> ".join([*stack, name]))
        if "cameras" in bundle[name]:
            raise ValueError(f"rig {name!r} is a root only, it cannot be a step of a timeline")
        given = dict(spec) if isinstance(spec, dict) else spec
        bounds = _bounds(given.pop("slice")) if isinstance(given, dict) and "slice" in given else None
        actions = expand_spec(bundle[name], given, (*stack, name), bundle)
        if bounds is None:
            return actions
        from .camera import _Slice

        return [_Slice(actions, *bounds)]
    raise ValueError(f"unknown camera name: {name!r} (verbs: {sorted(PRIMITIVES)}, shots: {sorted(bundle)})")


def expand_spec(spec: dict, params: object, stack: tuple[str, ...], bundle: Mapping[str, dict] | None = None) -> list[_Action]:
    """Expand a parsed shot spec under merged (defaults + caller) params."""
    given = params or {}
    if not isinstance(given, dict):
        raise ValueError(f"cam shot: params must be a mapping, got {type(given).__name__}")
    if "slice" in spec.get("params", {}):
        raise ValueError("cam shot: slice is a step key, not a param")
    args = {**spec.get("params", {}), **given}
    actions: list[_Action] = []
    for step in desugar(spec):
        ((name, raw),) = step.items()
        actions.extend(resolve(name, substitute(raw, args), stack, bundle))
    return actions


def _mount(name: str, entry: dict, args: dict) -> Mount:
    raw = substitute(entry, args)
    assert isinstance(raw, dict)
    stray = sorted(set(raw) - {"frame", "at", "pitch", "fov"})
    if stray or "frame" not in raw:
        raise ValueError(f"camera {name!r}: mount takes frame, at, pitch and fov, got {sorted(raw)}")
    x, y, z = raw.get("at", (0.0, 0.0, 0.0))
    return Mount(str(raw["frame"]), float(raw.get("fov", FOV_DEFAULT)), (float(x), float(y), float(z)), float(raw.get("pitch", 0.0)))


def _rig_camera(name: str, entry: object, args: dict, stack: tuple[str, ...], bundle: Mapping[str, dict]) -> RigCamera:
    if not isinstance(entry, dict):
        raise ValueError(f"camera {name!r}: expected a mapping, got {entry!r}")
    stray = sorted(set(entry) - _CAMERA_KEYS)
    if stray:
        raise ValueError(f"camera {name!r}: unknown keys {stray}")
    source = entry.get("source")
    if source is not None and source not in _SOURCES:
        raise ValueError(f"camera {name!r}: source is one of {_SOURCES}, got {source!r}")
    if "mount" in entry:
        if "reference" in entry or "timeline" in entry:
            raise ValueError(f"camera {name!r}: mount excludes reference and timeline")
        return RigCamera(name, source="sim", mount=_mount(name, entry["mount"], args))
    actions: list[_Action] = []
    for step in desugar({"reference": entry.get("reference"), "timeline": entry.get("timeline")}):
        ((step_name, raw),) = step.items()
        actions.extend(resolve(step_name, substitute(raw, args), stack, bundle))
    verbs = {atom.verb for action in actions for atom in action.atoms()}
    refused = sorted(verbs & {"projection", "latch"})
    if refused:
        raise ValueError(f"camera {name!r}: {refused} cannot run in a rig, a rig shares the projection and its surfaces cannot hold a latched reference")
    return RigCamera(name, actions, source)


def expand_rig(root: str, spec: dict, params: object, bundle: Mapping[str, dict]) -> Rig:
    """Expand a `cameras:` spec under merged params, timeline cameras first since mounts film at their stamps."""
    if "timeline" in spec:
        raise ValueError(f"rig {root!r}: timeline and cameras exclude each other")
    given = params or {}
    if not isinstance(given, dict):
        raise ValueError(f"rig {root!r}: params must be a mapping, got {type(given).__name__}")
    args = {**spec.get("params", {}), **given}
    entries = spec["cameras"] or {}
    cameras = [_rig_camera(name, entry, args, (root,), bundle) for name, entry in entries.items()]
    timelined = [camera.name for camera in cameras if camera.mount is None]
    if not timelined:
        raise ValueError(f"rig {root!r}: needs a camera with a timeline")
    live = spec.get("live", timelined[0])
    if live not in timelined:
        raise ValueError(f"rig {root!r}: live names one of {timelined}, got {live!r}")
    cameras.sort(key=lambda camera: camera.mount is not None)
    return Rig(cameras, spec.get("projection"), live)
