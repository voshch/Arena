"""Named, parametrized, recursively-composable camera shots.

A *shot* is a `params:` defaults block plus a `timeline:` of steps, where each
step `{name: params}` resolves to either a verb (`PRIMITIVES`) or another shot
(`SHOTS`). A meta-shot is a shot whose steps are other shots, so there is
no separate "sequence" tier. `resolve` expands a name eagerly to a flat list of
atomic actions, substituting `${param}` references and guarding against cycles.

Data shots ship as `configs/cam/shots/<name>.yaml` and are auto-registered by
file stem at import; the stem may not shadow a verb.
"""

from __future__ import annotations

import logging
import re
import typing
from pathlib import Path

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from .registry import PRIMITIVES

if typing.TYPE_CHECKING:
    from .camera import _Action

SHOTS: dict[str, dict] = {}

_LOGGER = logging.getLogger("arena_cam")
_TOKEN = re.compile(r"\$\{([^}]+)\}")
_DISCOVERED = False


def register(name: str, spec: dict) -> None:
    """Register a shot spec by name, a name that shadows a verb is an error."""
    if name in PRIMITIVES:
        raise ValueError(f"shot {name!r} shadows a camera verb of the same name")
    SHOTS[name] = spec


def discover() -> None:
    """Load YAML shots shipped under the package's share/configs/cam/shots/."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    try:
        share = get_package_share_directory("arena_cam")
    except PackageNotFoundError:
        return
    shots_dir = Path(share) / "configs" / "cam" / "shots"
    if not shots_dir.is_dir():
        return
    for path in sorted(shots_dir.glob("*.yaml")):
        register(path.stem, yaml.safe_load(path.read_text()) or {})


def _param(key: str, args: dict) -> object:
    if key not in args:
        raise ValueError(f"shot parameter ${{{key}}} not provided")
    return args[key]


def substitute(raw: object, args: dict) -> object:
    """Resolve `${param}` references through a step spec, preserving value types.

    A string that is exactly `${key}` becomes the parameter's value (list, number,
    string); a string with `${key}` embedded in text is interpolated as text.
    """
    if isinstance(raw, dict):
        return {k: substitute(v, args) for k, v in raw.items()}
    if isinstance(raw, list):
        return [substitute(v, args) for v in raw]
    if isinstance(raw, str):
        whole = _TOKEN.fullmatch(raw)
        if whole:
            return _param(whole.group(1), args)
        return _TOKEN.sub(lambda m: str(_param(m.group(1), args)), raw)
    return raw


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
        if ref.get("entity"):
            steps.append({"track": {"entity": ref["entity"], "mode": ref.get("mode", "full")}})
        elif ref.get("latch"):
            steps.append({"latch": {}})
        elif ref.get("pose"):
            steps.append({"reference": {"pose": ref["pose"], "mode": ref.get("mode", "full")}})
    steps.extend(spec.get("timeline", []))
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


def resolve(name: str, spec: object = None, stack: tuple[str, ...] = ()) -> list[_Action]:
    """Expand a verb-or-shot name + params to a flat list of atomic actions.

    Verbs win over shots; an unknown name lists both vocabularies. Recursion into
    shots is eager, so cycles surface here as a load-time error.
    """
    if name in PRIMITIVES:
        if not isinstance(spec, dict):
            return [PRIMITIVES[name].from_params(spec if spec is not None else {})]
        params = _Params(spec)
        action = PRIMITIVES[name].from_params(params)
        unknown = params.unknown()
        if unknown:
            _LOGGER.warning("cam: verb %r ignores unknown param(s) %s", name, sorted(unknown))
        return [action]
    if name in SHOTS:
        if name in stack:
            raise ValueError("shot cycle: " + " -> ".join([*stack, name]))
        return expand_spec(SHOTS[name], spec, (*stack, name))
    raise ValueError(f"unknown camera name: {name!r} (verbs: {sorted(PRIMITIVES)}, shots: {sorted(SHOTS)})")


def expand_spec(spec: dict, params: object, stack: tuple[str, ...]) -> list[_Action]:
    """Expand a parsed shot spec under merged (defaults + caller) params."""
    args = {**spec.get("params", {}), **(params or {})}
    actions: list[_Action] = []
    for step in desugar(spec):
        ((name, raw),) = step.items()
        actions.extend(resolve(name, substitute(raw, args), stack))
    return actions
