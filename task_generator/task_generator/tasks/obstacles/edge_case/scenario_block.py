"""Reading the `edge_case:` block out of the active scenario file.

A case is authored inside the scenario it applies to, as a top-level `edge_case:` key. Arena's
scenario loader accepts and ignores unknown top-level keys, so the block is inert to everything
except this mode - an annotated scenario still loads and runs as an ordinary scenario.

The block is the *only* source of a case's configuration. There is no parameter surface to
fill in and no precedence to resolve: a scenario is a complete, self-describing situation, and
what happens at run time is exactly the list of effects it carries.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs
import yaml

from .effects import Effect, EffectError, parse_effects

#: The top-level key. Matches `arena_benchmark`'s writer.
BLOCK_KEY = "edge_case"

#: Provenance fields, carried for humans and for the records; never acted on.
LABELS: tuple[str, ...] = ("id", "note", "base", "prompt_id", "steps", "provenance", "decisions", "owned", "formation")

#: Fields that do something.
ACTIVE: tuple[str, ...] = ("effects", "objects")


class BlockError(Exception):
    """The scenario carries a block that cannot be applied."""


@attrs.frozen
class Block:
    id: str = ""
    note: str = ""
    base: str = ""
    prompt_id: str = ""
    steps: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = attrs.Factory(dict)
    decisions: Mapping[str, Any] = attrs.Factory(dict)
    #: Step label -> agent names it owns, as the generator recorded them.
    owned: Mapping[str, tuple[str, ...]] = attrs.Factory(dict)
    #: Step D's formation as the generator placed it: `centre: [x, y]`, `radius`, `variant`, `at`.
    formation: Mapping[str, Any] = attrs.Factory(dict)
    effects: tuple[Effect, ...] = ()
    #: Object timeline: a shipped name, a path, or `./file.yaml` beside the scenario.
    objects: str = ""

    @property
    def empty(self) -> bool:
        return not self.effects and not self.objects

    def describe(self) -> str:
        kinds = ", ".join(str(e.kind) + (f"({e.label})" if e.label else "") for e in self.effects)
        return f"{self.id or '?'}: {len(self.effects)} effect(s) [{kinds}]" + (f", objects={self.objects}" if self.objects else "")


def scenario_path(world: str, scenario: str) -> Path | None:
    """Filesystem path of a scenario's `scenario.yaml`, or None if it cannot be resolved."""
    if not world or not scenario:
        return None
    try:
        from arena_simulation_setup.tree.World.World import WorldIdentifier  # noqa: PLC0415

        view = WorldIdentifier(world).resolve_sync().scenario(scenario).resolve_sync()
    except Exception:
        return None
    path = Path(getattr(view, "path", "")) / "scenario.yaml"
    return path if path.is_file() else None


def parse(raw: object, *, where: str = BLOCK_KEY) -> Block:
    """A block from its mapping. Strict: an unknown key is a typo that would otherwise be a
    silent no-op, which is the one failure this format exists to refuse."""
    if not isinstance(raw, Mapping):
        raise BlockError(f"{where}: must be a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - set(LABELS) - set(ACTIVE))
    if unknown:
        raise BlockError(f"{where}: unknown key(s) {unknown}; allowed: {sorted((*LABELS, *ACTIVE))}")
    try:
        effects = parse_effects(raw.get("effects"))
    except EffectError as exc:
        raise BlockError(f"{where}: {exc}") from None
    steps = raw.get("steps") or ()
    if isinstance(steps, str):
        steps = (steps,)
    if not isinstance(steps, (list, tuple)):
        raise BlockError(f"{where}.steps: expected a list of step labels")
    provenance = raw.get("provenance") or {}
    decisions = raw.get("decisions") or {}
    owned_raw = raw.get("owned") or {}
    if not isinstance(provenance, Mapping) or not isinstance(decisions, Mapping) or not isinstance(owned_raw, Mapping):
        raise BlockError(f"{where}: provenance, decisions and owned must be mappings")
    owned = {str(k): tuple(str(n) for n in (v or ())) for k, v in owned_raw.items()}
    formation = raw.get("formation") or {}
    if not isinstance(formation, Mapping):
        raise BlockError(f"{where}: formation must be a mapping")
    return Block(
        id=str(raw.get("id") or ""),
        note=str(raw.get("note") or ""),
        base=str(raw.get("base") or ""),
        prompt_id=str(raw.get("prompt_id") or ""),
        steps=tuple(str(s) for s in steps),
        provenance=dict(provenance),
        decisions=dict(decisions),
        owned=owned,
        formation=dict(formation),
        effects=effects,
        objects=str(raw.get("objects") or ""),
    )


def read(path: Path) -> Block | None:
    """The block from a scenario file. None when there is none.

    A malformed block raises: a scenario that *says* it carries a case and does not deliver
    one is the silent no-op this pipeline exists to refuse.
    """
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise BlockError(f"{path}: not valid YAML: {exc}") from None
    if not isinstance(doc, dict) or doc.get(BLOCK_KEY) is None:
        return None
    try:
        return parse(doc[BLOCK_KEY])
    except BlockError as exc:
        raise BlockError(f"{path}: {exc}") from None
