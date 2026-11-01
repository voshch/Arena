"""Opt-in mechanism-aware planner map: doors that cannot open for the robot are painted occupied."""

from __future__ import annotations

import io
import math
import typing

import PIL.Image
import PIL.ImageColor
import PIL.ImageDraw
import yaml

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from arena_simulation_setup.shared import Door

    from arena_runtime.sim._semantics import SemanticEntitySnapshot

_PASSABLE_STATES = frozenset({'open', 'opening'})
_GATE_FIELDS = frozenset({'locked', 'blocked'})


def gate_config(door: Door) -> tuple[bool, frozenset[str], bool]:
    """(has gate, authorized agent names, initial locked) from a door's `semantics:` list."""
    is_gate = False
    authorized: set[str] = set()
    locked = False
    for cfg in door.semantics:
        if cfg.name not in _GATE_FIELDS:
            continue
        is_gate = True
        authorized.update(str(a) for a in cfg.params.get('authorized', []) or [])
        if cfg.name == 'locked' and cfg.value is not None:
            locked = bool(cfg.value)
        elif 'locked' in cfg.params:
            locked = bool(cfg.params['locked'])
    return is_gate, frozenset(authorized), locked


def authorized_match(agents: Iterable[str], authorized: frozenset[str]) -> bool:
    """Same matching as the runtime gate: full name or last path segment in `authorized`."""
    return any(agent in authorized or agent.rsplit('/', 1)[-1] in authorized for agent in agents)


def blocked_doors(
    doors: Iterable[tuple[str, Door]],
    snapshots: Sequence[SemanticEntitySnapshot],
    agents: Sequence[str],
) -> frozenset[str]:
    """Realized names of doors that no agent in `agents` can open and that are not open or opening right now."""
    door_state: dict[str, str] = {}
    gate_locked: dict[str, bool] = {}
    for snap in snapshots:
        if snap.kind == 'door' and 'state' in snap.discrete:
            door_state[snap.entity] = snap.discrete['state']
        elif snap.kind == 'gate' and 'locked' in snap.predicates:
            gate_locked[snap.entity] = bool(snap.predicates['locked'])
    blocked: set[str] = set()
    for name, door in doors:
        is_gate, authorized, initial_locked = gate_config(door)
        cannot_trigger = max(door.activation_distance) <= 0.0
        if not cannot_trigger and is_gate:
            cannot_trigger = gate_locked.get(name, initial_locked) and not authorized_match(agents, authorized)
        if cannot_trigger and door_state.get(name, 'closed') not in _PASSABLE_STATES:
            blocked.add(name)
    return frozenset(blocked)


def static_shut_doors(doors: Iterable[Door], agents: Sequence[str]) -> frozenset[str]:
    """Names of doors that stay shut for every agent in `agents` whatever happens at runtime: no activation, or a locked gate they are not authorized for, with neither a pressure plate driving it nor an unlock trigger."""
    listed = list(doors)
    driven = {str(cfg.params['drives']) for door in listed for cfg in door.semantics if 'drives' in cfg.params}
    shut: set[str] = set()
    for door in listed:
        if door.name in driven or any('unlock_on' in cfg.params for cfg in door.semantics):
            continue
        is_gate, authorized, locked = gate_config(door)
        if max(door.activation_distance) <= 0.0 or (is_gate and locked and not authorized_match(agents, authorized)):
            shut.add(door.name)
    return frozenset(shut)


def mask_doors(doors: Iterable[Door], world_doors: Iterable[tuple[str, Door]], blocked: frozenset[str]) -> list[Door]:
    """The doors the door mask frees: all but those served as blocked, matched by their world-unique name."""
    blocked_names = {door.name for name, door in world_doors if name in blocked}
    return [door for door in doors if door.name not in blocked_names]


def paint_segments(png_bytes: bytes, map_yaml_text: str, segments: Iterable[tuple[tuple[float, float], tuple[float, float], float]]) -> bytes:
    """Return the map PNG with each (start, end, width) segment (map-frame metres) painted occupied."""
    map_yaml = yaml.safe_load(map_yaml_text)
    resolution = float(map_yaml['resolution'])
    origin_x, origin_y = (float(v) for v in list(map_yaml.get('origin', [0, 0, 0]))[:2])
    occupied = 255 if int(map_yaml.get('negate', 0)) else 0

    img = PIL.Image.open(io.BytesIO(png_bytes))
    img.load()
    height = img.size[1]
    fill = PIL.ImageColor.getcolor('white' if occupied else 'black', img.mode) if img.mode != 'L' else occupied
    draw = PIL.ImageDraw.Draw(img)

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (x - origin_x) / resolution, height - (y - origin_y) / resolution

    for (sx, sy), (ex, ey), width in segments:
        width_px = max(2, math.ceil(width / resolution))
        draw.line([to_px(sx, sy), to_px(ex, ey)], fill=fill, width=width_px)
        r = width_px / 2.0
        for cx, cy in (to_px(sx, sy), to_px(ex, ey)):
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)

    out = io.BytesIO()
    img.save(out, format='PNG')
    return out.getvalue()
