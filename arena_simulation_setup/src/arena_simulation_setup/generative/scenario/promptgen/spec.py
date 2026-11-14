"""The decision record: each step's decision as typed data parsed from a model's JSON answer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import attrs

from arena_simulation_setup.generative.core.profiles import PROFILES, TIER_NAMES, TIERS, ProfileError, profile

A_CASES = ("blackout", "evacuation", "stampede", "busy")
B_CASES = ("erratic_route", "erratic_behavior", "intrinsic")
# where erratic points lie: along the robot's route, or around each wanderer's own start
ERRATIC_AROUND = ("route", "home")
C_CASES = ("blocking", "approach")
D_CASES = ("stubborn", "interaction_unbroken", "formation_break")

C_AFTER = ("continue", "stop_facing", "follow", "veer")
D_VARIANTS = {"interaction_unbroken": ("meet", "standing", "blocking"), "formation_break": ("circle", "line"), "stubborn": ("one",)}
D_BREAKS = ("scatter", "disperse")

#: Meters. Two placements closer than this on the route are a conflict.
CONFLICT_M = 2.0

#: Meters. A formation_break with no `when` breaks when the robot comes this close.
FORMATION_TRIGGER_M = 3.0


def conflict_m(route_length: float) -> float:
    """`CONFLICT_M`, or a quarter of a route too short for that, at least 0.5 m."""
    return min(CONFLICT_M, max(0.5, route_length / 4.0))


FailureKind = Literal["schema", "placement", "applicability", "conflict", "infeasible", "unavailable", "denied"]


def conflicts(a: float, b: float, route_length: float) -> bool:
    """Whether route fractions `a` and `b` of a route `route_length` long are closer than `conflict_m` of it."""
    return abs(a - b) * route_length < conflict_m(route_length)


class SchemaError(ValueError):
    pass


class ConflictError(SchemaError):
    """An answer placing a step on a stretch of the route another step holds."""


@attrs.frozen
class When:
    at: float | None = None
    robot_within: float | None = None
    route_fraction: float | None = None

    def as_effect(self) -> dict[str, Any]:
        if self.robot_within is not None:
            return {"robot_within": self.robot_within}
        if self.route_fraction is not None:
            return {"route_fraction": self.route_fraction}
        return {"at": self.at or 0.0}

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in attrs.asdict(self).items() if v is not None}


PHRASES: tuple[tuple[str, When], ...] = (
    ("from the start", When(at=0.0)),
    ("straight away", When(at=0.0)),
    ("immediately", When(at=0.0)),
    ("as the robot arrives", When(robot_within=3.0)),
    ("when the robot arrives", When(robot_within=3.0)),
    ("as the robot approaches", When(robot_within=3.0)),
    ("when the robot comes near", When(robot_within=3.0)),
    ("as the robot gets close", When(robot_within=3.0)),
    ("after a while", When(at=5.0)),
    ("a little later", When(at=5.0)),
    ("after some time", When(at=5.0)),
    ("after five seconds", When(at=5.0)),
    ("five seconds in", When(at=5.0)),
    ("after ten seconds", When(at=10.0)),
    ("ten seconds in", When(at=10.0)),
    ("after twenty seconds", When(at=20.0)),
    ("twenty seconds in", When(at=20.0)),
    ("after fifteen seconds", When(at=15.0)),
    ("fifteen seconds in", When(at=15.0)),
    ("after half a minute", When(at=30.0)),
    ("half a minute in", When(at=30.0)),
    ("after a minute", When(at=60.0)),
    ("a minute in", When(at=60.0)),
    ("once the robot is halfway", When(route_fraction=0.5)),
    ("when the robot is halfway", When(route_fraction=0.5)),
    ("halfway", When(route_fraction=0.5)),
    ("when the robot is nearly there", When(route_fraction=0.8)),
    ("near the end", When(route_fraction=0.8)),
    ("as soon as the robot sets off", When(route_fraction=0.1)),
    ("early on", When(route_fraction=0.2)),
    ("when the robot comes within a few meters", When(robot_within=4.0)),
    ("when the robot is right there", When(robot_within=1.5)),
)


def parse_when(raw: object, *, default: When, where: str) -> When:
    if raw is None:
        return default
    if isinstance(raw, str):
        text = raw.strip().lower()
        for phrase, when in PHRASES:
            if phrase in text:
                return when
        raise SchemaError(f"{where}.when: {raw!r} is not a known phrase; use a mapping with at / robot_within / route_fraction")
    if not isinstance(raw, Mapping):
        raise SchemaError(f"{where}.when: expected a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - {"at", "robot_within", "route_fraction"})
    if unknown:
        raise SchemaError(f"{where}.when: unknown key(s) {unknown}")
    out: dict[str, float] = {}
    for key in ("at", "robot_within", "route_fraction"):
        if raw.get(key) is not None:
            out[key] = _num(raw[key], where=f"{where}.when.{key}", low=0.0)
    if len(out) > 1:
        raise SchemaError(f"{where}.when: set exactly one of at / robot_within / route_fraction, got {sorted(out)}")
    if "route_fraction" in out and not 0.0 < out["route_fraction"] < 1.0:
        raise SchemaError(f"{where}.when.route_fraction: must be strictly between 0 and 1")
    if "robot_within" in out and out["robot_within"] <= 0:
        raise SchemaError(f"{where}.when.robot_within: must be positive")
    return When(**out) if out else default


def _get(raw: Mapping[str, Any], key: str, default: object) -> object:
    """`raw[key]`, or `default` when it is absent or null."""
    value = raw.get(key)
    return default if value is None else value


def _num(raw: object, *, where: str, low: float | None = None) -> float:
    """`raw` as a finite float, at least `low` when given."""
    if not isinstance(raw, int | float | str):
        raise SchemaError(f"{where}: {raw!r} is not a number")
    try:
        value = float(raw)
    except ValueError:
        raise SchemaError(f"{where}: {raw!r} is not a number") from None
    if not math.isfinite(value):
        raise SchemaError(f"{where}: {raw!r} is not a finite number")
    if low is not None and value < low:
        raise SchemaError(f"{where}: {raw!r} must not be below {low:g}")
    return value


def _tier(raw: object, *, where: str) -> str:
    text = str(raw or "mid").strip().lower()
    if text not in TIER_NAMES:
        raise SchemaError(f"{where}: {raw!r} is not one of {list(TIER_NAMES)}")
    return text


def _choice(raw: object, options: Sequence[str], *, where: str) -> str:
    text = str(raw or "").strip().lower()
    if text not in options:
        raise SchemaError(f"{where}: {raw!r} is not one of {list(options)}")
    return text


def _profile_name(raw: object, *, where: str) -> str:
    try:
        return profile(str(raw or "adult")).name
    except ProfileError as exc:
        raise SchemaError(f"{where}: {exc}") from None


def _names(raw: object, *, where: str) -> tuple[str, ...]:
    """A list of names, a single string as one name."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if not isinstance(raw, Sequence):
        raise SchemaError(f"{where}: expected a list of names, got {type(raw).__name__}")
    return tuple(str(n) for n in raw)


def _reject_unknown(raw: Mapping[str, Any], allowed: Sequence[str], *, where: str) -> None:
    unknown = sorted(set(raw) - set(allowed))
    if unknown:
        raise SchemaError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _mapping(raw: object, *, where: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise SchemaError(f"{where}: expected a mapping, got {type(raw).__name__}")
    return raw


@attrs.frozen(kw_only=True)
class ADecision:
    case: str
    k: float = 2.0
    when: When = When(at=0.0)
    #: Runner-up cases the lexicon saw and dropped.
    dropped: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"case": self.case, "k": self.k, "when": self.when.as_dict(), "dropped": list(self.dropped)}


@attrs.frozen(kw_only=True)
class Selection:
    profile: str = ""
    amount: str = ""
    n: int = 0
    zone: str = ""
    names: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in attrs.asdict(self).items() if v}


def parse_selection(raw: object, *, where: str) -> Selection:
    raw = _mapping(raw, where=where)
    _reject_unknown(raw, ("profile", "amount", "n", "zone", "names"), where=where)
    sel = Selection(
        profile=_profile_name(raw["profile"], where=f"{where}.profile") if raw.get("profile") else "",
        amount=_choice(raw["amount"], ("half", "all_but_one", "all"), where=f"{where}.amount") if raw.get("amount") else "",
        n=int(_num(raw["n"], where=f"{where}.n", low=0.0)) if raw.get("n") is not None else 0,
        zone=str(raw.get("zone") or ""),
        names=_names(raw.get("names"), where=f"{where}.names"),
    )
    if not sel.as_dict():
        raise SchemaError(f"{where}: a selection needs one of profile / amount / n / zone / names")
    return sel


@attrs.frozen(kw_only=True)
class BDecision:
    case: str
    select: Selection
    params: Mapping[str, str] = attrs.Factory(dict)
    period: float = 4.0
    radius: float = 3.0
    around: str = "route"
    when: When = When(at=5.0)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"case": self.case, "select": self.select.as_dict(), "params": dict(self.params), "period": self.period, "radius": self.radius, "around": self.around, "when": self.when.as_dict()}


def parse_b(raw: object) -> BDecision | None:
    if raw is None or raw == "none":
        return None
    raw = _mapping(raw, where="B")
    _reject_unknown(raw, ("case", "select", "params", "period", "radius", "around", "when", "reason"), where="B")
    case = _choice(raw.get("case"), B_CASES, where="B.case")
    params = {}
    for key, tier in _mapping(raw.get("params") or {}, where="B.params").items():
        if key not in TIERS:
            raise SchemaError(f"B.params: {key!r} is not an allowed parameter; allowed: {sorted(TIERS)}")
        params[str(key)] = _tier(tier, where=f"B.params.{key}")
    if case == "intrinsic" and not params:
        raise SchemaError("B: 'intrinsic' needs one to three params")
    if len(params) > 3:
        raise SchemaError("B.params: at most three parameters")
    return BDecision(
        case=case,
        select=parse_selection(raw.get("select"), where="B.select"),
        params=params,
        period=max(1.0, _num(_get(raw, "period", 4.0), where="B.period", low=0.0)),
        radius=max(0.5, _num(_get(raw, "radius", 3.0), where="B.radius", low=0.0)),
        around=_choice(_get(raw, "around", "route"), ERRATIC_AROUND, where="B.around"),
        when=parse_when(raw.get("when"), default=When(at=5.0), where="B"),
        reason=str(raw.get("reason") or ""),
    )


@attrs.frozen(kw_only=True)
class CDecision:
    case: str
    profile: str = "adult"
    #: `blocking`: stand | lead. `approach`: unused.
    mode: str = "stand"
    angle: float = 180.0
    route_fraction: float = 0.5
    speed: str = "mid"
    after: str = "continue"
    duration: float = 10.0
    offset: float = 0.0
    veer_to: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in attrs.asdict(self).items() if k != "reason"}


def parse_c(raw: object) -> CDecision | None:
    if raw is None or raw == "none":
        return None
    raw = _mapping(raw, where="C")
    _reject_unknown(raw, ("case", "profile", "mode", "angle", "route_fraction", "speed", "after", "duration", "offset", "veer_to", "reason"), where="C")
    case = _choice(raw.get("case"), C_CASES, where="C.case")
    fraction = _num(_get(raw, "route_fraction", 0.5), where="C.route_fraction")
    if not 0.0 < fraction < 1.0:
        raise SchemaError("C.route_fraction: must be strictly between 0 and 1")
    after = _choice(_get(raw, "after", "continue"), C_AFTER, where="C.after")
    veer_to = str(raw.get("veer_to") or "")
    if after == "veer" and not veer_to:
        raise SchemaError("C.after 'veer' needs a 'veer_to' zone")
    return CDecision(
        case=case,
        profile=_profile_name(raw.get("profile"), where="C.profile"),
        mode=_choice(_get(raw, "mode", "stand"), ("stand", "lead"), where="C.mode"),
        angle=_num(_get(raw, "angle", 180.0), where="C.angle") % 360.0,
        route_fraction=fraction,
        speed=_tier(_get(raw, "speed", "mid"), where="C.speed"),
        after=after,
        duration=_num(_get(raw, "duration", 10.0), where="C.duration", low=0.0),
        offset=_num(_get(raw, "offset", 0.0), where="C.offset"),
        veer_to=veer_to,
        reason=str(raw.get("reason") or ""),
    )


@attrs.frozen(kw_only=True)
class DDecision:
    case: str
    variant: str = ""
    members: tuple[str, ...] = ("adult", "adult")
    #: Where: `{"route_fraction": f}` or `{"zone": name}`.
    place: Mapping[str, Any] = attrs.Factory(lambda: {"route_fraction": 0.5})
    when: When = When(at=0.0)
    duration: str = "high"
    break_kind: str = "scatter"
    radius: float = 2.0
    #: `stubborn` only.
    profile: str = "adult"
    speed: str = "mid"
    route_fraction: float = 0.5
    reuse: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Re-parseable by `parse_d`."""
        out = attrs.asdict(self)
        out.pop("reason")
        out["break"] = out.pop("break_kind")
        out["when"] = self.when.as_dict()
        out["place"] = dict(self.place)
        return out

    @property
    def fraction(self) -> float | None:
        f = self.place.get("route_fraction") if self.case != "stubborn" else self.route_fraction
        return None if f is None else float(f)


def parse_d(raw: object) -> DDecision | None:
    if raw is None or raw == "none":
        return None
    raw = _mapping(raw, where="D")
    _reject_unknown(raw, ("case", "variant", "members", "place", "when", "duration", "break", "radius", "profile", "speed", "route_fraction", "reuse", "reason"), where="D")
    case = _choice(raw.get("case"), D_CASES, where="D.case")
    variants = D_VARIANTS[case]
    variant = _choice(_get(raw, "variant", variants[0]), variants, where="D.variant")
    members = tuple(_profile_name(m, where=f"D.members[{i}]") for i, m in enumerate(raw.get("members") or ("adult", "adult")))
    limit = 8 if variant == "blocking" else 4
    if case != "stubborn" and not 2 <= len(members) <= limit:
        raise SchemaError(f"D.members: two to {limit} members")
    place_raw = raw.get("place") or {"route_fraction": 0.5}
    place_raw = _mapping(place_raw, where="D.place")
    _reject_unknown(place_raw, ("route_fraction", "zone"), where="D.place")
    place: dict[str, Any] = {}
    if place_raw.get("zone"):
        place["zone"] = str(place_raw["zone"])
    if place_raw.get("route_fraction") is not None:
        f = _num(place_raw["route_fraction"], where="D.place.route_fraction")
        if not 0.0 < f < 1.0:
            raise SchemaError("D.place.route_fraction: must be strictly between 0 and 1")
        place["route_fraction"] = f
    if not place:
        raise SchemaError("D.place: needs a route_fraction or a zone")
    default_when = When(robot_within=FORMATION_TRIGGER_M) if case == "formation_break" else When(at=0.0)
    fraction = _num(_get(raw, "route_fraction", 0.5), where="D.route_fraction")
    if not 0.0 < fraction < 1.0:
        raise SchemaError("D.route_fraction: must be strictly between 0 and 1")
    return DDecision(
        case=case,
        variant=variant,
        members=members,
        place=place,
        when=parse_when(raw.get("when"), default=default_when, where="D"),
        duration=_tier(_get(raw, "duration", "high"), where="D.duration"),
        break_kind=_choice(_get(raw, "break", "scatter"), D_BREAKS, where="D.break"),
        radius=max(0.5, _num(_get(raw, "radius", 2.0), where="D.radius", low=0.0)),
        profile=_profile_name(raw.get("profile"), where="D.profile"),
        speed=_tier(_get(raw, "speed", "mid"), where="D.speed"),
        route_fraction=fraction,
        reuse=_names(raw.get("reuse"), where="D.reuse"),
        reason=str(raw.get("reason") or ""),
    )


@attrs.define
class Failure:
    step: str
    kind: FailureKind
    reason: str

    def as_dict(self) -> dict[str, str]:
        return attrs.asdict(self)


@attrs.define
class Record:
    prompt_id: str
    prompt: str
    world: str
    base: str
    targets: tuple[str, ...] = ()
    a: ADecision | None = None
    d: DDecision | None = None
    b: BDecision | None = None
    c: CDecision | None = None
    #: Step label -> agent names it owns.
    owned: dict[str, list[str]] = attrs.Factory(dict)
    #: Where Step D's formation stands (`center`, `radius`, `variant`, `at`), for the success criterion.
    formation: dict[str, Any] | None = None
    failures: list[Failure] = attrs.Factory(list)
    provenance: dict[str, Any] = attrs.Factory(dict)
    warnings: list[str] = attrs.Factory(list)

    @property
    def steps(self) -> list[str]:
        out = []
        if self.a:
            out.append(f"A:{self.a.case}")
        if self.d:
            out.append(f"D:{self.d.case}")
        if self.b:
            out.append(f"B:{self.b.case}")
        if self.c:
            out.append(f"C:{self.c.case}")
        return out

    @property
    def status(self) -> str:
        if not self.steps:
            return "denied" if any(f.kind == "denied" for f in self.failures) else "infeasible"
        return "partial" if self.failures else "ok"

    def decisions(self) -> dict[str, Any]:
        return {
            "A": self.a.as_dict() if self.a else None,
            "D": self.d.as_dict() if self.d else None,
            "B": self.b.as_dict() if self.b else None,
            "C": self.c.as_dict() if self.c else None,
        }

    def dedup_key(self) -> str:
        """Hash of the decisions with reasons dropped and numbers rounded."""
        doc = json.loads(json.dumps(self.decisions(), sort_keys=True, default=str))

        def rnd(o: object) -> object:
            if isinstance(o, float):
                return round(o * 2) / 2 if abs(o) > 1.0 else round(o, 2)
            if isinstance(o, dict):
                return {k: rnd(v) for k, v in o.items() if k not in ("reason",)}
            if isinstance(o, list):
                return [rnd(v) for v in o]
            return o

        payload = json.dumps({"world": self.world, "base": self.base, "d": rnd(doc)}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "world": self.world,
            "base": self.base,
            "targets": list(self.targets),
            "steps": self.steps,
            "status": self.status,
            "decisions": self.decisions(),
            "owned": dict(self.owned),
            "formation": self.formation,
            "failures": [f.as_dict() for f in self.failures],
            "warnings": list(self.warnings),
            "dedup_key": self.dedup_key(),
            "provenance": dict(self.provenance),
        }


__all__ = [
    "A_CASES",
    "B_CASES",
    "C_CASES",
    "D_CASES",
    "CONFLICT_M",
    "PROFILES",
    "ADecision",
    "BDecision",
    "CDecision",
    "DDecision",
    "ConflictError",
    "Failure",
    "FailureKind",
    "Record",
    "SchemaError",
    "Selection",
    "When",
    "conflicts",
    "parse_b",
    "parse_c",
    "parse_d",
    "parse_selection",
    "parse_when",
]
