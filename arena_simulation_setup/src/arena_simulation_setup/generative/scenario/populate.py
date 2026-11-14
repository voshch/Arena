"""The four base scenarios for a world (`normal`, `break`, `night`, `empty`), pedestrians placed by zone name."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from arena_simulation_setup.generative.core.profiles import KINEMATICS, MODELS

from .zones import Role, zones_by_role

BT_AGENT_TYPE = "talker.yaml"

TALKERS = 2
TALKER_SHARE = 0.3
MIN_WALKERS = 2

PEOPLE_PER_ZONE = 2
MIN_CAST = 4

MIN_ROUTE_CLEARANCE = 1.5

ROUTE_STOPS: tuple[tuple[int, float], ...] = ((1, 0.34), (2, 0.30), (3, 0.22), (4, 0.14))

MIN_PED_ROUTE_M = 15.0

#: (mean, std, low, high) desired speed of each builtin type the mix draws, as HumanSim samples it.

MIX: dict[str, dict[str, int]] = {
    "normal": {"adult": 55, "hurried": 15, "distracted": 15, "elder": 10, "child": 5},
    "break": {"adult": 50, "distracted": 30, "elder": 10, "child": 10},
    "night": {"adult": 50, "elder": 30, "distracted": 10, "hurried": 10},
}

BUSY_MIX: dict[str, int] = {"adult": 50, "hurried": 30, "distracted": 15, "elder": 5}

ONLY_IN: dict[str, tuple[str, ...]] = {
    "child": ("residential",),
}

NOT_FOR_PEOPLE = ("wc", "toilet", "bath", "restroom", "lavatory", "stall", "storage", "closet", "elevator")


def for_people(names: Sequence[str]) -> list[str]:
    return [n for n in names if not any(h in str(n).lower() for h in NOT_FOR_PEOPLE)]


_HOSPITAL_HINTS = ("patient", "nurse", "ward", "exam", "pharmacy", "operating", "triage", "consult")

TEMPO_RANGE = (0.92, 1.08)
CROWD_RANGE = (0.8, 1.3)

PROFILES: dict[str, dict[str, float]] = {
    "normal": {"count": 1.0, "speed": 1.0},
    "break": {"count": 1.0, "speed": 1.0},
    "night": {"count": 0.5, "speed": 0.8},
    "empty": {"count": 0.0, "speed": 1.0},
}

MAX_VELOCITY_DEFAULT = 1.5


def polygon(zone: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(c["x"]), float(c["y"])) for c in zone.get("corners", [])]


def centroid(poly: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def zone_centroids(zones: Sequence[Mapping[str, Any]]) -> dict[str, tuple[float, float]]:
    """Zone name to the centroid of its corners, for every named zone with at least three."""
    return {str(z["name"]): centroid(polygon(z)) for z in zones if z.get("name") and len(polygon(z)) >= 3}


def robot_endpoints(robot: Mapping[str, Any]) -> list[Any]:
    """The robot's start, every phase's `goto` and its goal, in order, each a zone name or a pose."""
    out: list[Any] = []
    if (start := robot.get("start")) is not None:
        out.append(start)
    for phase in robot.get("phases") or []:
        if isinstance(phase, Mapping) and (goto := phase.get("goto")) is not None:
            out.append(goto)
    if (goal := robot.get("goal")) is not None:
        out.append(goal)
    return out


def endpoint_xy(endpoint: object, centroids: Mapping[str, tuple[float, float]]) -> tuple[float, float] | None:
    """A zone name as its centroid, a pose as its x and y, None for an unknown zone or no pose."""
    if isinstance(endpoint, str):
        return centroids.get(endpoint)
    if isinstance(endpoint, Sequence) and len(endpoint) >= 2:
        return (float(endpoint[0]), float(endpoint[1]))
    return None


def robot_zones(robot: Mapping[str, Any]) -> frozenset[str]:
    return frozenset(e for e in robot_endpoints(robot) if isinstance(e, str))


def route_points(
    robot: Mapping[str, Any],
    zones: Sequence[Mapping[str, Any]] | None = None,
) -> list[tuple[float, float]]:
    centroids = zone_centroids(zones or [])
    return [pt for e in robot_endpoints(robot) if (pt := endpoint_xy(e, centroids)) is not None]


def clearance(pt: tuple[float, float], route: Sequence[tuple[float, float]]) -> float:
    import shapely

    if len(route) < 2:
        return math.dist(pt, route[0]) if route else float("inf")
    return float(shapely.LineString(route).distance(shapely.Point(pt)))


def nominal_count(zones_of_role: Sequence[str]) -> int:
    return max(1, PEOPLE_PER_ZONE * len(zones_of_role))


def cast_split(n: int) -> tuple[int, int]:
    pairs = max(1, int(n * TALKER_SHARE) // TALKERS)
    while pairs > 0 and n - pairs * TALKERS < MIN_WALKERS:
        pairs -= 1
    talkers = pairs * TALKERS
    return talkers, n - talkers


def _targets(profile: str, by_role: dict[Role, list[str]]) -> tuple[list[str], str]:
    work, brk, private = by_role[Role.WORK], by_role[Role.BREAK], by_role[Role.PRIVATE]
    if work:
        return (brk or work, "workplace") if profile == "break" else (work, "workplace")

    if brk or private:
        return (private or brk, "residential") if profile == "break" else (brk or private, "residential")

    return by_role[Role.TRANSIT], "transit-only"


def _route(
    home: str,
    pool: Sequence[str],
    index: int,
    stops: int,
    centroids: Mapping[str, tuple[float, float]],
) -> list[str]:
    others = [z for z in pool if z != home and z in centroids]
    out: list[str] = []
    length, last = 0.0, centroids.get(home)
    for k in range(len(others)):
        if k >= stops and length >= MIN_PED_ROUTE_M:
            break
        zone = others[(index + k) % len(others)]
        if last is not None:
            length += math.dist(last, centroids[zone])
        last = centroids[zone]
        out.append(zone)
    return out


def _waypoint_mode(legs: int, rng: random.Random) -> str:
    if legs < 3:
        return "reverse"
    return rng.choices(("repeat", "reverse", "random"), weights=(0.5, 0.3, 0.2))[0]


def _pool(by_role: Mapping[Role, Sequence[str]], centroids: Mapping[str, tuple[float, float]]) -> list[str]:
    """Every placeable zone a walker may head for, in role order."""
    return [z for role in (Role.WORK, Role.BREAK, Role.TRANSIT, Role.UNKNOWN) for z in by_role[role] if z in centroids]


def _starts(targets: Sequence[str], by_role: Mapping[Role, Sequence[str]], homes: frozenset[str], centroids: Mapping[str, tuple[float, float]], forbidden: frozenset[str], *, widen: bool) -> list[str]:
    """The zones walkers start in: the screened homes of `targets`, topped up from the pool with `widen`, else `targets`."""
    placed = [z for z in targets if z in centroids]
    starts = for_people([z for z in placed if z in homes])
    if widen and len(starts) < 3 and homes:
        starts += for_people([z for z in _pool(by_role, centroids) if z in homes and z not in starts and z not in forbidden])
    return starts or for_people([z for z in placed if z not in forbidden]) or for_people(placed) or placed


def _walker(name: str, model: str, home: str, agent: dict[str, Any], waypoints: list[str], rng: random.Random) -> dict[str, Any]:
    return {"name": name, "model": model, "pose": home, "agent": agent, "waypoints": waypoints, "waypoint_mode": _waypoint_mode(len(waypoints), rng)}


def build(
    profile: str,
    *,
    zones: Sequence[Mapping[str, Any]],
    robot: Mapping[str, Any],
    overrides: Mapping[str, str] | None = None,
    seed: int = 0,
    homes: frozenset[str] = frozenset(),
    world: str = "",
) -> dict[str, Any]:
    """The `profile` base scenario, deterministic in its inputs, unscreened for empty `homes`."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}, expected one of {sorted(PROFILES)}")

    doc: dict[str, Any] = {"robots": [dict(robot)], "humans": []}
    spec = PROFILES[profile]
    if spec["count"] == 0.0:
        return doc

    by_role = zones_by_role(zones, overrides)
    targets, _basis = _targets(profile, by_role)
    if not targets:
        return doc

    centroids = zone_centroids(zones)
    starts = _starts(targets, by_role, homes, centroids, robot_zones(robot), widen=True)
    if not starts:
        return doc

    pool = _pool(by_role, centroids)
    destinations = for_people(pool) or pool

    rng = random.Random(f"{world}:{profile}:{seed}")
    world_rng = random.Random(f"{world}:{seed}")
    tempo = world_rng.uniform(*TEMPO_RANGE)
    crowd = world_rng.uniform(*CROWD_RANGE)
    models = list(MODELS)
    world_rng.shuffle(models)

    n = max(MIN_CAST, round(nominal_count(targets) * spec["count"] * crowd))
    talkers, _walkers = cast_split(n)
    talker_rooms = list(starts)
    rng_rooms = random.Random(f"{world}:{profile}:{seed}:talkers")
    rng_rooms.shuffle(talker_rooms)

    kind_of_world = world_kind(zones, overrides)
    weights = mix_for(profile, kind_of_world)
    types, type_weights = list(weights), list(weights.values())

    agents: list[dict[str, Any]] = []
    for i in range(n):
        stops = rng.choices([s for s, _ in ROUTE_STOPS], weights=[w for _, w in ROUTE_STOPS])[0]

        if i < talkers:
            home = talker_rooms[(i // TALKERS) % len(talker_rooms)]
            kind = f"./{BT_AGENT_TYPE}"
            speed = speed_for("adult", rng, spec["speed"] * tempo)
            waypoints: list[str] = [home, home]
        else:
            home = starts[i % len(starts)]
            kind = rng.choices(types, weights=type_weights)[0]
            speed = speed_for(kind, rng, spec["speed"] * tempo)
            legs = _route(home, destinations, i, stops, centroids)
            waypoints = [home, *legs] if legs else [home, home]

        agents.append(_walker(f"{profile}_{i + 1}", models[i % len(models)], home, {"agent_type": kind, "desired_velocity": speed}, waypoints, rng))

    walkers = [a for a in agents if not str(a["agent"]["agent_type"]).startswith("./")]
    if walkers and all(a["agent"]["agent_type"] != "elder" for a in walkers):
        last = walkers[-1]
        last["agent"] = {"agent_type": "elder", "desired_velocity": speed_for("elder", random.Random(f"{world}:{profile}:{seed}:elder"), spec["speed"] * tempo)}

    doc["humans"] = agents
    return doc


def bt_type_source() -> Path:
    return Path(__file__).resolve().parent / "agent_types" / BT_AGENT_TYPE


def basis_for(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> str:
    return _targets("normal", zones_by_role(zones, overrides))[1]


def work_zones(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> list[str]:
    return list(_targets("normal", zones_by_role(zones, overrides))[0])


def world_kind(zones: Sequence[Mapping[str, Any]], overrides: Mapping[str, str] | None = None) -> str:
    names = [str(z["name"]).lower() for z in zones if z.get("name")]
    if any(hint in n for n in names for hint in _HOSPITAL_HINTS):
        return "hospital"
    if basis_for(zones, overrides) == "residential":
        return "residential"
    return "office"


def _allowed(weights: Mapping[str, int], kind: str) -> dict[str, int]:
    """`weights` without the types a `kind` of world does not hold, a lone adult when none is left."""
    return {t: w for t, w in weights.items() if kind in ONLY_IN.get(t, (kind,))} or {"adult": 1}


def mix_for(profile: str, kind: str) -> dict[str, int]:
    return _allowed(MIX.get(profile, {"adult": 1}), kind)


def speed_for(kind: str, rng: random.Random, scale: float = 1.0) -> float:
    mean, std, lo, hi = KINEMATICS[kind]
    raw = rng.gauss(mean, std) * scale
    return round(min(max(raw, lo), hi), 2)


def extra_agents(
    n: int,
    *,
    zones: Sequence[Mapping[str, Any]],
    robot: Mapping[str, Any],
    homes: frozenset[str] = frozenset(),
    seed: int = 0,
    world: str = "",
    mix: Mapping[str, int] = BUSY_MIX,
    overrides: Mapping[str, str] | None = None,
    existing: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    by_role = zones_by_role(zones, overrides)
    targets, _basis = _targets("normal", by_role)
    centroids = zone_centroids(zones)
    starts = _starts(targets, by_role, homes, centroids, robot_zones(robot), widen=False)
    if not starts or n <= 0:
        return []
    destinations = for_people(_pool(by_role, centroids))
    rng = random.Random(f"{world}:busy:{seed}")
    models = list(MODELS)
    rng.shuffle(models)
    kind_of_world = world_kind(zones, overrides)
    weights = _allowed(mix, kind_of_world) if mix else mix_for("normal", kind_of_world)
    types, type_weights = list(weights), list(weights.values())
    taken = {str(a.get("name")) for a in ((existing or {}).get("humans") or []) if isinstance(a, Mapping)}
    agents: list[dict[str, Any]] = []
    i = 0
    while len(agents) < n:
        i += 1
        name = f"busy_{i}"
        if name in taken:
            continue
        stops = rng.choices([s for s, _ in ROUTE_STOPS], weights=[w for _, w in ROUTE_STOPS])[0]
        home = starts[i % len(starts)]
        kind = rng.choices(types, weights=type_weights)[0]
        legs = _route(home, destinations, i, stops, centroids)
        waypoints = [home, *legs] if legs else [home, home]
        agents.append(_walker(name, models[i % len(models)], home, {"agent_type": kind, "desired_velocity": speed_for(kind, rng)}, waypoints, rng))
    return agents
