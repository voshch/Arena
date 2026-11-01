"""Scenario `dynamic:` entries to itineraries: dwell and explicit elevator rides."""

from __future__ import annotations

import pytest
from arena_simulation_setup.shared.entities import Waypoint
from arena_runtime.shared import DynamicObstacle, Pose, Position
from arena_runtime.humans.arena_humansim.arena_humansim import PedMechanics
from arena_runtime.humans.itinerary import Stop
from task_generator.tasks.obstacles._itinerary import attach_itinerary

# level 0 at the origin, level 1 laid out 20 m to the right, each level is a 10 x 10 square
ORIGINS = {"": (0.0, 0.0), "0": (0.0, 0.0), "1": (20.0, 0.0)}
CABIN_0 = Stop(9.0, 5.0, ride=True, elevator="elev_0", target="")
CABIN_1 = Stop(21.0, 5.0, ride=True, elevator="elev_1", target="")


def _ride(name: str, toward: Position | None) -> Stop | None:
    return {"elev_0": CABIN_0, "elev_1": CABIN_1}.get(name)


def _ped(waypoints: list[Waypoint], mode: str = "once", level: str = "", x: float = 1.0, **fields: object) -> DynamicObstacle:
    return DynamicObstacle(name="p", model="arenian", pose=Pose(Position(x, 5.0, 0.0)), waypoints=waypoints, level=level, waypoint_mode=mode, **fields)


def _attach(obs: DynamicObstacle) -> DynamicObstacle:
    warnings: list[str] = []
    out = attach_itinerary(
        obs,
        level_origin=lambda level: ORIGINS[level],
        placement=lambda zone, level: "" if zone else level or obs.level,
        elevator_ride=_ride,
        warn=warnings.append,
    )
    out.extra.setdefault("_warnings", warnings)
    return out


def _map_frame(out: DynamicObstacle) -> list[tuple[float, float]]:
    ox, oy = ORIGINS[out.level]
    return [(w.x + ox, w.y + oy) for w in out.waypoints]


def test_waypoints_without_stops_are_untouched() -> None:
    obs = _ped([Waypoint(x=2.0, y=2.0), Waypoint(x=4.0, y=4.0, level="1")])
    assert _attach(obs) is obs


def test_dwell_waypoint_becomes_itinerary_metadata() -> None:
    out = _attach(_ped([Waypoint(x=2.0, y=2.0, dwell=4.0), Waypoint(x=4.0, y=4.0)]))
    assert [(w.x, w.y) for w in out.waypoints] == [(2.0, 2.0), (4.0, 4.0)]
    assert [m["dwell"] for m in out.extra["itinerary"]] == [4.0, 0.0]
    assert all(w.level == "" and w.zone == "" for w in out.waypoints)


def test_level_local_waypoints_come_back_local_to_the_ped_level() -> None:
    out = _attach(_ped([Waypoint(x=2.0, y=2.0, dwell=1.0), Waypoint(x=5.0, y=5.0, level="0")], level="1", x=5.0))
    assert _map_frame(out) == [(22.0, 2.0), (5.0, 5.0)]


def test_explicit_elevator_waypoint_uses_the_cabin_centre() -> None:
    out = _attach(_ped([Waypoint(x=8.2, y=4.4, elevator="elev_0"), Waypoint(x=5.0, y=5.0, level="1")]))
    assert [(w.x, w.y) for w in out.waypoints] == [(9.0, 5.0), (25.0, 5.0)]
    assert [m["ride"] for m in out.extra["itinerary"]] == [True, False]
    assert out.extra["itinerary"][0]["elevator"] == "elev_0"


def test_unknown_elevator_rides_from_the_parsed_point() -> None:
    out = _attach(_ped([Waypoint(x=3.0, y=3.0, elevator="nope")]))
    assert [(w.x, w.y) for w in out.waypoints] == [(3.0, 3.0)]
    assert out.extra["itinerary"][0]["ride"] is True
    assert out.extra["_warnings"]


def test_reverse_becomes_an_explicit_round_trip() -> None:
    out = _attach(_ped([Waypoint(x=2.0, y=5.0, dwell=2.0), Waypoint(x=8.2, y=4.4, elevator="elev_0"), Waypoint(x=5.0, y=5.0, level="1")], mode="reverse"))
    assert [(w.x, w.y) for w in out.waypoints] == [(2.0, 5.0), (9.0, 5.0), (25.0, 5.0), (9.0, 5.0)]
    assert out.waypoint_mode == "repeat"


def test_adapter_reads_the_metadata_back() -> None:
    out = _attach(_ped([Waypoint(x=2.0, y=2.0, dwell=4.0), Waypoint(x=4.0, y=4.0)], spawn_delay=3, semantic_alias={"elevator": "elev_0"}))
    mech = PedMechanics.from_dynamic_obstacle(out)
    assert mech.spawn_delay == 3.0
    assert mech.semantic_alias == {"elevator": "elev_0"}
    stops = mech.stops([(12.0, 2.0), (14.0, 4.0)])
    assert stops == [Stop(12.0, 2.0, dwell=4.0), Stop(14.0, 4.0)]
    assert mech.stops([(1.0, 1.0)]) is None


def test_ped_mechanics_validation() -> None:
    for bad in ({"spawn_delay": -1}, {"semantic_alias": {"a": 3}}, {"semantic_alias": ["a"]}, {"extra": {"itinerary": {"a": 1}}}):
        with pytest.raises(ValueError):
            PedMechanics.from_dynamic_obstacle(_ped([], **bad))
    assert PedMechanics.from_dynamic_obstacle(_ped([])) == PedMechanics()
