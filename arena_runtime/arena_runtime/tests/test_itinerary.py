"""Ped itineraries: ride insertion for cross-level legs and the leg-by-leg driver (dwell, cabin hold, teleport)."""

from __future__ import annotations

import typing

import pytest
from arena_runtime.humans.itinerary import ItineraryDriver, Phase, Stop, plan_itinerary, semantic_atom


def test_semantic_atom_matches_condition_atoms_without_spaces() -> None:
    assert semantic_atom("elev_a", "cabin_door", "open") == "elev_a.cabin_door==open"


def _ride(name: str, x: float) -> Stop:
    return Stop(x=x, y=0.0, ride=True, elevator=name)


def _hops(table: dict[tuple[str, str], list[Stop]]) -> tuple[typing.Callable[[str, str], list[Stop]], list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    def hops(a: str, b: str) -> list[Stop]:
        calls.append((a, b))
        return table.get((a, b), [])

    return hops, calls


def test_cross_level_leg_gets_the_boarding_cabin() -> None:
    hops, calls = _hops({("0", "1"): [_ride("elev_0", 5.0)]})
    a, b = Stop(1.0, 0.0), Stop(30.0, 0.0)
    planned, mode = plan_itinerary([a, b], ["0", "1"], "once", "0", hops)
    assert planned == [a, _ride("elev_0", 5.0), b]
    assert mode == "once"
    assert calls == [("0", "1")]


def test_start_level_differs_from_first_waypoint() -> None:
    hops, _ = _hops({("1", "0"): [_ride("elev_1", 25.0)]})
    planned, _ = plan_itinerary([Stop(1.0, 0.0)], ["0"], "once", "1", hops)
    assert planned == [_ride("elev_1", 25.0), Stop(1.0, 0.0)]


def test_unknown_levels_never_ride() -> None:
    hops, calls = _hops({})
    plan_itinerary([Stop(1.0, 0.0), Stop(2.0, 0.0)], ["", "1"], "once", "0", hops)
    assert calls == [("0", "1")]
    calls.clear()
    plan_itinerary([Stop(1.0, 0.0), Stop(2.0, 0.0)], ["0", ""], "once", "", hops)
    assert calls == []


def test_explicit_ride_is_not_routed_again() -> None:
    hops, calls = _hops({("0", "1"): [_ride("auto", 9.0)]})
    explicit = _ride("elev_0", 5.0)
    planned, _ = plan_itinerary([Stop(1.0, 0.0), explicit, Stop(30.0, 0.0)], ["0", "0", "1"], "once", "0", hops)
    assert planned == [Stop(1.0, 0.0), explicit, Stop(30.0, 0.0)]
    assert calls == []


def test_repeat_routes_the_way_back() -> None:
    hops, _ = _hops({("0", "1"): [_ride("up", 5.0)], ("1", "0"): [_ride("down", 25.0)]})
    a, b = Stop(1.0, 0.0), Stop(30.0, 0.0)
    planned, mode = plan_itinerary([a, b], ["0", "1"], "repeat", "0", hops)
    assert planned == [a, _ride("up", 5.0), b, _ride("down", 25.0)]
    assert mode == "repeat"


def test_reverse_becomes_an_explicit_round_trip() -> None:
    hops, _ = _hops({})
    a, b, c = Stop(1.0, 0.0), Stop(2.0, 0.0, dwell=3.0), Stop(3.0, 0.0)
    planned, mode = plan_itinerary([a, b, c], ["", "", ""], "reverse", "", hops)
    assert planned == [a, b, c, b]
    assert mode == "repeat"


def test_plan_rejects_misaligned_levels() -> None:
    with pytest.raises(ValueError):
        plan_itinerary([Stop(0.0, 0.0)], [], "once", "", lambda a, b: [])


# driver


def test_driver_hands_legs_up_to_each_dwell() -> None:
    stops = [Stop(0.0, 0.0), Stop(5.0, 0.0, dwell=2.0), Stop(10.0, 0.0), Stop(15.0, 0.0)]
    drv = ItineraryDriver(stops, "once")
    assert drv.start() == [(0.0, 0.0), (5.0, 0.0)]
    assert drv.phase is Phase.MOVING
    assert drv.update((3.0, 0.0), 0.0).waypoints is None
    assert drv.update((4.8, 0.1), 1.0).waypoints is None
    assert drv.phase is Phase.DWELL
    assert drv.update((4.8, 0.1), 2.9).waypoints is None
    step = drv.update((4.8, 0.1), 3.0)
    assert step.waypoints == [(10.0, 0.0), (15.0, 0.0)]
    assert drv.phase is Phase.FREE
    assert drv.update((15.0, 0.0), 10.0).waypoints is None


def test_driver_holds_in_the_cabin_until_teleported() -> None:
    cabin = Stop(5.0, 0.0, ride=True, elevator="elev_a", target="2.elev_b")
    stops = [Stop(0.0, 0.0), cabin, Stop(30.0, 0.0)]
    drv = ItineraryDriver(stops, "once")
    assert drv.start() == [(0.0, 0.0), (5.0, 0.0)]
    step = drv.update((5.1, 0.0), 1.0)
    assert step.boarded == cabin and step.waypoints is None
    assert drv.phase is Phase.RIDE
    for t in (5.0, 50.0, 500.0):
        assert drv.update((5.1, 0.0), t) == drv.update((5.1, 0.0), t + 1) and drv.phase is Phase.RIDE
    assert drv.on_teleport().waypoints == [(30.0, 0.0)]
    assert drv.phase is Phase.FREE


def test_teleport_before_reaching_the_cabin_centre_still_completes_the_ride() -> None:
    drv = ItineraryDriver([Stop(5.0, 0.0, ride=True), Stop(30.0, 0.0)], "once")
    drv.start()
    drv.update((4.0, 0.9), 1.0)
    assert drv.phase is Phase.MOVING
    assert drv.on_teleport().waypoints == [(30.0, 0.0)]


def test_incidental_teleport_is_ignored() -> None:
    drv = ItineraryDriver([Stop(0.0, 0.0), Stop(5.0, 0.0, dwell=3.0)], "once")
    drv.start()
    assert drv.on_teleport().waypoints is None
    assert drv.phase is Phase.MOVING


def test_cyclic_driver_wraps_to_the_first_hold() -> None:
    stops = [Stop(0.0, 0.0, dwell=1.0), Stop(5.0, 0.0), Stop(10.0, 0.0, ride=True)]
    drv = ItineraryDriver(stops, "repeat")
    assert drv.start() == [(0.0, 0.0)]
    drv.update((0.0, 0.0), 0.0)
    assert drv.update((0.0, 0.0), 1.0).waypoints == [(5.0, 0.0), (10.0, 0.0)]
    drv.update((10.0, 0.0), 2.0)
    assert drv.on_teleport().waypoints == [(0.0, 0.0)]
    assert drv.phase is Phase.MOVING


def test_driver_needs_stops() -> None:
    with pytest.raises(ValueError):
        ItineraryDriver([], "once")


def test_stop_dict_round_trip() -> None:
    stop = Stop(1.0, 2.0, dwell=3.0, ride=True, elevator="e", target="1.f")
    assert Stop.from_dict(stop.to_dict()) == stop
    assert Stop.from_dict({"x": 1, "y": 2}) == Stop(1.0, 2.0)
