from __future__ import annotations

import math

import pytest
from task_generator.shared import Position
from task_generator.tasks.obstacles.edge_case.geometry import (
    Encounter,
    GeometryError,
    angle_offsets,
    rotate,
    segment_is_free,
    solve_encounter,
)

#: A 20 m eastbound leg at 1 m/s, so times and distances are the same number.
ROBOT_START = (0.0, 0.0)
ROBOT_GOAL = (20.0, 0.0)


def solve(**kwargs: object) -> Encounter:
    params: dict[str, object] = {"robot_speed": 1.0, "ped_speed": 1.0}
    params.update(kwargs)
    return solve_encounter((ROBOT_START, ROBOT_GOAL), **params)  # type: ignore[arg-type]


# Co-arrival
# ----------


def test_both_reach_the_encounter_point_together() -> None:
    """The whole point of the solver: walking from `spawn` toward `goal` at `ped_speed`
    puts the pedestrian at `encounter` exactly when the robot gets there."""
    enc = solve(encounter_at=0.5, approach_angle=180.0)

    assert enc.t_encounter == pytest.approx(10.0)
    assert enc.encounter == pytest.approx((10.0, 0.0))
    # Pedestrian walks `ped_speed * t` along spawn->goal.
    assert math.dist(enc.spawn, enc.encounter) == pytest.approx(enc.ped_speed * enc.t_encounter)


def test_co_arrival_offset_delays_the_pedestrian_by_starting_it_further_out() -> None:
    """There is no spawn_tick in the Arena adapter, so a late arrival is bought with
    distance. +2 s at 1 m/s must be exactly 2 m of extra approach."""
    base = solve(encounter_at=0.5, approach_angle=180.0)
    late = solve(encounter_at=0.5, approach_angle=180.0, co_arrival_offset=2.0)

    extra = math.dist(late.spawn, late.encounter) - math.dist(base.spawn, base.encounter)
    assert extra == pytest.approx(2.0)
    assert late.designed_pet == pytest.approx(2.0)


def test_negative_offset_past_the_encounter_is_refused() -> None:
    """An offset more negative than the travel time would have the pedestrian start beyond
    the encounter point and walk away from it - silently inverting the experiment."""
    with pytest.raises(GeometryError, match="cancels the pedestrian's approach"):
        solve(encounter_at=0.5, co_arrival_offset=-10.0)


# Angle convention
# ----------------


def test_head_on_spawns_ahead_of_the_robot_on_its_own_path() -> None:
    enc = solve(encounter_at=0.5, approach_angle=180.0)
    # Ahead of the robot, still on the y=0 line, walking back toward it.
    assert enc.spawn[0] > enc.encounter[0]
    assert enc.spawn[1] == pytest.approx(0.0)
    assert enc.goal[0] < enc.encounter[0]


def test_crossing_spawns_off_the_robots_path() -> None:
    enc = solve(encounter_at=0.5, approach_angle=90.0)
    assert enc.encounter == pytest.approx((10.0, 0.0))
    # 90 deg CCW from an eastbound heading is northbound travel, so it starts to the south.
    assert enc.spawn[1] < 0.0
    assert enc.goal[1] > 0.0
    assert enc.spawn[0] == pytest.approx(enc.encounter[0])


def test_same_direction_spawns_behind_the_encounter() -> None:
    enc = solve(encounter_at=0.5, approach_angle=0.0, ped_speed=0.5)
    assert enc.spawn[0] < enc.encounter[0]
    assert enc.spawn[1] == pytest.approx(0.0)


def test_rotate_by_180_negates() -> None:
    assert rotate((1.0, 0.0), 180.0) == pytest.approx((-1.0, 0.0))
    assert rotate((0.0, 1.0), 90.0) == pytest.approx((-1.0, 0.0))


# lead_out
# --------


def test_goal_is_beyond_the_encounter_never_short_of_it() -> None:
    """The pedestrian must walk *through* the meeting point. Stopping in it turns a moving
    encounter into a static obstacle, which is a different experiment."""
    enc = solve(encounter_at=0.5, approach_angle=180.0, lead_out=4.0)
    assert math.dist(enc.encounter, enc.goal) == pytest.approx(4.0)
    # goal is further along the pedestrian's travel direction than the encounter is
    assert math.dist(enc.spawn, enc.goal) > math.dist(enc.spawn, enc.encounter)


# encounter_at
# ------------


def test_encounter_at_places_the_meeting_along_the_leg() -> None:
    early = solve(encounter_at=0.25)
    late = solve(encounter_at=0.75)
    assert early.encounter[0] == pytest.approx(5.0)
    assert late.encounter[0] == pytest.approx(15.0)
    assert early.t_encounter == pytest.approx(5.0)
    assert late.t_encounter == pytest.approx(15.0)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_encounter_at_must_be_strictly_inside_the_leg(bad: float) -> None:
    with pytest.raises(GeometryError, match="encounter_at"):
        solve(encounter_at=bad)


def test_coincident_start_and_goal_is_refused() -> None:
    with pytest.raises(GeometryError, match="no route to intercept"):
        solve_encounter(((1.0, 1.0), (1.0, 1.0)), robot_speed=1.0, ped_speed=1.0)


# Free-space search
# -----------------


def blocked_north(pt: Position) -> bool:
    """Everything with y > 0 is a wall."""
    return pt.y <= 1e-9


def test_unobstructed_case_gets_exactly_what_it_asked_for() -> None:
    enc = solve(encounter_at=0.5, approach_angle=180.0, is_valid=lambda _pt: True)
    assert enc.angle_achieved == pytest.approx(180.0)
    assert enc.retries == 0
    assert not enc.deviated


def test_blocked_angle_is_retried_and_the_deviation_is_reported() -> None:
    """A requested angle that lands in geometry must be moved *and said so*. Sweeping one
    angle while running another is the silent-failure this records against."""
    enc = solve(encounter_at=0.5, approach_angle=90.0, angle_search=180.0, angle_step=15.0, is_valid=blocked_north)

    assert enc.deviated
    assert enc.retries > 0
    assert enc.angle_requested == pytest.approx(90.0)
    # whatever it settled on must actually keep the whole walk south of the wall
    assert enc.spawn[1] <= 1e-9
    assert enc.goal[1] <= 1e-9


def test_angle_search_zero_refuses_rather_than_deviating() -> None:
    with pytest.raises(GeometryError, match="no navigable interception"):
        solve(encounter_at=0.5, approach_angle=90.0, angle_search=0.0, is_valid=blocked_north)


def only_the_leg_line(pt: Position) -> bool:
    """Free space is exactly the robot's leg and nothing either side of it."""
    return abs(pt.y) < 1e-6 and -0.1 <= pt.x <= 20.1


def test_abort_reason_names_the_factors_tried() -> None:
    """The leg itself is clear here, so the failure really is about the pedestrian's walk and
    the per-candidate detail is the useful thing to print."""
    with pytest.raises(GeometryError) as excinfo:
        solve(encounter_at=0.5, approach_angle=90.0, angle_search=0.0, is_valid=only_the_leg_line)
    message = str(excinfo.value)
    assert "no navigable interception" in message
    assert "angle=90" in message
    assert "encounter_at=0.5" in message


def test_spawn_on_top_of_the_robot_is_rejected() -> None:
    """0 deg with a fast pedestrian starts it behind the encounter - possibly right on the
    robot. An agent materialising inside the robot is a broken trial, not an edge case."""
    with pytest.raises(GeometryError, match="no navigable interception"):
        solve(encounter_at=0.5, approach_angle=0.0, ped_speed=1.0, angle_search=0.0, min_robot_clearance=1.5)


# Helpers
# -------


def test_angle_offsets_try_the_request_first_then_alternate_outward() -> None:
    assert list(angle_offsets(30.0, 15.0)) == [0.0, 15.0, -15.0, 30.0, -30.0]
    assert list(angle_offsets(0.0, 15.0)) == [0.0]
    assert list(angle_offsets(30.0, 0.0)) == [0.0]


def test_segment_scan_catches_a_wall_between_free_endpoints() -> None:
    """Endpoint-only checking would pass this: both ends are free, the middle is not."""

    def gap(pt: Position) -> bool:
        return not (4.0 < pt.x < 6.0)

    assert segment_is_free((0.0, 0.0), (3.0, 0.0), gap)
    assert not segment_is_free((0.0, 0.0), (10.0, 0.0), gap)


def test_no_predicate_means_no_filtering() -> None:
    assert segment_is_free((0.0, 0.0), (10.0, 0.0), None)


def test_designed_ttc_is_the_encounter_time() -> None:
    enc = solve(encounter_at=0.3)
    assert enc.designed_ttc == pytest.approx(enc.t_encounter)


def test_encounter_sweep_tries_the_request_first_then_spreads_both_ways() -> None:
    """A real route bends through doorways, so which parts of the straight-line leg are open
    is not knowable in advance — the sweep has to cover the leg, not guess at a few points."""
    from task_generator.tasks.obstacles.edge_case.geometry import encounter_fractions

    got = list(encounter_fractions(0.5))
    assert got[0] == 0.5
    assert got[1:5] == [0.45, 0.55, 0.4, 0.6]
    assert min(got) <= 0.15, "must reach approaches short enough for a furnished room"
    assert max(got) >= 0.85
    assert all(0.1 <= v <= 0.9 for v in got)
    assert len(got) == len(set(got))


def test_encounter_sweep_stays_in_range_for_an_off_centre_request() -> None:
    from task_generator.tasks.obstacles.edge_case.geometry import encounter_fractions

    got = list(encounter_fractions(0.85))
    assert got[0] == 0.85
    assert all(0.1 <= v <= 0.9 for v in got)


def test_a_corridor_only_reachable_by_a_short_approach_is_solved() -> None:
    """Free space only within 2 m of the encounter point: the requested 0.5 needs a ~5 m
    lead-in that does not fit, so the walker starts on the longest clear stretch (3 m, the
    minimum that is still an approach) and that much later - and says so."""
    E = (10.0, 0.0)

    def near_encounter(pt: Position) -> bool:
        return math.dist((pt.x, pt.y), E) <= 2.0

    enc = solve_encounter(
        (ROBOT_START, ROBOT_GOAL), robot_speed=1.0, ped_speed=1.0,
        encounter_at=0.5, angle_search=0.0, lead_out=1.0, is_valid=near_encounter,
    )
    assert math.dist(enc.spawn, enc.encounter) == pytest.approx(3.0) and enc.depart_at > 0.0
    assert enc.depart_at == pytest.approx(enc.t_encounter - 3.0)
    with pytest.raises(GeometryError):  # and with no 3 m stretch at all, it fails as before
        solve_encounter(
            (ROBOT_START, ROBOT_GOAL), robot_speed=1.0, ped_speed=1.0,
            encounter_at=0.5, angle_search=0.0, lead_out=1.0, is_valid=lambda pt: math.dist((pt.x, pt.y), E) <= 1.0,
        )

    # Free space only in the first 6 m of the leg. Head-on at 0.5 must spawn at x=20 and
    # fail; only a short approach fits, and the ladder has to find it.
    enc = solve_encounter(
        (ROBOT_START, ROBOT_GOAL), robot_speed=1.0, ped_speed=1.0,
        encounter_at=0.5, angle_search=0.0, lead_out=1.0, is_valid=lambda pt: pt.x <= 6.0,
    )
    assert enc.encounter_at_achieved < enc.encounter_at_requested
    assert math.dist(enc.spawn, enc.encounter) < 6.0
    assert enc.deviated, "a moved encounter must be reported, not silently used"


def test_a_leg_through_a_wall_is_diagnosed_not_buried_in_candidates() -> None:
    """The solver designs on the straight start->goal line, but a real route bends through
    doorways. When that line is blocked there is nothing valid to design on, and no amount of
    angle or fraction search helps — so say that, rather than reporting hundreds of failed
    candidates and leaving the reader to infer it.

    Note the check runs only *after* every candidate has failed: a leg blocked in the middle
    is no obstacle to an encounter designed near its open end, and that case must still solve.
    """

    def far_away(pt: Position) -> bool:
        return math.dist((pt.x, pt.y), (50.0, 50.0)) < 2.0

    with pytest.raises(GeometryError, match="straight-line leg .* is itself blocked"):
        solve(encounter_at=0.5, is_valid=far_away)


def test_a_leg_blocked_midway_still_solves_near_its_open_end() -> None:
    """Guards the ordering above: diagnosing the leg early would refuse cases that work."""

    def wall_band(pt: Position) -> bool:
        return not (8.0 < pt.x < 12.0)

    enc = solve(encounter_at=0.5, approach_angle=180.0, is_valid=wall_band)
    assert enc.deviated
    assert not (8.0 < enc.encounter[0] < 12.0)


def test_forbidden_endpoints_do_not_read_as_a_blocked_leg() -> None:
    """The robots scenario mode `forbid`s its own start and goal so the crowd cannot spawn on
    them, and the robot stands at the start — so both endpoints are always occupied. Sampling
    them would report "leg blocked" on every route, turning the diagnosis into noise.

    Here the leg's *interior* is clear but everything off it is not, so every 90 deg candidate
    fails. The report must be about the pedestrian's walk, not the leg.
    """

    def leg_interior_only(pt: Position) -> bool:
        return abs(pt.y) < 1e-6 and 1.0 <= pt.x <= 19.0

    with pytest.raises(GeometryError) as excinfo:
        solve(encounter_at=0.5, approach_angle=90.0, angle_search=0.0, is_valid=leg_interior_only)
    message = str(excinfo.value)
    assert "straight-line leg" not in message, "forbidden endpoints must not read as a blocked leg"
    assert "no navigable interception" in message


def test_a_lead_in_that_does_not_fit_starts_closer_and_later() -> None:
    from task_generator.tasks.obstacles.edge_case.geometry import solve_encounter

    route = [(0.0, 0.0), (20.0, 0.0)]

    def is_valid(p: Position) -> bool:  # a wall 3 m behind the start: x < -3 is occupied
        return p.x >= -3.0

    # An overtaker at 2 m/s reaching fraction 0.5 (10 m, 20 s at 0.5 m/s) needs a 40 m lead-in;
    # only ~13 m of it is free, so it spawns there and waits the difference.
    enc = solve_encounter(route, robot_speed=0.5, ped_speed=2.0, approach_angle=0.0, encounter_at=0.5, is_valid=is_valid, angle_search=0.0)
    assert enc.spawn[0] >= -3.0 and enc.depart_at > 0.0
    assert enc.depart_at == pytest.approx((40.0 - (10.0 - enc.spawn[0])) / 2.0, abs=0.3)
    assert enc.t_encounter == pytest.approx(20.0)


def test_a_spawn_inside_the_goals_reserved_disc_is_moved_closer_and_later() -> None:
    # A head-on walker on a 13 m leg with a lead-in that reaches the goal: the physical walk is
    # clear, but the spawn would sit in the disc the robot mode reserves around its goal.
    route = ((0.0, 0.0), (13.0, 0.0))

    def spawn_valid(pt):
        return math.dist((pt.x, pt.y), (13.0, 0.0)) > 2.0

    enc = solve_encounter(route, robot_speed=1.0, ped_speed=1.2, approach_angle=180.0, encounter_at=0.5, is_valid=None, spawn_valid=spawn_valid, angle_search=0.0)
    assert math.dist(enc.spawn, (13.0, 0.0)) > 2.0, f"spawn {enc.spawn} still inside the reserved disc"
    assert enc.depart_at > 0.0 and math.dist(enc.spawn, enc.encounter) >= 3.0


def test_the_solver_says_when_every_spawn_is_reserved() -> None:
    route = ((0.0, 0.0), (8.0, 0.0))
    with pytest.raises(GeometryError, match="reserved"):
        solve_encounter(route, robot_speed=1.0, ped_speed=1.0, approach_angle=180.0, encounter_at=0.5, is_valid=None, spawn_valid=lambda pt: False, angle_search=0.0)
