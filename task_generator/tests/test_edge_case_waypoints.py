from __future__ import annotations

import math

import pytest

from task_generator.tasks.obstacles.edge_case.waypoints import Mode, Plan, WaypointError

HOMES = {"a": (0.0, 0.0), "b": (10.0, 0.0)}


def test_a_plan_owns_agents() -> None:
    with pytest.raises(WaypointError, match="owns no agents"):
        Plan(mode=Mode.SHUFFLE, homes={})


def test_a_rally_needs_a_target_per_agent() -> None:
    with pytest.raises(WaypointError, match="target for every"):
        Plan(mode=Mode.RALLY, homes=HOMES, targets={"a": (1.0, 1.0)})


def test_resume_must_name_an_owned_agent() -> None:
    with pytest.raises(WaypointError, match="does not own"):
        Plan(mode=Mode.HOLD, homes=HOMES, resume={"zed": (1.0, 1.0)})


def test_shuffle_points_are_deterministic_and_on_the_radius() -> None:
    plan = Plan(mode=Mode.SHUFFLE, homes=HOMES, radius=2.0)
    first, again = plan.shuffle_routes(0), plan.shuffle_routes(0)
    assert first == again
    for name, (home, pts) in ((n, (HOMES[n], first[n])) for n in HOMES):
        assert math.dist(home, pts[0]) == pytest.approx(2.0), name
    assert plan.shuffle_routes(1)["a"] != first["a"]


def test_hold_and_resume_routes() -> None:
    plan = Plan(mode=Mode.HOLD, homes=HOMES, points={"a": (3.0, 3.0)}, resume={"b": (9.0, 9.0)})
    assert plan.hold_routes() == {"a": [(3.0, 3.0)], "b": [(10.0, 0.0)]}
    assert plan.resume_routes() == {"b": [(9.0, 9.0)]}


def test_as_dict_is_stable() -> None:
    plan = Plan(mode=Mode.RALLY, homes=HOMES, targets={"a": (1.0, 2.0), "b": (1.0, 2.0)}, label="rally#0")
    d = plan.as_dict()
    assert d["mode"] == "rally" and d["agents"] == ["a", "b"] and d["targets"] == {"a": [1.0, 2.0], "b": [1.0, 2.0]}
