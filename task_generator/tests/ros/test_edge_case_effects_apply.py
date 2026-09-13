from __future__ import annotations

import asyncio
import json
import math

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")
    pytest.importorskip("arena_humansim.core.agents.loader")


from edge_case_fixtures import _mode, obstacle  # noqa: E402


def _block(**raw):
    from task_generator.tasks.obstacles.edge_case.scenario_block import parse

    return parse({"id": "case", **raw})


def _records(tmp_path):
    path = tmp_path / "records" / "cases.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _population():
    return [obstacle("walker_a", (5.0, 5.0), [(15.0, 5.0)]), obstacle("walker_b", (15.0, 6.0), [(5.0, 6.0)], speed=1.3)]


def _run(tm):
    return asyncio.run(tm._reset())


def test_empty_block_is_the_base_arm(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block())
    _static, dynamic = _run(tm)
    assert [o.name for o in dynamic] == ["walker_a", "walker_b"]
    (row,) = _records(tmp_path)
    assert row["status"] == "baseline" and row["injected"] == 0 and tm.aborted == []


def test_intercept_adds_one_agent_on_the_route(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "intercept"}]))
    _static, dynamic = _run(tm)
    edge = dynamic[-1]
    assert edge.name == "edge_0" and len(dynamic) == 3
    assert edge.pose.position.x > 10.0 and edge.pose.position.y == pytest.approx(0.0)
    (row,) = _records(tmp_path)
    assert row["status"] == "ok" and row["injected"] == 1 and row["place"] == "intercept"
    assert row["encounter_point"] == pytest.approx([10.0, 0.0])
    assert row["target_agent"] == "edge_0" and row["effects"][0]["type"] == "intercept"
    assert row["inject_type"] == "adult" and row["inject_model"] == "arenian"


def test_intercept_speed_is_the_speed_the_geometry_used(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "intercept", "speed": 0.8}]))
    _static, dynamic = _run(tm)
    (row,) = _records(tmp_path)
    edge = dynamic[-1]
    assert edge.extra["agent"]["desired_velocity"] == pytest.approx(0.8)
    assert math.dist((edge.pose.position.x, edge.pose.position.y), row["encounter_point"]) == pytest.approx(0.8 * row["t_encounter"])


def test_stand_places_the_agent_on_the_route(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "intercept", "after": "stand", "route_fraction": 0.3, "duration": 6}]))
    _static, dynamic = _run(tm)
    edge = dynamic[-1]
    assert (edge.pose.position.x, edge.pose.position.y) == pytest.approx((6.0, 0.0))
    assert edge.extra["waypoint_mode"] == "once"
    (row,) = _records(tmp_path)
    assert row["place"] == "stand"
    (plan, trigger, _ref), = tm.driver.entries
    assert plan.mode == "hold" and plan.duration == 6.0 and trigger is None
    assert plan.resume["edge_0"][0] > 6.0  # released further along the route


def test_stop_facing_holds_at_the_encounter_then_resumes(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "intercept", "after": "stop_facing", "duration": 4}]))
    _run(tm)
    (row,) = _records(tmp_path)
    (plan, _t, _r), = tm.driver.entries
    assert plan.mode == "hold" and plan.at == pytest.approx(row["t_encounter"])
    assert plan.points["edge_0"] == pytest.approx(tuple(row["encounter_point"]))
    assert plan.resume["edge_0"] == pytest.approx(tuple(row["ped_goal"]))


def test_intercept_timing_does_not_shift_with_the_robots_hold(tmp_path) -> None:
    # The driver's clock starts when the robot is under way, so a hold at the start (or nav2's
    # first-plan delay) shifts nothing: the plans count from the robot's actual departure.
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "intercept", "after": "stop_facing", "duration": 4}]))
    tm.node.rosparam._values["task.edge_case_robot.hold"] = 20.0
    _run(tm)
    (row,) = _records(tmp_path)
    (stop, _t, _r), = tm.driver.entries
    assert stop.mode == "hold" and stop.at == pytest.approx(row["t_encounter"]) and row["robot_hold"] == 20.0


def test_follow_tracks_the_robot_after_the_encounter(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "intercept", "after": "follow", "duration": 10}]))
    _run(tm)
    (plan, _t, _r), = tm.driver.entries
    assert plan.mode == "track_robot" and plan.duration == 10.0 and set(plan.homes) == {"edge_0"}


def test_veer_walks_through_the_encounter_to_a_zone(tmp_path) -> None:
    zones = [("lobby", [(0, 10), (4, 10), (4, 14), (0, 14)])]
    tm = _mode(tmp_path, population=_population(), zones=zones, block=_block(effects=[{"type": "intercept", "after": "veer", "veer_to": "lobby"}]))
    _static, dynamic = _run(tm)
    edge = dynamic[-1]
    assert [(w.x, w.y) for w in edge.waypoints][-1] == pytest.approx((2.0, 12.0))
    assert edge.extra["waypoint_mode"] == "once"


def test_rally_over_everyone_includes_the_injected_agent(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[
        {"type": "intercept"}, {"type": "rally", "scope": "all", "target": [30.0, 0.0], "when": {"at": 8}},
    ]))
    _run(tm)
    plans = [p for p, _t, _r in tm.driver.entries]
    (rally,) = [p for p in plans if p.mode == "rally"]
    assert set(rally.homes) == {"walker_a", "walker_b", "edge_0"} and rally.at == 8.0
    assert set(rally.targets.values()) == {(30.0, 0.0)}
    (row,) = _records(tmp_path)
    assert [p["mode"] for p in row["plans"]] == ["rally"]


def test_rally_targets_are_dealt_round_robin(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[
        {"type": "rally", "scope": ["walker_a", "walker_b"], "targets": [[1.0, 1.0], [2.0, 2.0]]},
    ]))
    _run(tm)
    (plan, _t, _r), = tm.driver.entries
    assert plan.targets == {"walker_a": (1.0, 1.0), "walker_b": (2.0, 2.0)}


def test_scope_naming_an_absent_agent_aborts(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "scatter", "scope": ["nobody"]}]))
    _static, dynamic = _run(tm)
    assert len(dynamic) == 2 and tm.aborted and "nobody" in tm.aborted[0]
    (row,) = _records(tmp_path)
    assert row["status"] == "aborted"


def test_injected_scope_with_nothing_injected_aborts(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "hold", "scope": "injected"}]))
    _run(tm)
    assert tm.aborted and "matched none" in tm.aborted[0]


def test_route_fraction_gate_resolves_to_a_point_on_the_route(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[
        {"type": "scatter", "scope": ["walker_a"], "when": {"route_fraction": 0.25}},
    ]))
    _run(tm)
    (plan, trigger, reference), = tm.driver.entries
    assert plan.mode == "shuffle" and plan.once
    assert trigger is not None and trigger.robot_within == 1.5 and reference == pytest.approx((5.0, 0.0))


def test_proximity_gate_defaults_to_the_owned_centroid(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[
        {"type": "scatter", "scope": ["walker_a", "walker_b"], "when": {"robot_within": 3.0}},
    ]))
    _run(tm)
    (_plan, trigger, reference), = tm.driver.entries
    assert trigger.robot_within == 3.0 and reference == pytest.approx((10.0, 5.5))


def test_unknown_zone_aborts_with_the_known_ones(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), zones=[("lobby", [(0, 0), (1, 0), (1, 1), (0, 1)])],
               block=_block(effects=[{"type": "rally", "target": "loby"}]))
    _run(tm)
    assert tm.aborted and "lobby" in tm.aborted[0]


def test_hold_releases_to_the_agents_own_goal(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "hold", "scope": ["walker_b"], "duration": 5}]))
    _run(tm)
    (plan, _t, _r), = tm.driver.entries
    assert plan.resume == {"walker_b": (5.0, 6.0)}


def test_no_robot_aborts_an_intercept(tmp_path) -> None:
    tm = _mode(tmp_path, robots={}, population=_population(), block=_block(effects=[{"type": "intercept"}]))
    _run(tm)
    assert tm.aborted and "no robot" in tm.aborted[0]


def test_profile_adopts_from_the_population_by_sorted_name(tmp_path) -> None:
    population = [obstacle("w", (5.0, 5.0), [(6.0, 5.0)], agent_type="elder"), obstacle("v", (6.0, 5.0), [(7.0, 5.0)], agent_type="hurried")]
    tm = _mode(tmp_path, population=population, block=_block(effects=[{"type": "intercept"}]))
    _static, dynamic = _run(tm)
    assert dynamic[-1].extra["agent"]["agent_type"] == "elder"


def test_unknown_profile_aborts(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "intercept", "profile": "./types/nope.yaml"}]))
    _run(tm)
    assert tm.aborted and "nope.yaml" in tm.aborted[0]


# The RViz prompt input
# ---------------------


def test_no_prompt_is_a_noop(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block())
    tm._apply_prompt()
    assert tm.node.rosparam.sets == []


def test_a_prompt_already_realised_switches_the_scenario_without_regenerating(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(), prompt="  The lights go out.  ")
    tm._prompt_scenarios["The lights go out."] = "normal_a__rviz_abcd1234"
    tm._apply_prompt()
    assert tm.node.rosparam.sets == [("task.scenario.file", "normal_a__rviz_abcd1234")]
    tm.node.rosparam.sets.clear()
    tm._apply_prompt()  # already current: nothing to set
    assert tm.node.rosparam.sets == []


def test_a_prompt_on_a_missing_base_aborts_with_the_path(tmp_path, monkeypatch) -> None:
    from task_generator.tasks.obstacles.edge_case import impl
    from task_generator.tasks.obstacles.edge_case.impl import CaseAborted

    class _View:
        path = str(tmp_path / "worlds" / "pytest_world")

    class _Ident:
        def __init__(self, _name):
            pass

        def resolve_sync(self):
            return _View()

    monkeypatch.setattr(impl, "WorldIdentifier", _Ident)
    tm = _mode(tmp_path, population=_population(), block=_block(), prompt="The lights go out.", scenario="normal_a")
    with pytest.raises(CaseAborted, match="base scenario 'normal_a' not found"):
        tm._apply_prompt()


def test_a_prompt_scenario_name_yields_its_base(tmp_path, monkeypatch) -> None:
    from task_generator.tasks.obstacles.edge_case import impl
    from task_generator.tasks.obstacles.edge_case.impl import CaseAborted

    class _View:
        path = str(tmp_path / "worlds" / "pytest_world")

    monkeypatch.setattr(impl, "WorldIdentifier", lambda _name: type("_I", (), {"resolve_sync": staticmethod(lambda: _View())})())
    tm = _mode(tmp_path, population=_population(), block=_block(), prompt="Two nurses block the corridor.", scenario="break_a__rviz_0badcafe")
    with pytest.raises(CaseAborted, match="base scenario 'break_a' not found"):
        tm._apply_prompt()


# Retune: parameters change when the case runs, not at t=0
# ----------------------------------------------------------


def test_retune_plans_the_new_agent_blocks(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[
        {"type": "retune", "scope": ["walker_a", "walker_b"], "profile": "elder", "speed_scale": 0.5, "when": {"at": 10}},
    ]))
    _run(tm)
    (plan, _t, _r), = tm.driver.entries
    assert plan.mode == "retune" and plan.at == 10.0
    assert plan.retune["walker_a"] == {"agent_type": "elder", "desired_velocity": 0.5}
    assert plan.retune["walker_b"] == {"agent_type": "elder", "desired_velocity": 0.65}
    (row,) = _records(tmp_path)
    assert row["plans"][0]["retune"]["walker_b"]["agent_type"] == "elder"


def test_retune_with_an_unknown_profile_aborts(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "retune", "scope": "all", "profile": "./types/nope.yaml"}]))
    _run(tm)
    assert tm.aborted and "nope.yaml" in tm.aborted[0]



def test_retune_updates_the_agent_in_place(tmp_path) -> None:
    tm = _mode(tmp_path, population=_population(), block=_block(effects=[{"type": "retune", "scope": ["walker_a"], "speed": 0.6}]))
    _run(tm)
    env = tm._ctx.environment_manager
    n = asyncio.run(tm._retune_agents({"walker_a": {"agent_type": "adult", "desired_velocity": 0.6}}))
    assert n == 1 and env.removed == [] and env.spawned == []
    (obs,), = env.updated
    assert obs.name == "walker_a" and obs.extra["agent"] == {"agent_type": "adult", "desired_velocity": 0.6}
    assert [(w.x, w.y) for w in obs.waypoints] == [(15.0, 5.0)]  # the route is the agent's own; nothing respawns
    assert tm._population["walker_a"].extra["agent"]["desired_velocity"] == 0.6
    assert asyncio.run(tm._retune_agents({"ghost": {"agent_type": "adult"}})) == 0
    env.update_ok = False  # a backend that refuses: nothing changed, and the driver counts failures
    assert asyncio.run(tm._retune_agents({"walker_a": {"agent_type": "elder"}})) == 0
    assert tm._population["walker_a"].extra["agent"]["agent_type"] == "adult"


