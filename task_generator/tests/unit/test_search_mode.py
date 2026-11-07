"""The search mode dispatches every robot to its own start and judges it on the hidden human."""

from __future__ import annotations

import pytest

ORIGINS = {"0": (0.0, 0.0), "1": (0.0, -40.0)}


def _entry(**body: object):
    from arena_simulation_setup.tree.World.Scenario import RobotGoal

    return RobotGoal.parse({"start": [2.0, 3.0, 0.5], **body})


def test_the_only_scenario_human_is_the_default_target():
    from task_generator.tasks.robots.search.impl import search_target

    assert search_target(["caller"], "") == "caller"


def test_a_configured_target_picks_one_of_several_humans():
    from task_generator.tasks.robots.search.impl import search_target

    assert search_target(["bystander", "caller"], "caller") == "caller"


def test_several_humans_without_a_target_are_refused():
    from task_generator.tasks.robots.search.impl import search_target

    with pytest.raises(ValueError, match="task.search.target"):
        search_target(["bystander", "caller"], "")


def test_a_target_outside_the_scenario_is_refused():
    from task_generator.tasks.robots.search.impl import search_target

    with pytest.raises(ValueError, match="not a human"):
        search_target(["caller"], "victim")


def test_the_search_goal_dispatches_the_start_and_names_the_target():
    from task_generator.tasks.robots.search.impl import search_goal

    goal = search_goal(_entry(), "caller", 1.5)
    (phase,) = goal.phases
    assert phase.target == "caller"
    assert phase.tolerance_radius == 1.5
    assert (phase.pose.position.x, phase.pose.position.y) == (2.0, 3.0)


def test_the_search_goal_is_judged_but_never_sent_to_the_planner():
    from task_generator.tasks.robots.search.impl import search_goal

    (phase,) = search_goal(_entry(), "caller", 1.5).phases
    assert not phase.dispatch


def test_the_dispatch_pose_lands_on_the_start_on_every_level():
    from arena_runtime.env.task_api import Realizer
    from task_generator.tasks.robots.scenario.impl import localize_phase, to_map_frame
    from task_generator.tasks.robots.search.impl import search_goal

    realizer = Realizer(level_origins=dict(ORIGINS))
    goal = search_goal(_entry(level="1", goal_level="0"), "caller", 1.5)
    start = to_map_frame(goal.start, goal.start_zone, goal.level, realizer)
    (phase,) = [localize_phase(phase, goal.goal_level, realizer) for phase in goal.phases]
    assert (phase.pose.position.x, phase.pose.position.y) == (start.position.x, start.position.y) == (2.0, -37.0)


def test_scenario_phases_are_refused_since_they_would_reach_the_planner():
    from task_generator.tasks.robots.search.impl import search_goal

    with pytest.raises(ValueError, match="start only"):
        search_goal(_entry(phases=[{"goto": "caller"}]), "caller", 1.5)


def test_the_shipped_search_scenario_hides_its_caller_from_the_planner():
    from pathlib import Path

    import yaml
    from arena_simulation_setup.tree.World.Scenario import parse_scenario
    from task_generator.tasks.robots.search.impl import search_goal, search_target

    path = Path(__file__).parents[3] / "arena_simulation_setup/worlds/hospital_1/scenarios/search_caller/scenario.yaml"
    scenario = parse_scenario(yaml.safe_load(path.read_text()))
    target = search_target(sorted(human.name for human in scenario.humans), "")
    (phase,) = search_goal(scenario.robots[0], target, 1.5).phases
    assert target == "caller"
    assert (phase.pose.position.x, phase.pose.position.y) == (10.0, 2.0)
    assert [sound.name for sound in scenario.sounds] == ["call"]


@pytest.mark.parametrize(("distance", "found"), [(1.4, True), (1.6, False)])
def test_the_robot_has_found_the_target_within_the_radius(distance: float, found: bool):
    from arena_simulation_setup.shared.judge import Sample, arrived
    from task_generator.tasks.robots.search.impl import search_goal

    (phase,) = search_goal(_entry(), "caller", 1.5).phases
    sample = Sample(t=0.0, robots={"robot_0": (10.0 + distance, 5.0, 0.0)}, peds={"caller": (10.0, 5.0)})
    assert arrived(phase, sample, {}, "robot_0") is found


def _monitor(**knobs: object):
    from arena_simulation_setup.shared.judge import PhaseMonitor
    from task_generator.tasks.robots.search.impl import search_goal

    (phase,) = search_goal(_entry(), "caller", 1.5, **knobs).phases
    return phase, PhaseMonitor(phase, "robot_0")


def _at(t: float, robot_x: float, walls=None):
    from arena_simulation_setup.shared.judge import Sample

    return Sample(t=t, robots={"robot_0": (robot_x, 5.0, 0.0)}, peds={"caller": (5.0, 5.0)}, walls=walls)


def test_without_line_of_sight_or_dwell_arrival_alone_finds_the_target():
    phase, monitor = _monitor()
    assert phase.until is None
    assert monitor.step(_at(0.0, 6.0), {})


def test_a_dwell_needs_the_target_found_for_its_whole_duration():
    _, monitor = _monitor(dwell_s=2.0)
    assert not monitor.step(_at(0.0, 6.0), {})
    assert not monitor.step(_at(1.0, 6.0), {})
    assert monitor.step(_at(2.5, 6.0), {})


def test_leaving_the_target_restarts_the_dwell():
    _, monitor = _monitor(dwell_s=2.0)
    assert not monitor.step(_at(0.0, 6.0), {})
    assert not monitor.step(_at(1.5, 9.0), {})
    assert not monitor.step(_at(2.0, 6.0), {})
    assert not monitor.step(_at(3.5, 6.0), {})
    assert monitor.step(_at(4.5, 6.0), {})


def _wall_at(x: float):
    import numpy as np
    from arena_simulation_setup.shared.judge import WallGrid

    cells = np.zeros((200, 200), dtype=bool)
    cells[:, int(x / 0.05)] = True
    return WallGrid(cells=cells, origin=(0.0, 0.0), resolution=0.05)


def test_line_of_sight_finds_the_target_only_without_a_wall_between():
    _, blocked = _monitor(line_of_sight=True)
    assert not blocked.step(_at(0.0, 6.0, walls=_wall_at(5.5)), {})
    _, clear = _monitor(line_of_sight=True)
    assert clear.step(_at(0.0, 6.0, walls=_wall_at(8.0)), {})
