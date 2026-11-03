from __future__ import annotations

import pytest

from task_generator.modes import Modes


def test_tm_obstacles_default_returns_random():
    d = Modes.TM_Obstacles.default()
    assert d is Modes.TM_Obstacles.RANDOM


def test_tm_robots_default_returns_explore():
    d = Modes.TM_Robots.default()
    assert d is Modes.TM_Robots.EXPLORE


def test_tm_module_default_returns_empty_set():
    d = Modes.TM_Module.default()
    assert isinstance(d, set)
    assert len(d) == 0


def test_tm_obstacles_values_stable():
    TM = Modes.TM_Obstacles
    assert TM.PARAMETRIZED.value == "parametrized"
    assert TM.RANDOM.value == "random"
    assert TM.SCENARIO.value == "scenario"
    assert TM.ENVIRONMENT.value == "environment"
    assert TM.PROMPT.value == "prompt"


def test_tm_robots_values_stable():
    TM = Modes.TM_Robots
    assert TM.GUIDED.value == "guided"
    assert TM.EXPLORE.value == "explore"
    assert TM.RANDOM.value == "random"
    assert TM.SCENARIO.value == "scenario"


def test_tm_module_values_stable():
    TM = Modes.TM_Module
    assert TM.STAGED.value == "staged"
    assert TM.DYNAMIC_MAP.value == "dynamic_map"
    assert TM.CLEAR_FORBIDDEN_ZONES.value == "clear_forbidden_zones"
    assert TM.RVIZ_UI.value == "rviz_ui"


@pytest.mark.parametrize("axis", [Modes.TM_Robots, Modes.TM_Obstacles, Modes.TM_Module])
def test_an_unknown_mode_name_is_refused_and_a_known_one_round_trips(axis: type) -> None:
    with pytest.raises(ValueError, match="__bad__"):
        axis("__bad__")
    for member in axis:
        assert axis(member.value) is member
