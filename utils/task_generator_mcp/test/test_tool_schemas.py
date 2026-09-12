"""Test: enum constraints in tool JSON schemas are derived from Constants.TaskMode."""

import pytest

from task_generator.constants import Constants
from task_generator_mcp.tools import TM_MODULE_VALUES, TM_OBSTACLES_VALUES, TM_ROBOTS_VALUES


def test_tm_robots_values_match_enum():
    expected = [m.value for m in Constants.TaskMode.TM_Robots] + [""]
    assert TM_ROBOTS_VALUES == expected


def test_tm_obstacles_values_match_enum():
    expected = [m.value for m in Constants.TaskMode.TM_Obstacles] + [""]
    assert TM_OBSTACLES_VALUES == expected


def test_tm_module_values_match_enum():
    expected = [m.value for m in Constants.TaskMode.TM_Module]
    assert TM_MODULE_VALUES == expected


def test_empty_allowed_for_tm_robots():
    assert "" in TM_ROBOTS_VALUES


def test_empty_allowed_for_tm_obstacles():
    assert "" in TM_OBSTACLES_VALUES


def test_empty_not_in_tm_modules():
    # TM_Module does not include "", the field is a list, not a scalar keep-current
    assert "" not in TM_MODULE_VALUES
