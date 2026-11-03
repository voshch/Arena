"""Tests for the robot:= bracket grammar helpers (.claude/parametrized-robots.md sec2.2)."""

from __future__ import annotations

import pytest
from arena_runtime.env.robot_manager.robots_manager import (
    desugar_robot_entry,
    split_robot_arg,
)


class TestSplitRobotArg:
    def test_bracket_aware_split(self):
        assert split_robot_arg("jackal[2, lidar=sick], husky") == [
            "jackal[2, lidar=sick]",
            "husky",
        ]

    def test_no_brackets_plain_split(self):
        assert split_robot_arg("jackal, husky") == ["jackal", "husky"]

    def test_single_entry(self):
        assert split_robot_arg("jackal") == ["jackal"]

    def test_empty_tokens_dropped(self):
        assert split_robot_arg("jackal,, husky,") == ["jackal", "husky"]

    def test_entries_are_stripped(self):
        assert split_robot_arg("  jackal  ,  husky  ") == ["jackal", "husky"]

    def test_nested_brackets_in_list_values_not_split(self):
        assert split_robot_arg("jackal[lidar=[sick,vlp16]], husky") == [
            "jackal[lidar=[sick,vlp16]]",
            "husky",
        ]


class TestDesugarRobotEntry:
    def test_bare_model(self):
        assert desugar_robot_entry("jackal") == {"robot": "jackal"}

    def test_count_only(self):
        assert desugar_robot_entry("jackal[3]") == {"robot": "jackal", "count": 3}

    def test_count_and_adapter_kind(self):
        assert desugar_robot_entry("jackal[2, mobile.adapter=drl]") == {
            "robot": "jackal",
            "count": 2,
            "mobile.adapter": "drl",
        }

    def test_list_value(self):
        assert desugar_robot_entry("jackal[lidar=[sick,vlp16]]") == {
            "robot": "jackal",
            "lidar": ["sick", "vlp16"],
        }

    def test_repeated_keys_accumulate(self):
        assert desugar_robot_entry("jackal[lidar=sick, lidar=vlp16]") == {
            "robot": "jackal",
            "lidar": ["sick", "vlp16"],
        }

    def test_keys_pass_through_with_dots(self):
        result = desugar_robot_entry("jackal[mobile.adapter=drl]")
        assert "mobile.adapter" in result
        assert result["mobile.adapter"] == "drl"

    def test_keys_pass_through_with_at_mount(self):
        result = desugar_robot_entry("jackal[lidar@front=sick]")
        assert "lidar@front" in result
        assert result["lidar@front"] == "sick"

    def test_two_integer_items_raises(self):
        with pytest.raises(RuntimeError):
            desugar_robot_entry("jackal[2,3]")

    def test_bare_non_integer_word_raises(self):
        with pytest.raises(RuntimeError):
            desugar_robot_entry("jackal[nolidar]")

    def test_unbalanced_open_bracket_raises(self):
        with pytest.raises(RuntimeError):
            desugar_robot_entry("jackal[2")

    def test_unbalanced_close_bracket_raises(self):
        with pytest.raises(RuntimeError):
            desugar_robot_entry("jackal2]")
