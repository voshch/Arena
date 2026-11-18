"""Tests for the robot:= bracket grammar helpers (.claude/parametrized-robots.md sec2.2)."""

from __future__ import annotations

import pytest
from arena_robots.inline import format_inline_override
from arena_robots.SetupFile import Config
from arena_runtime.env.robot_manager.robot_manager import cap_file_fields, reject_cap_field_overlay
from arena_runtime.env.robot_manager.robots_manager import (
    _morphology_params,
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


class TestCapOverrideGrammar:
    ENTRY = "jackal[mobile.velocity_limits.linear=[-0.5,1.0], mobile.radius=*1.2, mobile.adapter=drl]"

    def test_cap_keys_desugar_to_values_and_scales(self):
        assert desugar_robot_entry(self.ENTRY) == {
            "robot": "jackal",
            "mobile.velocity_limits.linear": [-0.5, 1.0],
            "mobile.radius": {"scale": 1.2},
            "mobile.adapter": "drl",
        }

    def test_desugared_entry_parses_into_config_caps(self):
        (config,) = Config.parse(desugar_robot_entry(self.ENTRY))
        assert config.caps == {"mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.radius": {"scale": 1.2}}
        assert config.adapters == {"mobile": "drl"}

    def test_formatted_overrides_round_trip_through_the_inline_grammar(self):
        (config,) = Config.parse(desugar_robot_entry(self.ENTRY))
        inline = ", ".join(f"{path}={format_inline_override(value)}" for path, value in config.caps.items())
        (again,) = Config.parse(desugar_robot_entry(f"jackal[{inline}]"))
        assert again.caps == config.caps

    def test_duplicate_cap_key_raises(self):
        with pytest.raises(RuntimeError, match="duplicate cap override"):
            desugar_robot_entry("jackal[mobile.radius=0.3, mobile.radius=0.4]")


class TestPartBracketGrammar:
    ENTRY = "jackal[top=arm/fr3[stall_time=2.0,workspace.max=[0.6,0.4,0.9]],top_tool=gripper/franka_hand[max_effort=*0.5],mobile.radius=0.4]"

    def test_part_brackets_desugar_to_raw_part_values(self):
        assert desugar_robot_entry(self.ENTRY) == {
            "robot": "jackal",
            "top": "arm/fr3[stall_time=2.0,workspace.max=[0.6,0.4,0.9]]",
            "top_tool": "gripper/franka_hand[max_effort=*0.5]",
            "mobile.radius": 0.4,
        }

    def test_part_brackets_survive_config_parse_as_directives(self):
        (config,) = Config.parse(desugar_robot_entry(self.ENTRY))
        assert config.parts == {"top": ["arm/fr3[stall_time=2.0,workspace.max=[0.6,0.4,0.9]]"], "top_tool": ["gripper/franka_hand[max_effort=*0.5]"]}
        assert config.caps == {"mobile.radius": 0.4}

    def test_a_list_of_bracketed_parts_splits_at_depth_zero(self):
        assert desugar_robot_entry("jackal[lidar=[sick[range=10.0],vlp16]]") == {"robot": "jackal", "lidar": ["sick[range=10.0]", "vlp16"]}


class TestCapFieldOverlayGuard:
    def test_cap_field_key_names_the_per_instance_syntax(self):
        with pytest.raises(ValueError, match=r"robot:=<model>\[mobile\.velocity_limits\.linear="):
            reject_cap_field_overlay("mobile", ["velocity_limits.linear"], {"velocity_limits", "footprint"})

    def test_adapter_argument_keys_pass(self):
        reject_cap_field_overlay("mobile", ["planner", "mapping"], {"velocity_limits", "footprint"})

    def test_mobile_fields_include_typed_fields_the_file_leaves_unset(self):
        fields = cap_file_fields("mobile", {"radius": 0.3, "nav2": {"x": 1}}, {"nav2", "drl"})
        assert {"radius", "velocity_limits", "footprint", "planar"} <= fields
        assert "nav2" not in fields

    def test_instance_caps_have_no_guarded_fields(self):
        assert cap_file_fields("arm", {"fr3": {}, "moveit": {}}, {"moveit"}) == frozenset()


class TestFleetStateCapParams:
    def test_each_cap_override_is_a_json_valued_cap_entry(self):
        from arena_robots.Robot import RobotIdentifier
        from arena_runtime.shared import Pose, Robot

        robot = Robot(name="slow", pose=Pose(), model=RobotIdentifier.parse("jackal"), caps={"mobile.velocity_limits.linear": [-0.5, 1.0], "mobile.radius": {"scale": 1.2}}, extra={})
        assert {p.key: p.value for p in _morphology_params(robot, robot.caps)} == {"cap:mobile.velocity_limits.linear": "[-0.5, 1.0]", "cap:mobile.radius": '{"scale": 1.2}'}
