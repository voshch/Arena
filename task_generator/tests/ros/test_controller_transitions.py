from __future__ import annotations

import pytest

from task_generator.manager.robot_manager.controller_transitions import next_transition

EXPECTED = ["joint_state_broadcaster", "base_controller", "arm_controller"]


@pytest.mark.parametrize(
    ("states", "transition"),
    [
        ({n: "active" for n in EXPECTED}, None),
        ({}, ("load", ("joint_state_broadcaster",))),
        ({"joint_state_broadcaster": "active"}, ("load", ("base_controller",))),
        ({n: "unconfigured" for n in EXPECTED}, ("configure", ("joint_state_broadcaster",))),
        ({"joint_state_broadcaster": "active", "base_controller": "inactive", "arm_controller": "unconfigured"}, ("configure", ("arm_controller",))),
        ({"joint_state_broadcaster": "inactive", "base_controller": "inactive", "arm_controller": "inactive"}, ("activate", tuple(EXPECTED))),
        ({"joint_state_broadcaster": "active", "base_controller": "inactive", "arm_controller": "inactive"}, ("activate", ("base_controller", "arm_controller"))),
        ({"joint_state_broadcaster": "activating", "base_controller": "inactive", "arm_controller": "inactive"}, ("wait", ("joint_state_broadcaster",))),
        ({"joint_state_broadcaster": "active", "base_controller": "deactivating", "arm_controller": "active"}, ("wait", ("base_controller",))),
        ({"joint_state_broadcaster": "unknown", "base_controller": "active", "arm_controller": "active"}, ("wait", ("joint_state_broadcaster",))),
        ({"joint_state_broadcaster": "active", "base_controller": "inactive", "arm_controller": "finalized"}, ("fail", ("arm_controller",))),
        ({"joint_state_broadcaster": "finalized", "arm_controller": "finalized"}, ("fail", ("joint_state_broadcaster", "arm_controller"))),
        ({"joint_state_broadcaster": "active", "arm_controller": "inactive"}, ("load", ("base_controller",))),
    ],
)
def test_next_transition(states: dict[str, str], transition: tuple[str, tuple[str, ...]] | None) -> None:
    assert next_transition(EXPECTED, states) == transition


def test_empty_expected() -> None:
    assert next_transition([], {"stray": "inactive"}) is None


def test_extra_controllers_ignored() -> None:
    states = {n: "active" for n in EXPECTED}
    states["stray"] = "finalized"
    assert next_transition(EXPECTED, states) is None
