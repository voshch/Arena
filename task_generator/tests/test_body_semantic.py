from arena_simulation_setup.shared.body import SEMANTIC_DOFS

from task_generator.simulators.human.hands import DIGITS
from task_generator.simulators.human.profile import JOINT_NAMES


def test_body_rig_semantic_dofs_are_the_wire_joint_names_then_the_digits() -> None:
    assert SEMANTIC_DOFS[: len(JOINT_NAMES)] == JOINT_NAMES
    assert set(SEMANTIC_DOFS[len(JOINT_NAMES) :]) == {f"{s}_{dof}" for s in "lr" for dof in (*(f"curl_{d}" for d in DIGITS), "spread")}
