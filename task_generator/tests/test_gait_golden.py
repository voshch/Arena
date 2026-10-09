"""Bit-exact replay of the recorded GaitGenerator grid."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from task_generator.simulators.human.gait import WRIST_JOINTS, GaitGenerator

GOLDEN = Path(__file__).resolve().parent / "golden" / "gait_grid.json"
AGENT_IDS = (1, 2, 7)
STATES = (0, 1, 2, 3)
SPEEDS = (0.0, 0.3, 1.0, 1.6, 2.5)
STEPS = 40
DT = 0.05


def _grid() -> dict[str, list[dict[str, str]]]:
    with open(GOLDEN) as fh:
        return json.load(fh)


@pytest.mark.parametrize("speed", SPEEDS)
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("agent_id", AGENT_IDS)
def test_default_profile_matches_golden_grid(agent_id: int, state: int, speed: float) -> None:
    expected = _grid()[f"{agent_id}/{state}/{speed!r}"]
    gen = GaitGenerator()
    for step, frame in enumerate(expected):
        angles = gen.compute(agent_id, state, speed, DT)
        # the grid predates the wrists: the recorded joints replay bit-exact, the wrists the gait never moves stay 0.0
        assert list(angles) == [*frame, *WRIST_JOINTS]
        for name, value in frame.items():
            assert angles[name] == float(value), f"{name} at step {step}"
        assert all(angles[name] == 0.0 for name in WRIST_JOINTS)
