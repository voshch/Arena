from __future__ import annotations

import pytest
from task_generator.simulators.human.utils import stimulus_edge


@pytest.mark.parametrize(
    ("previous", "audible", "fires"),
    [
        (None, True, True),
        (False, True, True),
        (True, True, False),
        (False, False, False),
        (True, False, True),
        (None, False, False),
    ],
)
def test_stimulus_edge(previous: bool | None, audible: bool, fires: bool):
    assert stimulus_edge(previous, audible) is fires
