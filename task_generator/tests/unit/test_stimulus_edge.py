from __future__ import annotations

from task_generator.simulators.human.utils import stimulus_edge


def test_onset_fires():
    assert stimulus_edge(None, True) is True
    assert stimulus_edge(False, True) is True


def test_repeat_does_not_fire():
    assert stimulus_edge(True, True) is False
    assert stimulus_edge(False, False) is False


def test_offset_fires_after_onset():
    assert stimulus_edge(True, False) is True


def test_first_inaudible_does_not_fire():
    assert stimulus_edge(None, False) is False
