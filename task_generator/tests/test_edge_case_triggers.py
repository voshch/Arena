"""Firing a case on where the robot is rather than on a clock.

The predicate half. Whether the gate actually holds a plan back is covered in
`ros/test_edge_case_waypoint_driver.py`.
"""

from __future__ import annotations

import pytest

from task_generator.tasks.obstacles.edge_case.triggers import (
    Trigger, TriggerError, centroid, distance_to, parse,
)


def test_a_within_trigger_fires_on_approach():
    assert parse({"robot_within": 3.0}).satisfied(2.9) is True
    assert parse({"robot_within": 3.0}).satisfied(3.1) is False


def test_a_beyond_trigger_is_the_mirror():
    assert parse({"robot_beyond": 3.0}).satisfied(4.0) is True
    assert parse({"robot_beyond": 3.0}).satisfied(1.0) is False


def test_an_unknown_distance_never_fires():
    """A trigger that fired because it could not see the robot would report a case that ran
    when nothing had happened."""
    assert parse({"robot_within": 3.0}).satisfied(None) is False
    assert parse({"robot_within": 3.0}).satisfied(float("nan")) is False


def test_a_trigger_needs_a_condition():
    with pytest.raises(TriggerError, match="needs one of"):
        parse({"of": "lobby"})


def test_both_sides_at_once_is_refused():
    """A band is not a proximity condition, and which edge fires would be ambiguous."""
    with pytest.raises(TriggerError, match="not both"):
        parse({"robot_within": 3.0, "robot_beyond": 1.0})


def test_a_typo_is_refused_rather_than_ignored():
    with pytest.raises(TriggerError, match="unknown key"):
        parse({"robot_wihin": 3.0})


def test_a_then_clause_is_refused_with_the_reason():
    """The backend allocates a new agent on every spawn, so a knob cannot be applied to a
    running agent - and a `then:` that never fires is worse than one that is refused."""
    with pytest.raises(TriggerError, match="mid-episode agent-type replacement"):
        parse({"robot_within": 3.0, "then": {"knob": "interaction_radius"}})


def test_a_non_positive_distance_is_refused():
    with pytest.raises(TriggerError, match="positive distance"):
        parse({"robot_within": 0})


def test_a_non_numeric_distance_names_itself():
    with pytest.raises(TriggerError, match="trigger.robot_within"):
        parse({"robot_within": "close"})


def test_a_non_mapping_is_refused():
    with pytest.raises(TriggerError, match="expected a mapping"):
        parse(["robot_within"])


def test_an_explicit_reference_survives_unresolved():
    """Resolving a zone name needs the world, which this half does not have."""
    assert parse({"robot_within": 3.0, "of": "lobby"}).of == "lobby"


def test_the_description_names_the_side_and_the_point():
    described = Trigger(robot_within=3.0).describe((1.0, 2.0))
    assert "within 3m" in described and "1.00,2.00" in described


def test_the_centroid_is_the_default_reference():
    assert centroid([(0.0, 0.0), (2.0, 0.0), (1.0, 3.0)]) == (1.0, 1.0)


def test_an_empty_crowd_has_no_centroid():
    assert centroid([]) is None


def test_distance_is_unknown_when_either_end_is():
    assert distance_to(None, (0.0, 0.0)) is None
    assert distance_to((0.0, 0.0), None) is None
    assert distance_to((3.0, 4.0), (0.0, 0.0)) == 5.0
