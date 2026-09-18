"""Pose parsing across the encodings authored scenario YAML actually uses.

Regression guard for a silent data-loss bug: `pose: {x: 9.5, y: 0.5}` used to fall through
`Pose.parse` to the generic Parseable fallback, which structured it against the
`position`/`orientation` fields, matched neither, and returned the origin without error.
`hospital_1/scenarios/showcase` uses that form, so three of its agents spawned at (0, 0).
"""

from __future__ import annotations

import math

import pytest
from arena_simulation_setup.shared.entities import DynamicObstacle
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Pose


def _pose(value) -> Pose:
    return converter.structure(value, Pose)


def test_pair_sequence():
    p = _pose([9.5, 0.5])
    assert (p.position.x, p.position.y) == (9.5, 0.5)
    assert p.orientation.to_yaw() == pytest.approx(0.0)


def test_triple_sequence_carries_yaw():
    p = _pose([9.5, 0.5, 1.57])
    assert (p.position.x, p.position.y) == (9.5, 0.5)
    assert p.orientation.to_yaw() == pytest.approx(1.57)


def test_nested_mapping():
    p = _pose({"position": {"x": 9.5, "y": 0.5}})
    assert (p.position.x, p.position.y) == (9.5, 0.5)


def test_flat_mapping():
    """The form used by hospital_1/scenarios/showcase."""
    p = _pose({"x": 9.5, "y": 0.5})
    assert (p.position.x, p.position.y) == (9.5, 0.5)


def test_flat_mapping_never_silently_yields_the_origin():
    p = _pose({"x": 9.5, "y": 0.5})
    assert (p.position.x, p.position.y) != (0.0, 0.0)


@pytest.mark.parametrize("key", ["theta", "yaw"])
def test_flat_mapping_with_orientation(key):
    p = _pose({"x": 1.0, "y": 2.0, key: math.pi / 2})
    assert p.orientation.to_yaw() == pytest.approx(math.pi / 2)


def test_flat_mapping_with_z():
    p = _pose({"x": 1.0, "y": 2.0, "z": 3.0})
    assert p.position.z == 3.0


def test_unknown_keys_do_not_take_the_flat_path():
    """A mapping that is not a flat pose must not be silently coerced; it should either
    structure normally or fail loudly."""
    with pytest.raises(Exception):
        _pose({"x": 1.0, "y": 2.0, "bogus": 3.0})


def test_dynamic_obstacle_flat_pose_round_trips():
    """End to end through the entity converter, which is the path scenario loading uses."""
    obs = converter.structure(
        {"name": "queue_patient_1", "model": "male_adult_construction_02", "pose": {"x": 9.5, "y": 0.5}},
        DynamicObstacle,
    )
    assert (obs.pose.position.x, obs.pose.position.y) == (9.5, 0.5)
