from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def test_obstacle_layer_ordering():
    from task_generator.simulators.human.utils import ObstacleLayer

    assert ObstacleLayer.UNUSED < ObstacleLayer.INUSE < ObstacleLayer.WORLD


def test_obstacle_layer_integer_values():
    from task_generator.simulators.human.utils import ObstacleLayer

    assert ObstacleLayer.UNUSED == 0
    assert ObstacleLayer.INUSE == 1
    assert ObstacleLayer.WORLD == 2


def test_create_or_get_creates_on_first_call():
    from task_generator.simulators.human.utils import KnownObstacles

    known = KnownObstacles()
    entry = known.create_or_get("alpha", "first_payload")
    assert entry.obstacle == "first_payload"


def test_create_or_get_is_idempotent():
    from task_generator.simulators.human.utils import KnownObstacles

    known = KnownObstacles()
    first = known.create_or_get("alpha", "first_payload")
    second = known.create_or_get("alpha", "different_payload")
    assert second is first
    assert second.obstacle == "first_payload"


def test_forget_removes_entry():
    from task_generator.simulators.human.utils import KnownObstacles

    known = KnownObstacles()
    known.create_or_get("beta", "data")
    known.forget("beta")
    assert known.get("beta") is None
    assert "beta" not in known


def test_forget_unknown_name_is_noop():
    from task_generator.simulators.human.utils import KnownObstacles

    known = KnownObstacles()
    known.forget("nonexistent")


def test_get_returns_none_for_unknown():
    from task_generator.simulators.human.utils import KnownObstacles

    known = KnownObstacles()
    assert known.get("missing") is None


def test_clear_empties_container():
    from task_generator.simulators.human.utils import KnownObstacles

    known = KnownObstacles()
    known.create_or_get("x", 1)
    known.create_or_get("y", 2)
    known.clear()
    assert list(known.keys()) == []
    assert "x" not in known
    assert "y" not in known


def test_contains_keys_values_items():
    from task_generator.simulators.human.utils import KnownObstacles

    known = KnownObstacles()
    a = known.create_or_get("a", 10)
    b = known.create_or_get("b", 20)
    assert "a" in known
    assert "b" in known
    assert set(known.keys()) == {"a", "b"}
    assert a in known.values()
    assert b in known.values()
    assert ("a", a) in known.items()
    assert ("b", b) in known.items()


def test_create_or_get_accepts_layer_kwarg():
    from task_generator.simulators.human.utils import KnownObstacles, ObstacleLayer

    known = KnownObstacles()
    entry = known.create_or_get("world_obs", "payload", layer=ObstacleLayer.WORLD)
    assert entry.layer is ObstacleLayer.WORLD
