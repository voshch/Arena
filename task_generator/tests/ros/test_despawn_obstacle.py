"""Mid-episode obstacle removal — the counterpart `extend` never had.

Everything else in the human-simulator base removes by *layer* at a reset boundary. That is
the right granularity for rebuilding a population and the wrong one for an edge case whose
whole point is that the world changes while the robot is in it: a cart appears in a doorway,
and later it goes away.

These exercise `BaseHumanSimulator.remove_obstacles_by_id` against a stub backend, because
the interesting behaviour is bookkeeping — what resolves, what is forgotten, and what is
reported when the backend cannot do it — not the service call underneath.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


class _Sim:
    """Stands in for the runtime simulator; records what it was asked to delete."""

    def __init__(self):
        self.obstacles_deleted: list[str] = []
        self.pedestrians_deleted: list[str] = []

    async def obstacle_delete(self, obstacles):
        self.obstacles_deleted += [o.name for o in obstacles]
        return [True] * len(obstacles)

    async def pedestrian_delete(self, obstacles):
        self.pedestrians_deleted += [o.name for o in obstacles]
        return [True] * len(obstacles)


class _Logger:
    def debug(self, *a, **kw): ...
    def info(self, *a, **kw): ...
    def warn(self, *a, **kw): ...
    def warning(self, *a, **kw): ...
    def error(self, *a, **kw): ...


def _sim_under_test(*, static_ok=True, dynamic_supported=True):
    """A stand-in carrying only the pieces `remove_obstacles_by_id` touches.

    Not a `BaseHumanSimulator` subclass: that ABC has eleven other abstract methods, none of
    which this code path reaches, and stubbing them all would obscure which ones matter. The
    method under test is invoked unbound against this object instead - the same trick the
    edge_case registry tests use for `TM_EdgeCase._base`.
    """
    from task_generator.simulators.human.utils import KnownObstacles

    calls: dict[str, list] = {"static": [], "dynamic": []}

    class _Backend:
        def __init__(self):
            self._known_obstacles = KnownObstacles()
            self._ped_bus_index = {}
            self._logger = _Logger()
            self._simulator = _Sim()
            self.gate_closed = 0
            self.calls = calls

        async def _remove_obstacles_impl(self, names):
            calls["static"].append(list(names))
            return static_ok

        async def _remove_pedestrians_impl(self):
            raise AssertionError("removing one obstacle must never clear the whole crowd")

        async def _remove_pedestrian_names_impl(self, names):
            if not dynamic_supported:
                # The base default: "this backend cannot remove one agent."
                from task_generator.simulators.human import BaseHumanSimulator

                return await BaseHumanSimulator._remove_pedestrian_names_impl(self, names)
            calls["dynamic"].append(list(names))
            return True

        def _close_stream_gate(self):
            self.gate_closed += 1

    return _Backend()


def _remove(backend, ids):
    from task_generator.simulators.human import BaseHumanSimulator

    return asyncio.run(BaseHumanSimulator.remove_obstacles_by_id(backend, ids))


def _static(name, sim_path=None):
    from task_generator.shared import Obstacle, Pose, Position

    obstacle = Obstacle(name=name, model="cart", pose=Pose(Position(0.0, 0.0)))
    obstacle.sim_path = sim_path or f"env_0/{name}"
    return obstacle


def _dynamic(name, sim_path=None):
    from task_generator.shared import DynamicObstacle, Pose, Position

    obstacle = DynamicObstacle(name=name, model="adult", pose=Pose(Position(0.0, 0.0)), waypoints=[])
    obstacle.sim_path = sim_path or f"env_0/{name}"
    return obstacle


def _register(backend, obstacle):
    from task_generator.simulators.human.utils import ObstacleLayer

    entry = backend._known_obstacles.create_or_get(obstacle.name, obstacle, layer=ObstacleLayer.INUSE)
    entry.spawned = True
    return obstacle


# Resolution
# ----------


def test_removes_by_the_id_the_spawn_returned():
    """`extend` returns `sim_path`, not the bare name. Taking the name here would fail to
    match for exactly the entities this exists to remove."""
    backend = _sim_under_test()
    obstacle = _register(backend, _static("blocking_cart", sim_path="env_0/blocking_cart"))

    removed, missing = _remove(backend, [obstacle.sim_path])

    assert removed == ["env_0/blocking_cart"]
    assert missing == []
    assert backend._simulator.obstacles_deleted == ["blocking_cart"]
    assert backend._known_obstacles.get("blocking_cart") is None


def test_the_plain_name_resolves_too():
    backend = _sim_under_test()
    _register(backend, _static("blocking_cart"))

    removed, missing = _remove(backend, ["blocking_cart"])

    assert removed == ["blocking_cart"]
    assert missing == []


def test_an_unknown_id_is_missing_not_an_error():
    """Already gone is the state the caller wanted."""
    backend = _sim_under_test()

    removed, missing = _remove(backend, ["env_0/never_existed"])

    assert removed == []
    assert missing == ["env_0/never_existed"]
    assert backend._simulator.obstacles_deleted == []


def test_an_empty_request_touches_nothing():
    backend = _sim_under_test()
    _register(backend, _static("cart"))

    assert _remove(backend, []) == ([], [])
    assert backend._known_obstacles.get("cart") is not None


def test_other_obstacles_are_untouched():
    """The failure that would make a timeline useless: removing one thing removes the rest."""
    backend = _sim_under_test()
    _register(backend, _static("cart"))
    _register(backend, _static("sign"))

    _remove(backend, ["env_0/cart"])

    assert backend._known_obstacles.get("sign") is not None
    assert backend._simulator.obstacles_deleted == ["cart"]


# Static vs dynamic
# -----------------


def test_a_pedestrian_goes_through_the_per_agent_path():
    """Not `_remove_pedestrians_impl`, which clears everything - the stub raises if it is
    reached, because despawning one cart must never empty the crowd."""
    backend = _sim_under_test()
    _register(backend, _dynamic("edge_0"))

    removed, missing = _remove(backend, ["env_0/edge_0"])

    assert removed == ["env_0/edge_0"]
    assert missing == []
    assert backend.calls["dynamic"] == [["env_0/edge_0"]]
    assert backend._simulator.pedestrians_deleted == ["edge_0"]


def test_removing_a_pedestrian_closes_the_stream_gate():
    """Otherwise a pedestrian tick can arrive for an agent that no longer exists."""
    backend = _sim_under_test()
    _register(backend, _dynamic("edge_0"))

    _remove(backend, ["env_0/edge_0"])

    assert backend.gate_closed == 1


def test_a_mixed_request_splits_by_kind():
    backend = _sim_under_test()
    _register(backend, _static("cart"))
    _register(backend, _dynamic("edge_0"))

    removed, missing = _remove(backend, ["env_0/cart", "env_0/edge_0"])

    assert sorted(removed) == ["env_0/cart", "env_0/edge_0"]
    assert missing == []
    assert backend.calls["static"] == [["cart"]]
    assert backend.calls["dynamic"] == [["env_0/edge_0"]]


def test_a_backend_that_cannot_remove_one_agent_says_so():
    """The base default is "not supported"; only arena_humansim keeps the id mapping. It
    must report the id as not removed rather than clearing the whole crowd instead."""
    backend = _sim_under_test(dynamic_supported=False)
    _register(backend, _dynamic("edge_0"))

    removed, missing = _remove(backend, ["env_0/edge_0"])

    assert removed == []
    assert missing == ["env_0/edge_0"]
    assert backend._known_obstacles.get("edge_0") is not None, "a failed removal must not forget the entity"


def test_a_failed_static_removal_keeps_the_registry_honest():
    backend = _sim_under_test(static_ok=False)
    _register(backend, _static("cart"))

    removed, missing = _remove(backend, ["env_0/cart"])

    assert removed == []
    assert missing == ["env_0/cart"]
    assert backend._known_obstacles.get("cart") is not None
    assert backend._simulator.obstacles_deleted == [], "nothing may be deleted in the sim if the backend refused"


# TM_Obstacles.retract
# --------------------


def _mode_with_manager(removed, missing):
    from task_generator.tasks.obstacles import TM_Obstacles

    calls: list[list[str]] = []

    class _Manager:
        async def remove_obstacles_by_id(self, ids):
            calls.append(list(ids))
            return list(removed), list(missing)

    mode = object.__new__(type("_TestableMode", (TM_Obstacles,), {}))
    mode._ctx = type("_Ctx", (), {"environment_manager": _Manager()})()
    return mode, calls


def test_retract_reports_whether_the_id_was_there():
    mode, calls = _mode_with_manager(removed=["env_0/cart"], missing=[])
    assert asyncio.run(mode.retract("env_0/cart")) is True
    assert calls == [["env_0/cart"]]


def test_retract_of_something_already_gone_is_false_not_a_raise():
    """A timeline that despawns an object twice, or despawns one whose spawn aborted, is a
    normal thing to write. It must be reportable, not fatal."""
    mode, _ = _mode_with_manager(removed=[], missing=["env_0/cart"])
    assert asyncio.run(mode.retract("env_0/cart")) is False


# The service
# -----------


def _despawn_response(retract_result):
    import task_generator_msgs.srv
    from task_generator.node import TaskGenerator

    class _Obstacles:
        async def retract(self, entity_id):
            if isinstance(retract_result, Exception):
                raise retract_result
            return retract_result

    node = object.__new__(TaskGenerator)
    node._task = type("_Task", (), {"tm_obstacles": _Obstacles()})()
    node._flip_integrity = lambda: None

    request = task_generator_msgs.srv.DespawnObstacle.Request(id="env_0/cart")
    response = task_generator_msgs.srv.DespawnObstacle.Response()
    return asyncio.run(TaskGenerator._cb_despawn_obstacle(node, request, response))


def test_service_reports_found_separately_from_success():
    """An id that resolved to nothing is the state the caller wanted, so `success` stays
    true. Collapsing the two would make a timeline unable to tell "already gone" from "the
    backend refused"."""
    assert _despawn_response(True).found is True
    assert _despawn_response(True).success is True

    gone = _despawn_response(False)
    assert gone.success is True
    assert gone.found is False


def test_service_reports_a_raise_as_failure():
    response = _despawn_response(RuntimeError("simulator is down"))
    assert response.success is False
    assert response.found is False
    assert "simulator is down" in response.error_msg
