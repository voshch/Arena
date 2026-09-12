from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


class _FakeLogger:
    def get_child(self, name):
        return self

    def debug(self, *a, **kw): ...
    def info(self, *a, **kw): ...
    def warn(self, *a, **kw): ...
    def error(self, *a, **kw): ...


def _make_task_mode(namespace_str="root"):
    from task_generator.tasks.mode import TaskMode
    from arena_rclpy_mixins.shared import Namespace

    class _FakeConf:
        class Robot:
            class TIMEOUT:
                value = 60

    node = SimpleNamespace(
        conf=_FakeConf(),
        sim_time=SimpleNamespace(sec=0),
        get_logger=lambda: _FakeLogger(),
    )

    ctx = SimpleNamespace(
        robots_manager=SimpleNamespace(managers={}),
        environment_manager=SimpleNamespace(),
        world_manager=SimpleNamespace(),
    )

    class _ConcreteMode(TaskMode):
        pass

    return _ConcreteMode(
        node=node,
        ctx=ctx,
        namespace=Namespace(namespace_str),
    )


def test_namespace_returns_namespace_type():
    from arena_rclpy_mixins.shared import Namespace

    tm = _make_task_mode("root")
    result = tm.namespace("child")
    assert isinstance(result, Namespace)


def test_namespace_no_args():
    tm = _make_task_mode("myns")
    result = tm.namespace()
    assert isinstance(result, str)


def test_namespace_single_arg():
    tm = _make_task_mode("base")
    result = tm.namespace("sub")
    assert "sub" in str(result)


def test_namespace_multiple_args():
    tm = _make_task_mode("base")
    result = tm.namespace("a", "b")
    s = str(result)
    assert "a" in s
    assert "b" in s


def test_ctx_accessible():
    tm = _make_task_mode()
    assert tm._ctx is not None


def test_node_accessible():
    tm = _make_task_mode()
    assert tm.node is not None
