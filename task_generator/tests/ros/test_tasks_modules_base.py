from __future__ import annotations

from types import SimpleNamespace


class _FakeLogger:
    def get_child(self, name):
        return self

    def debug(self, *a, **kw): ...
    def info(self, *a, **kw): ...
    def warn(self, *a, **kw): ...
    def error(self, *a, **kw): ...


def _make_tm_module():
    from task_generator.tasks.modules import TM_Module
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

    task = SimpleNamespace()

    class _ConcreteModule(TM_Module):
        pass

    return _ConcreteModule(
        node=node,
        ctx=ctx,
        namespace=Namespace("test"),
        task=task,
    )


def test_tm_module_before_reset_callable():
    tm = _make_tm_module()
    result = tm.before_reset()
    assert result is None


def test_tm_module_after_reset_callable():
    tm = _make_tm_module()
    result = tm.after_reset()
    assert result is None


def test_tm_module_task_stored():
    from task_generator.tasks.modules import TM_Module
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

    task = SimpleNamespace(name="test_task")

    class _ConcreteModule(TM_Module):
        pass

    tm = _ConcreteModule(
        node=node,
        ctx=ctx,
        namespace=Namespace("test"),
        task=task,
    )
    assert tm._task is task


def test_tm_module_ctx_stored():
    tm = _make_tm_module()
    assert tm._ctx is not None


def test_clear_forbidden_zones_clears_them_before_every_reset() -> None:
    from task_generator.tasks.modules.clear_forbidden_zones.impl import Mod_ClearForbiddenZones

    cleared: list[bool] = []
    ctx = SimpleNamespace(world_manager=SimpleNamespace(forbid_clear=lambda: cleared.append(True)))
    module = Mod_ClearForbiddenZones(ctx=ctx, namespace=SimpleNamespace(), node=SimpleNamespace(), task=SimpleNamespace())
    module.before_reset()
    assert cleared == [True]
