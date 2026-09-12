from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _make_registry():
    from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry

    return Registry[str, object]()


def test_register_and_get_basic():
    reg = _make_registry()

    @reg.register("alpha")
    async def _loader(*args, **kwargs):
        return "alpha_result"

    result = asyncio.run(reg.get("alpha"))
    assert result == "alpha_result"


def test_register_duplicate_raises():
    reg = _make_registry()

    @reg.register("dup")
    async def _first():
        return "first"

    with pytest.raises(ValueError, match="already registered"):

        @reg.register("dup")
        async def _second():
            return "second"


def test_get_nonexistent_raises():
    reg = _make_registry()
    with pytest.raises(KeyError):
        asyncio.run(reg.get("nonexistent"))


def test_get_passes_args():
    reg = _make_registry()

    @reg.register("adder")
    async def _adder(a, b):
        return a + b

    result = asyncio.run(reg.get("adder", 3, 4))
    assert result == 7


def test_get_passes_kwargs():
    reg = _make_registry()

    @reg.register("kwarg_fn")
    async def _fn(x=0, y=0):
        return x * y

    result = asyncio.run(reg.get("kwarg_fn", x=3, y=5))
    assert result == 15


def test_init_from_entries():
    async def _base():
        return "base"

    from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry

    reg = Registry[str, object](entries={"base_key": _base})
    result = asyncio.run(reg.get("base_key"))
    assert result == "base"


def test_init_none_entries_empty():
    from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry

    reg = Registry[str, object](entries=None)
    with pytest.raises(KeyError):
        asyncio.run(reg.get("anything"))


def test_register_multiple_distinct():
    reg = _make_registry()

    @reg.register("one")
    async def _one():
        return 1

    @reg.register("two")
    async def _two():
        return 2

    assert asyncio.run(reg.get("one")) == 1
    assert asyncio.run(reg.get("two")) == 2


def test_registered_callable_invoked_each_time():
    reg = _make_registry()
    call_count = [0]

    @reg.register("counter")
    async def _counter():
        call_count[0] += 1
        return call_count[0]

    r1 = asyncio.run(reg.get("counter"))
    r2 = asyncio.run(reg.get("counter"))
    assert r1 == 1
    assert r2 == 2
