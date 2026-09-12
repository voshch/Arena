from __future__ import annotations

import asyncio

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _make_registry():
    from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry

    return Registry[str, object]()


@given(st.lists(st.text(min_size=1, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))), min_size=1, max_size=20, unique=True))
@settings(max_examples=50)
def test_all_registered_names_retrievable(names):
    reg = _make_registry()

    for name in names:
        captured = name

        async def _loader(_n=captured):
            return _n

        reg.register(captured)(_loader)

    for name in names:
        result = asyncio.run(reg.get(name))
        assert result == name


@given(st.lists(st.text(min_size=1, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))), min_size=2, max_size=10, unique=True))
@settings(max_examples=30)
def test_duplicate_register_always_raises(names):
    reg = _make_registry()
    first = names[0]

    @reg.register(first)
    async def _base():
        return "x"

    for duplicate in names[1:]:
        reg2 = _make_registry()

        @reg2.register(duplicate)
        async def _orig():
            return "y"

        with pytest.raises(ValueError):

            @reg2.register(duplicate)
            async def _dupe():
                return "z"


@given(st.lists(st.text(min_size=1, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"))), min_size=1, max_size=15, unique=True))
@settings(max_examples=30)
def test_unregistered_name_never_retrievable(names):
    reg = _make_registry()

    registered = names[: len(names) // 2]
    unregistered = names[len(names) // 2 :]

    for name in registered:
        cap = name

        @reg.register(cap)
        async def _fn(_n=cap):
            return _n

        reg.register(cap)(_fn) if False else None

    for name in unregistered:
        with pytest.raises(KeyError):
            asyncio.run(reg.get(name))
