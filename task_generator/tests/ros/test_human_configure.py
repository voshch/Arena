from __future__ import annotations

import asyncio
import logging

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _param(name: str):
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue

    return Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=0.1))


def _configure(namespace: str | None, accepts: set[str], names: list[str]) -> tuple[list[str], list[str]]:
    from task_generator.simulators.human import BaseHumanSimulator

    seen: list[str] = []

    class Backend:
        PARAM_NAMESPACE = namespace
        _logger = logging.getLogger("test_human_configure")

        async def _configure_impl(self, params):
            seen.extend(p.name for p in params)
            return {p.name for p in params if p.name in accepts}

    accepted = asyncio.run(BaseHumanSimulator.configure(Backend(), [_param(n) for n in names]))
    return seen, [p.name for p in accepted]


def test_own_namespace_is_stripped_and_accepted_names_keep_it():
    seen, accepted = _configure("humansim", {"a.b"}, ["humansim.a.b", "humansim.c"])
    assert seen == ["a.b", "c"]
    assert accepted == ["humansim.a.b"]


def test_own_params_arrive_sorted_by_name():
    seen, _ = _configure("humansim", set(), ["humansim.local_planner.x", "humansim.global_planner", "humansim.local_planner"])
    assert seen == ["global_planner", "local_planner", "local_planner.x"]


def test_other_namespace_is_skipped():
    seen, accepted = _configure("hunav", {"a.b"}, ["humansim.a.b"])
    assert seen == [] and accepted == []


def test_backend_without_namespace_takes_nothing():
    seen, accepted = _configure(None, {"a.b"}, ["humansim.a.b"])
    assert seen == [] and accepted == []
