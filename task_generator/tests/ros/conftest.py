from __future__ import annotations

import types
from types import SimpleNamespace

import pytest


class _FakeConf:
    class Robot:
        class BEHAVIOR:
            value = "rosnav"

        class CONTROLLER:
            value = "dwa"

        class PLANNER:
            value = "navfn"

        class AGENT:
            value = "rosnav"

        class TIMEOUT:
            value = 60

        class GOAL_TOLERANCE_RADIUS:
            value = 0.3

        class GOAL_TOLERANCE_ANGLE:
            value = 0.2

        class SPAWN_ROBOT_SAFE_DIST:
            value = 0.5


class _FakeLogger:
    def debug(self, *a, **kw): ...
    def info(self, *a, **kw): ...
    def warn(self, *a, **kw): ...
    def warning(self, *a, **kw): ...
    def error(self, *a, **kw): ...
    def fatal(self, *a, **kw): ...
    def get_child(self, name: str) -> "_FakeLogger":
        return self


@pytest.fixture()
def fake_conf():
    return _FakeConf()


@pytest.fixture()
def fake_logger():
    return _FakeLogger()


@pytest.fixture()
def stub_node(fake_conf, fake_logger):
    node = SimpleNamespace(
        conf=fake_conf,
        sim_time=SimpleNamespace(sec=0),
        get_logger=lambda: fake_logger,
    )
    return node
