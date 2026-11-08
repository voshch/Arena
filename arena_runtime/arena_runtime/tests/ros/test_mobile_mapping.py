"""Mapping modes of the mobile adapters: slam where the bringup maps from the lidar, a refusal everywhere else."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")

from arena_runtime.env.adapters import ADAPTERS

ROBOT = SimpleNamespace(namespace="/env_0/jackal")
SPAWN_KWARGS = {"mapping": "slam", "planner": "none"}


def _mobile(kind: str) -> type:
    try:
        return ADAPTERS["mobile"].get(kind)
    except ImportError as exc:
        pytest.skip(f"the {kind} adapter does not import here: {exc}")


@pytest.mark.parametrize("kind", sorted(kind for kind in ADAPTERS["mobile"].keys() if kind != "nav2"))
def test_an_adapter_that_does_not_map_refuses_slam(kind: str) -> None:
    with pytest.raises(ValueError, match=r"robot\.mobile\.mapping:=slam"):
        _mobile(kind)(ROBOT, **SPAWN_KWARGS)


def test_the_nav2_adapter_maps_with_slam() -> None:
    assert _mobile("nav2").mappings == frozenset({"static", "slam"})
