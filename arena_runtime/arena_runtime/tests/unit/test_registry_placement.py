"""Slot placement across evictions and rect changes."""

from __future__ import annotations

import pytest

pytest.importorskip("builtin_interfaces.msg")
pytest.importorskip("arena_runtime_msgs.msg")
pytest.importorskip("arena_runtime.registry")

import arena_runtime_msgs.msg  # noqa: E402
import builtin_interfaces.msg  # noqa: E402

from arena_runtime.registry import EnvRegistry  # noqa: E402


def _extent(w: float, h: float) -> arena_runtime_msgs.msg.WorldExtent:
    e = arena_runtime_msgs.msg.WorldExtent()
    e.x_min, e.y_min, e.x_max, e.y_max = 0.0, 0.0, w, h
    return e


def _slot(reg: EnvRegistry, env_id: int, extent: arena_runtime_msgs.msg.WorldExtent) -> tuple[float, float, float, float]:
    p = reg.place(env_id, extent)
    cx = p.reference[0] + (extent.x_min + extent.x_max) / 2.0
    cy = p.reference[1] + (extent.y_min + extent.y_max) / 2.0
    w, h = p.slot_extent
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] - 1e-6 and b[0] < a[2] - 1e-6 and a[1] < b[3] - 1e-6 and b[1] < a[3] - 1e-6


def test_eviction_keeps_live_slots_and_new_slot_is_disjoint():
    reg = EnvRegistry(slot_buffer=5.0)
    now = builtin_interfaces.msg.Time()
    slots = {}
    extents = {}
    for i, (w, h) in enumerate([(20.0, 15.0), (12.0, 30.0), (25.0, 10.0), (18.0, 18.0)]):
        env_id, _ = reg.reserve(now=now)
        extents[env_id] = _extent(w, h)
        slots[env_id] = _slot(reg, env_id, extents[env_id])
        assert env_id == i

    reg.start_eviction(1)
    reg.complete_eviction(1)
    del slots[1]

    for env_id, extent in extents.items():
        if env_id in slots:
            assert _slot(reg, env_id, extent) == slots[env_id]

    new_id, _ = reg.reserve(now=now)
    new_slot = _slot(reg, new_id, _extent(22.0, 22.0))
    for env_id, slot in slots.items():
        assert not _overlap(new_slot, slot), (new_id, env_id)


def test_rect_change_moves_only_that_env():
    reg = EnvRegistry(slot_buffer=5.0)
    now = builtin_interfaces.msg.Time()
    a, _ = reg.reserve(now=now)
    b, _ = reg.reserve(now=now)
    slot_a = _slot(reg, a, _extent(20.0, 20.0))
    slot_b = _slot(reg, b, _extent(20.0, 20.0))
    new_a = _slot(reg, a, _extent(30.0, 10.0))
    assert _slot(reg, b, _extent(20.0, 20.0)) == slot_b
    assert not _overlap(new_a, slot_b)
    assert new_a != slot_a
