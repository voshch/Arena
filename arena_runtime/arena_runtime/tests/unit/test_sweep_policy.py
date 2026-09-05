"""Pure-logic tests for sweep_verdict and reserve() heartbeat stamping.

Skipped if arena_runtime / builtin_interfaces aren't importable (no sourced
overlay).
"""
from __future__ import annotations

import pytest

pytest.importorskip("builtin_interfaces.msg")
pytest.importorskip("arena_runtime.registry")

import builtin_interfaces.msg  # noqa: E402

from arena_runtime.registry import EnvRecord, EnvRegistry, sweep_verdict  # noqa: E402


def _record(*, ready: bool = True, draining: bool = False) -> EnvRecord:
    return EnvRecord(env_id=0, fqn="/test", ready=ready, draining=draining)


def _time(sec: int = 0) -> builtin_interfaces.msg.Time:
    t = builtin_interfaces.msg.Time()
    t.sec = sec
    return t


def _verdict(
    record: EnvRecord,
    elapsed: float,
    *,
    age: float | None = None,
    has_reset_hold: bool = False,
    process_alive: bool | None = None,
) -> str | None:
    return sweep_verdict(
        record,
        elapsed=elapsed,
        age=age if age is not None else elapsed,
        has_reset_hold=has_reset_hold,
        process_alive=process_alive,
        heartbeat_timeout=5.0,
        reset_timeout=30.0,
        bootstrap_timeout=600.0,
    )


def test_draining_returns_none():
    assert _verdict(_record(draining=True), 9999.0) is None


def test_process_exited_trumps_ready():
    assert _verdict(_record(ready=True), 0.1, process_alive=False) == "process_exited"


def test_pre_ready_under_bootstrap_timeout_returns_none():
    assert _verdict(_record(ready=False), 599.9) is None


def test_pre_ready_over_bootstrap_timeout_evicts():
    assert _verdict(_record(ready=False), 600.1) == "bootstrap_timeout"


def test_ready_no_hold_under_timeout_returns_none():
    assert _verdict(_record(ready=True), 4.9) is None


def test_ready_no_hold_over_timeout_evicts():
    assert _verdict(_record(ready=True), 5.1) == "heartbeat_timeout"


def test_ready_reset_hold_mid_window_returns_none():
    assert _verdict(_record(ready=True), 20.0, has_reset_hold=True) is None


def test_ready_reset_hold_over_reset_timeout_evicts():
    assert _verdict(_record(ready=True), 30.1, has_reset_hold=True) == "heartbeat_timeout"


def test_process_alive_none_falls_through_to_phase():
    assert _verdict(_record(ready=True), 5.1, process_alive=None) == "heartbeat_timeout"
    assert _verdict(_record(ready=True), 4.9, process_alive=None) is None


def test_process_alive_true_pre_ready_still_uses_bootstrap_budget():
    assert _verdict(_record(ready=False), 599.9, process_alive=True) is None
    assert _verdict(_record(ready=False), 600.1, process_alive=True) == "bootstrap_timeout"


def test_pre_ready_fresh_heartbeat_over_age_budget_evicts():
    assert _verdict(_record(ready=False), 0.1, age=600.1) == "bootstrap_timeout"


def test_pre_ready_stale_heartbeat_under_age_budget_returns_none():
    assert _verdict(_record(ready=False), 9999.0, age=599.9) is None


def test_reserve_stamps_last_heartbeat():
    reg = EnvRegistry()
    env_id, _ns = reg.reserve(now=_time(sec=42))
    record = reg.get(env_id)
    assert record is not None
    assert record.last_heartbeat.sec == 42


def test_reserve_stamps_reserved_at():
    reg = EnvRegistry()
    env_id, _ns = reg.reserve(now=_time(sec=42))
    record = reg.get(env_id)
    assert record is not None
    assert record.reserved_at.sec == 42


def test_reserve_rejects_owned_namespace_until_freed():
    reg = EnvRegistry()
    env_id, ns = reg.reserve(requested_ns="arena/env_7/task_generator_node", now=_time(sec=1))
    with pytest.raises(ValueError, match="already owned"):
        reg.reserve(requested_ns=ns, now=_time(sec=2))
    assert reg.get(env_id + 1) is None
    reg.free(env_id)
    reused, _ns = reg.reserve(requested_ns=ns, now=_time(sec=3))
    assert reused != env_id


def test_reserve_ids_are_monotonic_after_free():
    reg = EnvRegistry()
    first, _ns = reg.reserve(now=_time(sec=1))
    reg.free(first)
    second, _ns = reg.reserve(now=_time(sec=2))
    assert second == first + 1


def test_pre_ready_bootstrap_timeout_zero_never_evicts():
    assert (
        sweep_verdict(
            _record(ready=False),
            elapsed=9999.0,
            age=9999.0,
            has_reset_hold=False,
            process_alive=True,
            heartbeat_timeout=5.0,
            reset_timeout=30.0,
            bootstrap_timeout=0.0,
        )
        is None
    )
