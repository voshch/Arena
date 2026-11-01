"""Lockstep on the gazebo host: a step returns once /clock carries it, an unpause waits for gz to finish its last batch."""

from __future__ import annotations

import statistics
import time

from conftest import DT, Rig

STEPS = 60
STATS_PERIOD_S = 0.2


async def _steps_return_without_waiting_for_a_stats_report(rig: Rig) -> None:
    from arena_runtime_msgs.srv import LifecycleHold

    probe = rig.probe
    caller = probe.get_fully_qualified_name()
    response = await probe.hold.call_timeout(LifecycleHold.Request(action=LifecycleHold.Request.ACQUIRE, caller_id=caller, reason="test"))
    assert response is not None and response.paused
    lifecycle = rig.arena._lifecycle
    durations = []
    for _ in range(STEPS):
        started = time.monotonic()
        await lifecycle.step_seconds(DT)
        durations.append(time.monotonic() - started)
    per_step = statistics.median(durations)
    assert per_step < STATS_PERIOD_S / 2, f"{per_step * 1e3:.1f} ms wall per {DT * 1e3:.0f} ms tick"

    await lifecycle.step_seconds(2.5)
    response = await probe.hold.call_timeout(LifecycleHold.Request(action=LifecycleHold.Request.RELEASE, caller_id=caller, reason="test"))
    assert response is not None and not response.paused
    released = probe.clock()
    await rig.until(lambda: probe.clock() > released + 0.5, 10.0, "gz clock running after the release")


def test_steps_return_without_waiting_for_a_stats_report(gazebo_rig) -> None:
    gazebo_rig(_steps_return_without_waiting_for_a_stats_report)
