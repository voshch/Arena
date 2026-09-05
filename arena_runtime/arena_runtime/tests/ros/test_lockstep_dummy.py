"""Lockstep control plane end to end on the dummy sim."""

from __future__ import annotations

import asyncio

import pytest

from conftest import DT, Rig

PERIOD = 0.1
HEARTBEAT = "arena_runtime_msgs/msg/LockstepHeartbeat"


async def _held_step(rig: Rig) -> None:
    from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep

    probe = rig.probe

    # free-running dummy clock advances at wall rate
    t0 = probe.clock()
    await asyncio.sleep(0.3)
    assert 0.1 < probe.clock() - t0 < 1.0

    # a step without a hold is refused at the service layer
    response = await probe.step.call_timeout(LifecycleStep.Request(seconds=PERIOD))
    assert response is not None and not response.success and response.advanced == 0.0
    assert "hold" in response.error_msg

    # a held step advances the quantized delta and the clock stays frozen around it
    caller = probe.get_fully_qualified_name()
    acquire = LifecycleHold.Request(action=LifecycleHold.Request.ACQUIRE, caller_id=caller, reason="test")
    release = LifecycleHold.Request(action=LifecycleHold.Request.RELEASE, caller_id=caller, reason="test")
    response = await probe.hold.call_timeout(acquire)
    assert response is not None and response.paused and response.hold_count == 1
    await asyncio.sleep(3 * DT)
    before = await rig.frozen(0.2)
    response = await probe.step.call_timeout(LifecycleStep.Request(seconds=0.061))
    assert response is not None and response.success
    assert response.advanced == pytest.approx(3 * DT)
    await rig.until(lambda: probe.clock() >= before + 3 * DT - 1e-6, 2.0, "stepped clock on the wire")
    assert (await rig.frozen(0.2)) == pytest.approx(before + 3 * DT, abs=1e-6)
    response = await probe.hold.call_timeout(release)
    assert response is not None and not response.paused and response.hold_count == 0
    t0 = probe.clock()
    await asyncio.sleep(0.3)
    assert probe.clock() > t0


async def _gated_run(rig: Rig) -> None:
    from arena_runtime_msgs.msg import LockstepChannel

    probe = rig.probe

    # registration validates before mutating
    good = LockstepChannel(name="beat", topic=probe.beat_topic, type=HEARTBEAT, period_s=PERIOD, hard=True)
    response = await probe.register_channels([good])
    assert response.success, response.error_msg
    response = await probe.register_channels([LockstepChannel(name="bad", topic=probe.beat_topic, type=HEARTBEAT, period_s=0.0, hard=True)])
    assert not response.success and "period" in response.error_msg
    response = await probe.register_channels([LockstepChannel(name="bad", topic=probe.beat_topic, type="nope/msg/Nope", period_s=PERIOD, hard=True)])
    assert not response.success and "type" in response.error_msg
    await rig.until(lambda: probe.registered_callers() == ["probe"], 2.0, "registration status")
    assert [ch.name for ch in probe.status.registrations[0].channels] == ["beat"]

    # a gated run steps one window and freezes on the silent hard channel
    await probe.start_run()
    await rig.until(lambda: probe.stalled_on("beat"), 5.0, "first gate stall")
    tick = probe.status_tick()
    clock = await rig.frozen(0.3)
    assert clock == pytest.approx(tick, abs=1e-3)
    await rig.until(lambda: probe.paused is not None and probe.paused.data, 2.0, "paused state")

    # a registration arriving mid-stall republishes status with the stall still visible
    soft = LockstepChannel(name="peds", topic="/probe/peds", type=HEARTBEAT, period_s=PERIOD, hard=False)
    response = await probe.register_channels([soft], caller="probe2")
    assert response.success
    await rig.until(lambda: sorted(probe.registered_callers()) == ["probe", "probe2"], 2.0, "second registration")
    assert probe.stalled_on("beat")
    response = await probe.register_channels([], caller="probe2")
    assert response.success
    await rig.until(lambda: probe.registered_callers() == ["probe"], 2.0, "second deregistration")
    assert probe.stalled_on("beat")

    # the previous window's boundary sample does not cover this window
    probe.pulse(tick - PERIOD)
    await asyncio.sleep(0.3)
    assert probe.stalled_on("beat")
    assert probe.clock() == pytest.approx(clock, abs=1e-9)

    # a covering sample releases exactly one window
    probe.pulse(tick)
    await rig.until(lambda: probe.stalled_on("beat") and probe.status_tick() > tick + PERIOD / 2, 5.0, "second gate stall")
    assert probe.status_tick() == pytest.approx(tick + PERIOD, abs=1e-6)
    assert (await rig.frozen(0.3)) == pytest.approx(tick + PERIOD, abs=1e-3)

    # deregistering the stalled channel unblocks the tick, with no gates the run free-steps
    response = await probe.register_channels([])
    assert response.success
    await rig.until(lambda: probe.clock() > tick + 2.0, 3.0, "free-stepping clock")
    await rig.until(lambda: probe.status is not None and probe.status.measured_rtf > 1.0, 3.0, "measured rtf")

    # stop returns promptly, frees the hold, and the clock runs at wall rate again
    t0 = asyncio.get_running_loop().time()
    await probe.stop_run()
    assert asyncio.get_running_loop().time() - t0 < 1.0
    await rig.until(lambda: probe.status is not None and not probe.status.active, 2.0, "inactive status")
    await rig.until(lambda: probe.paused is not None and not probe.paused.data, 2.0, "unpaused state")
    await asyncio.sleep(0.6)
    t0 = probe.clock()
    await asyncio.sleep(0.3)
    assert 0.1 < probe.clock() - t0 < 1.0


async def _paced_run(rig: Rig) -> None:
    """target_rtf bounds the free-stepping rate."""
    probe = rig.probe
    await probe.start_run(target_rtf=2.0)
    await rig.until(lambda: probe.status is not None and probe.status.active, 2.0, "active status")
    await asyncio.sleep(0.5)
    t0 = probe.clock()
    await asyncio.sleep(1.0)
    assert 1.5 < probe.clock() - t0 < 2.6
    await rig.until(lambda: probe.status is not None and 1.5 < probe.status.measured_rtf < 2.6, 3.0, "measured rtf near target")
    await probe.stop_run()


def test_held_step(arena_rig) -> None:
    arena_rig(_held_step)


def test_gated_run(arena_rig) -> None:
    arena_rig(_gated_run)


def test_paced_run(arena_rig) -> None:
    arena_rig(_paced_run)
