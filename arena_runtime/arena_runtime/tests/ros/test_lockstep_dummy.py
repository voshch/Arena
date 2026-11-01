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
    await rig.until(lambda: probe.clock() > t0 + 0.1, 2.0, "free-running clock")

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
    await rig.until(lambda: probe.clock() > t0, 2.0, "clock after the release")


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
    assert (await rig.frozen(0.3)) == pytest.approx(clock, abs=1e-9)
    assert probe.stalled_on("beat")

    # a covering sample releases exactly one window
    probe.pulse(tick)
    await rig.until(lambda: probe.stalled_on("beat") and probe.status_tick() > tick + PERIOD / 2, 5.0, "second gate stall")
    assert probe.status_tick() == pytest.approx(tick + PERIOD, abs=1e-6)
    assert (await rig.frozen(0.3)) == pytest.approx(tick + PERIOD, abs=1e-3)

    # deregistering the stalled channel unblocks the tick, with no gates the unpaced run steps at wall rate
    response = await probe.register_channels([])
    assert response.success
    await rig.until(lambda: probe.clock() > tick + 0.5, 3.0, "unblocked clock")
    await rig.until(lambda: probe.status is not None and 0.6 < probe.status.measured_rtf < 1.4, 3.0, "measured rtf near wall rate")

    # stop returns promptly, frees the hold, and the clock runs at wall rate again
    await probe.stop_run()
    await rig.until(lambda: probe.status is not None and not probe.status.active, 2.0, "inactive status")
    await rig.until(lambda: probe.paused is not None and not probe.paused.data, 2.0, "unpaused state")
    t0 = probe.clock()
    await rig.until(lambda: probe.clock() > t0 + 0.1, 2.0, "free-running clock after the stop")


async def _paced_run(rig: Rig) -> None:
    """target_rtf bounds the free-stepping rate."""
    probe = rig.probe
    await probe.start_run(target_rtf=2.0)
    await rig.until(lambda: probe.status is not None and probe.status.active, 2.0, "active status")
    await rig.until(lambda: probe.status is not None and 1.5 < probe.status.measured_rtf < 2.6, 3.0, "measured rtf near target")
    loop = asyncio.get_running_loop()
    sim0, wall0 = probe.clock(), loop.time()
    await asyncio.sleep(1.0)
    assert (probe.clock() - sim0) / (loop.time() - wall0) < 2.6
    await probe.stop_run()


async def _ungated_run(rig: Rig) -> None:
    """An unpaced ungated run free-steps past wall rate."""
    probe = rig.probe
    await probe.start_run(ungated=True)
    await rig.until(lambda: probe.status is not None and probe.status.active, 2.0, "active status")
    await rig.until(lambda: probe.status is not None and probe.status.measured_rtf > 2.0, 3.0, "measured rtf")
    await probe.stop_run()


async def _channel_dropped_after_its_last_beat(rig: Rig) -> None:
    """A hard channel deregistered right after its last beat never stalls the run."""
    from arena_runtime_msgs.msg import LockstepChannel

    probe = rig.probe
    beat = LockstepChannel(name="beat", topic=probe.beat_topic, type=HEARTBEAT, period_s=PERIOD, hard=True)
    await probe.start_run()
    for _ in range(20):
        response = await probe.register_channels([beat])
        assert response.success, response.error_msg
        await rig.until(lambda: probe.stalled_on("beat"), 5.0, "stall on beat")
        tick = probe.status_tick()
        probe.pulse(tick)
        response = await probe.register_channels([])
        assert response.success, response.error_msg
        await rig.until(lambda tick=tick: probe.clock() > tick + 1.0, 3.0, "clock past the dropped channel")
    await probe.stop_run()


async def _window_under_a_run(rig: Rig) -> None:
    """An unpause window opened under a gated run gets gated steps, never a free-running clock."""
    from arena_runtime_msgs.msg import LockstepChannel
    from arena_runtime_msgs.srv import LifecycleHold

    probe = rig.probe
    host = rig.arena._lifecycle._inner
    holder = "/arena/env_0/task_generator_node"
    beat = LockstepChannel(name="beat", topic=probe.beat_topic, type=HEARTBEAT, period_s=PERIOD, hard=True)
    own = LockstepChannel(name="own", topic="/probe/own", type=HEARTBEAT, period_s=PERIOD, hard=True)
    assert (await probe.register_channels([beat])).success
    assert (await probe.register_channels([own], caller="own", env=holder)).success
    await probe.start_run()
    await rig.until(lambda: probe.status is not None and probe.status.active and sorted(probe.status.waiting_on) == ["beat", "own"], 5.0, "stall on both channels")

    tick = probe.status_tick()
    await probe.set_window(holder, True)
    assert host.paused
    assert (await rig.frozen(0.5)) == pytest.approx(tick, abs=1e-3)
    probe.pulse(tick)
    await rig.until(lambda: probe.stalled_on("beat") and probe.status_tick() > tick + PERIOD / 2, 5.0, "gated tick inside the window")
    assert host.paused

    caller = probe.get_fully_qualified_name()
    response = await probe.hold.call_timeout(LifecycleHold.Request(action=LifecycleHold.Request.ACQUIRE, caller_id=caller, reason="reset"))
    assert response is not None and response.paused
    rig.arena._lockstep.pause()
    tick = probe.status_tick()
    probe.pulse(tick)
    await rig.until(lambda: probe.stalled_on("beat") and probe.status_tick() > tick + PERIOD / 2, 5.0, "gated tick inside the window under a foreign hold and a paused run")

    await probe.set_window(holder, False)
    assert host.paused
    tick = probe.status_tick()
    probe.pulse(tick)
    clock = await rig.frozen(0.5)
    assert clock == pytest.approx(tick, abs=1e-3)
    response = await probe.hold.call_timeout(LifecycleHold.Request(action=LifecycleHold.Request.RELEASE, caller_id=caller, reason="reset"))
    assert response is not None and response.paused
    rig.arena._lockstep.resume()
    await rig.until(lambda: probe.status_tick() > tick + PERIOD / 2 and sorted(probe.status.waiting_on) == ["beat", "own"], 5.0, "holder's channel back in the gate")

    await probe.set_window(holder, True)
    await probe.stop_run()
    await rig.until(lambda: not host.paused, 3.0, "window unpauses once the run stops")
    t0 = probe.clock()
    await rig.until(lambda: probe.clock() > t0, 2.0, "clock after the window")
    await probe.set_window(holder, False)


def test_held_step(arena_rig) -> None:
    arena_rig(_held_step)


def test_gated_run(arena_rig) -> None:
    arena_rig(_gated_run)


def test_paced_run(arena_rig) -> None:
    arena_rig(_paced_run)


def test_ungated_run(arena_rig) -> None:
    arena_rig(_ungated_run)


def test_channel_dropped_after_its_last_beat(arena_rig) -> None:
    arena_rig(_channel_dropped_after_its_last_beat)


def test_window_under_a_run(arena_rig) -> None:
    arena_rig(_window_under_a_run)
