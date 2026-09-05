"""arena_robots.LockstepBeat against the real scheduler: refcounting, grace keepalive, silence holds."""

from __future__ import annotations

import asyncio

import pytest

from conftest import Rig

PERIOD = 0.1
ROBOT_NS = "/arena/env_0/jackal"


def _robot_node(rig: Rig):
    import rclpy
    from arena_rclpy_mixins.Async import AsyncNode

    node = AsyncNode("robot", namespace=ROBOT_NS, parameter_overrides=[rclpy.Parameter("use_sim_time", value=True)])
    rig.add_node(node)
    return node


async def _refcount_and_grace(rig: Rig) -> None:
    from arena_rclpy_mixins.Time import Time
    from arena_robots.lockstep_beat import LockstepBeat

    probe = rig.probe
    node = _robot_node(rig)
    beat = LockstepBeat(node, "nav", sleep=asyncio.sleep)
    await probe.start_run()

    # first acquire registers under the beat topic, owned by the env namespace
    await beat.acquire(PERIOD)
    await rig.until(lambda: probe.registered_callers() == [f"{ROBOT_NS}/lockstep/nav"], 3.0, "beat registration")
    reg = probe.status.registrations[0]
    assert reg.env == "/arena/env_0"
    assert [(ch.name, ch.period_s, ch.hard) for ch in reg.channels] == [("nav/jackal", pytest.approx(PERIOD), True)]

    # silent producer stalls the tick
    await rig.until(lambda: probe.stalled_on("nav/jackal"), 5.0, "stall on nav beat")
    tick = probe.status_tick()

    # pulses release one window each
    beat.pulse(stamp=Time.from_float(tick).to_msg())
    await rig.until(lambda: probe.stalled_on("nav/jackal") and probe.status_tick() > tick + PERIOD / 2, 5.0, "next window")
    assert probe.status_tick() == pytest.approx(tick + PERIOD, abs=1e-6)

    # after grace_s of silence the watchdog covers the sim forward and the run keeps moving
    tick = probe.status_tick()
    await rig.until(lambda: probe.clock() > tick + 1.0, 8.0, "keepalive-driven progress")

    # nested acquisition is refcounted: one registration, released only by the last release
    await beat.acquire(PERIOD)
    await asyncio.sleep(0.3)
    assert probe.registered_callers() == [f"{ROBOT_NS}/lockstep/nav"]
    await beat.release()
    await asyncio.sleep(0.3)
    assert probe.registered_callers() == [f"{ROBOT_NS}/lockstep/nav"]
    await beat.release()
    await rig.until(lambda: probe.registered_callers() == [], 3.0, "beat deregistration")
    await beat.release()
    assert probe.registered_callers() == []

    await probe.stop_run()


async def _silence_holds_without_grace(rig: Rig) -> None:
    from arena_robots.lockstep_beat import LockstepBeat

    probe = rig.probe
    node = _robot_node(rig)
    beat = LockstepBeat(node, "planner", grace_s=None, sleep=asyncio.sleep)
    await probe.start_run()
    await beat.acquire(PERIOD)
    await rig.until(lambda: probe.stalled_on("planner/jackal"), 5.0, "stall on planner beat")
    tick = probe.status_tick()
    clock = await rig.frozen(3.0)
    assert clock == pytest.approx(tick, abs=1e-3)
    assert probe.stalled_on("planner/jackal")

    # release deregisters and the stalled tick clears
    await beat.release()
    await rig.until(lambda: probe.clock() > tick + 1.0, 3.0, "run resumes after release")
    await probe.stop_run()


async def _release_during_register(rig: Rig) -> None:
    """release racing the in-flight register call must leave nothing registered."""
    from arena_robots.lockstep_beat import LockstepBeat

    probe = rig.probe
    node = _robot_node(rig)
    beat = LockstepBeat(node, "nav", sleep=asyncio.sleep)
    await probe.start_run()
    acquire = asyncio.create_task(beat.acquire(PERIOD))
    await asyncio.sleep(0)
    await beat.release()
    await acquire
    await asyncio.sleep(0.5)
    assert probe.registered_callers() == []
    await probe.stop_run()


def test_refcount_and_grace(arena_rig) -> None:
    arena_rig(_refcount_and_grace)


def test_silence_holds_without_grace(arena_rig) -> None:
    arena_rig(_silence_holds_without_grace)


def test_release_during_register(arena_rig) -> None:
    arena_rig(_release_during_register)
