"""Lockstep on the flatsim host: the server takes the steps and a step returns once /clock carries it."""

from __future__ import annotations

import asyncio

import pytest

from conftest import DT, Rig


async def _steps_are_exact_and_the_hold_freezes_the_server(rig: Rig) -> None:
    from arena_runtime_msgs.srv import LifecycleHold, LifecycleStep

    from arena_runtime.sim.flatsim_simulator.flatsim_simulator import FlatsimHost

    probe = rig.probe
    host = rig.arena._lifecycle._inner
    assert isinstance(host, FlatsimHost)
    running = probe.clock()
    await rig.until(lambda: probe.clock() > running + 5 * DT, 30.0, "free-running flatsim clock")

    caller = probe.get_fully_qualified_name()
    response = await probe.hold.call_timeout(LifecycleHold.Request(action=LifecycleHold.Request.ACQUIRE, caller_id=caller, reason="test"))
    assert response is not None and response.paused
    frozen = await rig.frozen(0.2)

    response = await probe.step.call_timeout(LifecycleStep.Request(seconds=3 * DT))
    assert response is not None and response.success
    assert response.advanced == pytest.approx(3 * DT)
    assert rig.arena.sim_time.to_seconds() == pytest.approx(frozen + 3 * DT, abs=1e-6)
    assert await rig.frozen(0.2) == pytest.approx(frozen + 3 * DT, abs=1e-6)

    lifecycle = rig.arena._lifecycle
    before = rig.arena.sim_time.to_seconds()
    for i in range(1, 51):
        assert await lifecycle.step_seconds(DT) == pytest.approx(DT)
        assert rig.arena.sim_time.to_seconds() == pytest.approx(before + i * DT, abs=1e-6)

    long = round(2.5 / DT)
    assert await lifecycle.step_seconds(2.5) == pytest.approx(long * DT)
    assert rig.arena.sim_time.to_seconds() == pytest.approx(before + (50 + long) * DT, abs=1e-6)

    response = await probe.hold.call_timeout(LifecycleHold.Request(action=LifecycleHold.Request.RELEASE, caller_id=caller, reason="test"))
    assert response is not None and not response.paused
    released = probe.clock()
    await rig.until(lambda: probe.clock() > released + 5 * DT, 30.0, "flatsim clock after the release")


async def _run_survives_a_server_outage_longer_than_a_step_call(rig: Rig) -> None:
    probe = rig.probe
    server = rig.servers[0]
    await probe.start_run(ungated=True)
    running = probe.clock()
    await rig.until(lambda: probe.clock() > running + 10 * DT, 30.0, "lockstep steps the server")
    await rig.until(lambda: probe.status is not None and probe.status.active, 30.0, "active run status")

    rig.executor.remove_node(server)
    await asyncio.sleep(rig.arena._lifecycle._inner._control._timeout + 2.0)
    assert probe.status is not None and probe.status.active
    assert probe.paused is not None and probe.paused.data
    rig.executor.add_node(server)

    resumed = probe.clock()
    await rig.until(lambda: probe.clock() > resumed + 10 * DT, 30.0, "lockstep steps again after the outage")
    assert probe.status.active and probe.paused.data
    await probe.stop_run()


def test_steps_are_exact_and_the_hold_freezes_the_server(flatsim_rig) -> None:
    flatsim_rig(_steps_are_exact_and_the_hold_freezes_the_server)


def test_run_survives_a_server_outage_longer_than_a_step_call(flatsim_rig) -> None:
    flatsim_rig(_run_survives_a_server_outage_longer_than_a_step_call, timeout=150.0)
