"""Env registration, eviction and holds of ArenaNode on the dummy sim."""

from __future__ import annotations

import asyncio

import pytest

from conftest import Rig

AUTO_ENV_ID = 0xFFFF


class _Fleet:
    """Plays the task generators: registers envs, reports lifecycle states and heartbeats."""

    def __init__(self, rig: Rig) -> None:
        import rclpy.qos
        from arena_runtime_msgs.msg import EnvRegistry, HoldRegistry, ShutdownRequest
        from arena_runtime_msgs.srv import DespawnEnv, RegisterEnv

        self.rig = rig
        self.envs: list = []
        self.holds: list = []
        self.shutdowns: list = []
        latched = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
        probe = rig.probe
        probe.create_subscription(EnvRegistry, "/arena/state/envs", self._on_envs, latched)
        probe.create_subscription(HoldRegistry, "/arena/state/holders", self._on_holds, latched)
        probe.create_subscription(ShutdownRequest, "/arena/shutdown_request", self.shutdowns.append, 10)
        self._register = probe.create_client_wrapper(RegisterEnv, "/arena/register_env", timeout=5.0)
        self._despawn = probe.create_client_wrapper(DespawnEnv, "/arena/despawn_env", timeout=5.0)

    def _on_envs(self, msg) -> None:
        self.envs = list(msg.envs)

    def _on_holds(self, msg) -> None:
        self.holds = list(msg.holds)

    async def ready(self) -> None:
        assert await self._register.ensure(timeout_sec=10.0)
        assert await self._despawn.ensure(timeout_sec=10.0)
        await self.rig.until(lambda: self.rig.arena._pub_shutdown_request.get_subscription_count() >= 1, 5.0, "shutdown subscription")

    async def register(self, env_id: int = AUTO_ENV_ID, ns: str = ""):
        from arena_runtime_msgs.srv import RegisterEnv

        response = await self._register.call_timeout(RegisterEnv.Request(caller_id="fleet", env_id=env_id, ns=ns))
        assert response is not None
        return response

    async def despawn(self, env_id: int):
        from arena_runtime_msgs.srv import DespawnEnv

        response = await self._despawn.call_timeout(DespawnEnv.Request(env_id=env_id))
        assert response is not None
        return response

    async def _publisher(self, msg_type: type, topic: str):
        publisher = self.rig.probe.create_publisher(msg_type, topic, 10)
        await self.rig.until(lambda: publisher.get_subscription_count() >= 1, 5.0, f"subscriber on {topic}")
        return publisher

    async def report_state(self, namespace: str, state_id: int) -> None:
        from lifecycle_msgs.msg import TransitionEvent

        message = TransitionEvent()
        message.goal_state.id = state_id
        (await self._publisher(TransitionEvent, f"/{namespace}/transition_event")).publish(message)

    async def report_stale_heartbeat(self, namespace: str) -> None:
        from arena_runtime_msgs.msg import Heartbeat

        message = Heartbeat(fqn=f"/{namespace}")
        message.stamp.sec = 1
        (await self._publisher(Heartbeat, f"/{namespace}/state/heartbeat")).publish(message)

    async def hold(self, caller: str, reason: str, *, acquire: bool):
        from arena_runtime_msgs.srv import LifecycleHold

        action = LifecycleHold.Request.ACQUIRE if acquire else LifecycleHold.Request.RELEASE
        response = await self.rig.probe.hold.call_timeout(LifecycleHold.Request(action=action, caller_id=caller, reason=reason))
        assert response is not None
        return response


async def _registration(rig: Rig) -> None:
    fleet = _Fleet(rig)
    await fleet.ready()

    first = await fleet.register()
    assert (first.success, first.env_id, first.ns, first.sim) == (True, 0, "arena/env_0/task_generator_node", "dummy")
    await rig.until(lambda: [env.env_id for env in fleet.envs] == [0], 3.0, "registered env")
    assert fleet.envs[0].fqn == f"/{first.ns}"
    assert not fleet.envs[0].ready

    taken_id = await fleet.register(env_id=0)
    assert not taken_id.success and "already in use" in taken_id.error_msg
    taken_ns = await fleet.register(ns=first.ns)
    assert not taken_ns.success and "already owned" in taken_ns.error_msg
    second = await fleet.register()
    assert (second.success, second.env_id) == (True, 1)


async def _ready_follows_the_lifecycle(rig: Rig) -> None:
    from lifecycle_msgs.msg import State

    fleet = _Fleet(rig)
    await fleet.ready()
    env = await fleet.register()
    await fleet.report_state(env.ns, State.PRIMARY_STATE_ACTIVE)
    await rig.until(lambda: bool(fleet.envs) and fleet.envs[0].ready, 3.0, "ready env")


async def _despawn_requests_a_shutdown(rig: Rig) -> None:
    fleet = _Fleet(rig)
    await fleet.ready()
    unknown = await fleet.despawn(7)
    assert not unknown.success and "not registered" in unknown.error_msg

    env = await fleet.register()
    response = await fleet.despawn(env.env_id)
    assert response.success
    await rig.until(lambda: bool(fleet.shutdowns), 3.0, "shutdown request")
    assert (fleet.shutdowns[0].env_id, fleet.shutdowns[0].reason) == (env.env_id, "despawn_env")
    assert [record.env_id for record in fleet.envs] == [env.env_id]


async def _evicted(rig: Rig, fleet: _Fleet, env, reason: str) -> None:
    await rig.until(lambda: bool(fleet.shutdowns), 5.0, "shutdown request")
    assert (fleet.shutdowns[0].env_id, fleet.shutdowns[0].reason) == (env.env_id, reason)
    await rig.until(lambda: fleet.envs == [] and fleet.holds == [], 8.0, "evicted env")
    await rig.until(lambda: _running(rig), 3.0, "unpaused sim")
    for _ in range(50):
        reused = await fleet.register(env_id=env.env_id)
        if reused.success:
            break
        await asyncio.sleep(0.2)
    assert (reused.success, reused.env_id) == (True, env.env_id)


def _running(rig: Rig) -> bool:
    return rig.probe.paused is not None and not rig.probe.paused.data


async def _finalized_env_is_evicted_and_loses_its_holds(rig: Rig) -> None:
    from lifecycle_msgs.msg import State

    fleet = _Fleet(rig)
    await fleet.ready()
    env = await fleet.register()
    assert (await fleet.hold(f"/{env.ns}", "reset", acquire=True)).paused
    await rig.until(lambda: len(fleet.holds) == 1, 3.0, "published hold")
    await fleet.report_state(env.ns, State.PRIMARY_STATE_FINALIZED)
    await _evicted(rig, fleet, env, "lifecycle_finalized")


async def _silent_ready_env_is_swept(rig: Rig) -> None:
    from lifecycle_msgs.msg import State

    fleet = _Fleet(rig)
    await fleet.ready()
    env = await fleet.register()
    await fleet.report_state(env.ns, State.PRIMARY_STATE_ACTIVE)
    await rig.until(lambda: bool(fleet.envs) and fleet.envs[0].ready, 3.0, "ready env")
    await fleet.report_stale_heartbeat(env.ns)
    await _evicted(rig, fleet, env, "heartbeat_timeout")


async def _holds_pause_until_the_last_release(rig: Rig) -> None:
    fleet = _Fleet(rig)
    first = await fleet.hold("/a", "test", acquire=True)
    assert (first.paused, first.hold_count) == (True, 1)
    second = await fleet.hold("/b", "test", acquire=True)
    assert (second.paused, second.hold_count) == (True, 2)
    released = await fleet.hold("/a", "test", acquire=False)
    assert (released.paused, released.hold_count) == (True, 1)
    await asyncio.sleep(0.3)
    held_at = await rig.frozen(0.3)
    last = await fleet.hold("/b", "test", acquire=False)
    assert (last.paused, last.hold_count) == (False, 0)
    await rig.until(lambda: rig.probe.clock() > held_at, 3.0, "clock after the last release")


@pytest.mark.parametrize(
    "scenario",
    [
        _registration,
        _ready_follows_the_lifecycle,
        _despawn_requests_a_shutdown,
        _finalized_env_is_evicted_and_loses_its_holds,
        _silent_ready_env_is_swept,
        _holds_pause_until_the_last_release,
    ],
    ids=lambda scenario: scenario.__name__.lstrip("_"),
)
def test_arena_node_env_lifecycle(arena_rig, scenario) -> None:
    arena_rig(scenario)
