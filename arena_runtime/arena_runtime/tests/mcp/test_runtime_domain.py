"""The runtime MCP domain against an in-process ArenaRuntimeNode on the dummy sim and a peer env speaking the env protocol."""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import time

import pytest

pytest.importorskip("arena_mcp")
pytest.importorskip("rclpy")

from arena_mcp.core import JOBS, ROS, ToolError, cli

from arena_runtime import mcp

DT = 0.02
BEAT = "arena_runtime_msgs/msg/LockstepHeartbeat"


class Rig:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.cold: dict = {}
        self.cold_s = 0.0
        self.arena = None
        self.env = None
        self.other = None
        self.executor = None
        self.spin = None
        self.cold_envs = ""

    async def call(self, client: object, request: object) -> object:
        assert await asyncio.to_thread(client.wait_for_service, 10.0)
        return await asyncio.wrap_future(_as_future(client.call_async(request)))

    def run(self, coro: object, timeout: float = 60.0) -> object:
        return self.loop.run_until_complete(asyncio.wait_for(coro, timeout))

    async def until(self, pred: object, timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        while not await pred():
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out after {timeout}s waiting for {what}")
            await asyncio.sleep(0.05)


async def _start(rig: Rig) -> None:
    import rclpy.node
    from arena_rclpy_mixins.spin import create_executor
    from arena_runtime_msgs.msg import Heartbeat, ShutdownRequest
    from arena_runtime_msgs.srv import ConfirmWorld, LifecycleHold, RegisterEnv
    from lifecycle_msgs.msg import State, TransitionEvent

    from arena_runtime.arena_runtime_node import ArenaRuntimeNode

    await ROS.node()
    started = time.monotonic()
    rig.cold = await mcp.runtime_state()
    with pytest.raises(ToolError) as cold_envs:
        await mcp.runtime_envs()
    rig.cold_envs = str(cold_envs.value)
    rig.cold_s = time.monotonic() - started

    class EnvPeer(rclpy.node.Node):
        """Registers like an unmanaged env, goes ACTIVE, and finalizes on despawn_env."""

        def __init__(self) -> None:
            super().__init__("mcp_env_peer")
            self.register = self.create_client(RegisterEnv, "/arena/register_env")
            self.confirm = self.create_client(ConfirmWorld, "/arena/confirm_world")
            self.env_id: int | None = None
            self.ns = ""
            self.create_subscription(ShutdownRequest, "/arena/shutdown_request", self._on_shutdown, 10)
            self.transition = None
            self.heartbeat = None

        def attach(self, env_id: int, ns: str) -> None:
            self.env_id, self.ns = env_id, ns
            self.transition = self.create_publisher(TransitionEvent, f"/{ns}/transition_event", 10)
            self.heartbeat = self.create_publisher(Heartbeat, f"/{ns}/state/heartbeat", 10)

        def publish_state(self, state_id: int) -> None:
            msg = TransitionEvent()
            msg.goal_state.id = state_id
            self.transition.publish(msg)

        def beat(self) -> None:
            self.heartbeat.publish(Heartbeat(fqn=f"/{self.ns}", stamp=self.get_clock().now().to_msg()))

        def _on_shutdown(self, msg: ShutdownRequest) -> None:
            if msg.env_id == self.env_id and msg.reason == "despawn_env":
                self.publish_state(State.PRIMARY_STATE_FINALIZED)

    class Other(rclpy.node.Node):
        def __init__(self) -> None:
            super().__init__("mcp_other_holder")
            self.hold = self.create_client(LifecycleHold, "/arena/sim_lifecycle/hold")

    arena = ArenaRuntimeNode()
    arena.declare_parameter("physics_dt", DT)
    arena.declare_parameter("sim", "dummy")
    arena.declare_parameter("heartbeat_timeout_sec", 600.0)
    rig.arena = arena
    rig.executor = create_executor()
    rig.executor.add_node(arena)
    rig.spin = rig.loop.run_in_executor(None, rig.executor.spin)
    try:
        await arena.setup()
    except RuntimeError as exc:
        if "already running" in str(exc):
            pytest.skip(str(exc))
        raise
    rig.env, rig.other = EnvPeer(), Other()
    rig.executor.add_node(rig.env)
    rig.executor.add_node(rig.other)


def _as_future(ros_future: object) -> object:
    import concurrent.futures

    out: concurrent.futures.Future = concurrent.futures.Future()
    ros_future.add_done_callback(lambda done: out.set_result(done.result()))
    return out


async def _stop(rig: Rig) -> None:
    if rig.arena is not None and rig.arena._clock_task is not None:
        rig.arena._clock_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await rig.arena._clock_task
    if ROS.started:
        await ROS.shutdown()
    if rig.executor is not None:
        with contextlib.suppress(Exception):
            rig.executor.shutdown()
        with contextlib.suppress(Exception):
            await rig.spin
    for node in (rig.arena, rig.env, rig.other):
        if node is not None:
            with contextlib.suppress(Exception):
                node.destroy_node()


@pytest.fixture(scope="module")
def rig() -> Rig:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    rig = Rig(loop)
    try:
        rig.run(_start(rig), timeout=60.0)
        yield rig
    finally:
        rig.run(_stop(rig), timeout=30.0)
        loop.close()


def test_state_and_envs_answer_fast_without_a_runtime(rig: Rig) -> None:
    assert rig.cold["running"] is False and "system_launch" in rig.cold["hint"]
    assert "/arena/state/envs" in rig.cold_envs and "system_launch" in rig.cold_envs
    assert rig.cold_s < 8.0


def test_state_reports_live_healthy_runtime(rig: Rig) -> None:
    state = rig.run(mcp.runtime_state())
    assert state["running"] and state["healthy"]
    assert state["sim"] == {"alive": True, "reason": "", "since_wall": pytest.approx(time.time(), abs=120)}
    assert state["paused"] is False
    assert state["holders"] == []
    assert state["envs"] == {"count": 0, "ready": 0, "env_ids": []}
    assert state["params"]["sim"] == "dummy"
    assert state["params"]["physics_dt"] == DT
    assert state["lockstep"]["active"] is False
    assert state["own"]["caller_id"].startswith("/arena_mcp_")


async def _register_env(rig: Rig) -> int:
    from arena_runtime_msgs.msg import WorldExtent
    from arena_runtime_msgs.srv import ConfirmWorld, RegisterEnv
    from lifecycle_msgs.msg import State

    response = await rig.call(rig.env.register, RegisterEnv.Request(caller_id="/mcp_env_peer", env_id=0xFFFF, ns=""))
    assert response.success, response.error_msg
    rig.env.attach(response.env_id, response.ns)
    placed = await rig.call(rig.env.confirm, ConfirmWorld.Request(env_id=response.env_id, extent=WorldExtent(x_min=0.0, y_min=0.0, x_max=4.0, y_max=2.0)))
    assert placed.success

    async def ready() -> bool:
        rig.env.publish_state(State.PRIMARY_STATE_ACTIVE)
        rig.env.beat()
        rows = await mcp.runtime_envs()
        return bool(rows) and rows[0]["ready"]

    await rig.until(ready, 10.0, "env ready in the table")
    return response.env_id


def test_envs_lists_registered_env_with_slot_and_heartbeat(rig: Rig) -> None:
    env_id = rig.run(_register_env(rig))
    rows = rig.run(mcp.runtime_envs())
    assert len(rows) == 1
    row = rows[0]
    assert row["env_id"] == env_id
    assert row["fqn"] == f"/arena/env_{env_id}/arena_env_node"
    assert row["ns"] == f"/arena/env_{env_id}"
    assert row["ready"] and row["placed"]
    assert row["slot"]["size"] == pytest.approx([14.0, 12.0])
    assert row["slot"]["extent"] == {"x_min": 0.0, "y_min": 0.0, "x_max": 4.0, "y_max": 2.0}
    assert 0.0 <= row["heartbeat_age_s"] < 10.0
    state = rig.run(mcp.runtime_state())
    assert state["envs"] == {"count": 1, "ready": 1, "env_ids": [env_id]}


async def _other_hold(rig: Rig, action: int, caller_id: str = "/mcp_other_holder", reason: str = "foreign") -> None:
    from arena_runtime_msgs.srv import LifecycleHold

    await rig.call(rig.other.hold, LifecycleHold.Request(action=action, caller_id=caller_id, reason=reason))


def test_holds_are_counted_and_release_spares_other_callers(rig: Rig) -> None:
    from arena_runtime_msgs.srv import LifecycleHold

    first = rig.run(mcp.runtime_hold("inspect"))
    caller = first["caller_id"]
    assert first["paused"] and first["hold_count"] == 1
    second = rig.run(mcp.runtime_hold("inspect"))
    assert second["own"] == {"inspect": 2}
    assert {"caller_id": caller, "reason": "inspect", "count": 2} in second["holders"]
    rig.run(_other_hold(rig, LifecycleHold.Request.ACQUIRE))
    with pytest.raises(ToolError, match="holds no 'foreign'"):
        rig.run(mcp.runtime_release("foreign"))
    one = rig.run(mcp.runtime_release("inspect"))
    assert one["released"] == ["inspect"] and one["own"] == {"inspect": 1}
    rest = rig.run(mcp.runtime_release())
    assert rest["released"] == ["inspect"] and rest["own"] == {}
    assert rest["paused"] is True
    assert rest["holders"] == [{"caller_id": "/mcp_other_holder", "reason": "foreign", "count": 1}]
    assert rig.arena._holds.has("/mcp_other_holder", "foreign")
    rig.run(_other_hold(rig, LifecycleHold.Request.RELEASE))

    async def unpaused() -> bool:
        return (await mcp.runtime_state())["paused"] is False

    rig.run(rig.until(unpaused, 5.0, "unpaused after the last hold"))
    assert rig.run(mcp.runtime_release())["released"] == []


def test_own_holds_are_the_runtime_holders_under_this_caller(rig: Rig) -> None:
    from arena_runtime_msgs.srv import LifecycleHold

    caller = rig.run(mcp._caller())
    rig.run(_other_hold(rig, LifecycleHold.Request.ACQUIRE, caller_id=caller, reason="earlier"))

    async def listed() -> bool:
        return (await mcp.runtime_state())["own"]["holds"] == {"earlier": 1}

    rig.run(rig.until(listed, 5.0, "a hold taken under this caller_id listed as own"))
    released = rig.run(mcp.runtime_release())
    assert released["released"] == ["earlier"] and released["own"] == {}
    assert not rig.arena._holds.has(caller, "earlier")


def test_step_advances_a_held_sim_and_refuses_without_hold(rig: Rig) -> None:
    with pytest.raises(ToolError, match="hold"):
        rig.run(mcp.runtime_step(0.1))
    rig.run(mcp.runtime_hold("step"))
    try:
        rig.run(asyncio.sleep(0.2))
        stepped = rig.run(mcp.runtime_step(0.1))
        assert stepped["advanced"] == pytest.approx(5 * DT)
        assert stepped["sim_time_after"] - stepped["sim_time_before"] == pytest.approx(stepped["advanced"], abs=1e-3)
    finally:
        rig.run(mcp.runtime_release("step"))


def test_unpause_window_runs_the_held_sim_then_repauses(rig: Rig) -> None:
    rig.run(mcp.runtime_hold("window"))
    try:
        rig.run(asyncio.sleep(0.2))
        window = rig.run(mcp.runtime_unpause_window(0.5))
        assert window["sim_time_after"] - window["sim_time_before"] > 0.2
        assert window["paused"] is True
        assert not rig.arena._windows.has(rig.run(mcp._caller()), "unpause_window")
        rig.run(asyncio.sleep(0.2))
        frozen = rig.run(mcp._sim_now())
        rig.run(asyncio.sleep(0.3))
        assert rig.run(mcp._sim_now()) == pytest.approx(frozen, abs=1e-6)
    finally:
        rig.run(mcp.runtime_release("window"))


async def _status_where(rig: Rig, pred: object, what: str) -> dict:
    holder: dict = {}

    async def check() -> bool:
        holder["status"] = await mcp.runtime_lockstep_status()
        return pred(holder["status"])

    await rig.until(check, 10.0, what)
    return holder["status"]


def _cli_channels(status: dict) -> list[dict]:
    return [channel for reg in status["registrations"] if reg["caller"] == "cli" for channel in reg["channels"]]


async def _gate_reachable() -> bool:
    try:
        await mcp.runtime_lockstep_gate([])
    except ToolError as exc:
        if "lockstep unreachable" in str(exc):
            return False
        raise
    return True


@pytest.mark.skipif(not cli.sourced(), reason="needs a shell that sourced arena")
def test_lockstep_gate_runs_the_cli_verb_with_specs_and_presets(rig: Rig) -> None:
    rig.run(rig.until(_gate_reachable, 60.0, "the ros2 cli to discover the register service"), timeout=120.0)
    gated = rig.run(mcp.runtime_lockstep_gate([f"beat|/mcp/beat|{BEAT}|0.1|soft"]), timeout=120.0)
    assert gated["command"].startswith("arena lockstep gate") and gated["output"] == "cli gates: beat"
    status = rig.run(_status_where(rig, lambda status: [channel["name"] for channel in _cli_channels(status)] == ["beat"], "cli registration"))
    assert _cli_channels(status)[0]["hard"] is False
    with pytest.raises(ToolError, match="cannot resolve type"):
        rig.run(mcp.runtime_lockstep_gate(["bad|/mcp/beat|nope/msg/Nope|0.1|hard"]), timeout=120.0)
    with pytest.raises(ToolError, match="period_s must be > 0"):
        rig.run(mcp.runtime_lockstep_gate([f"bad|/mcp/beat|{BEAT}|0|hard"]), timeout=120.0)
    with pytest.raises(ToolError, match="bad channel spec 'nope'"):
        rig.run(mcp.runtime_lockstep_gate(["nope"]), timeout=120.0)
    assert rig.run(mcp.runtime_lockstep_gate(["peds"]), timeout=120.0)["output"] == "cli gates: peds"
    peds = rig.run(_status_where(rig, lambda status: [channel["name"] for channel in _cli_channels(status)] == ["peds"], "peds preset registered"))
    assert _cli_channels(peds)[0]["hard"] is True and "{env}" in _cli_channels(peds)[0]["topic"]
    assert rig.run(mcp.runtime_lockstep_gate([]), timeout=120.0)["output"] == "cli gates cleared"
    rig.run(_status_where(rig, lambda status: status["registrations"] == [], "registrations cleared"))


def test_lockstep_start_pause_resume_stop(rig: Rig) -> None:
    with pytest.raises(ToolError, match="not running"):
        rig.run(mcp.runtime_lockstep_pause())
    assert rig.run(mcp.runtime_lockstep_start(rtf=1.0, gated=False)) == {"lockstep": "on", "target_rtf": 1.0, "gated": False}
    rig.run(_status_where(rig, lambda status: status["active"] and status["ungated"] and status["measured_rtf"] > 0.3, "active ungated run"))
    assert rig.run(mcp.runtime_state())["paused"] is True
    assert rig.run(mcp.runtime_lockstep_pause()) == {"lockstep": "paused"}
    rig.run(_status_where(rig, lambda status: status["paused"], "paused run"))
    assert rig.run(mcp.runtime_lockstep_resume()) == {"lockstep": "resumed"}
    rig.run(_status_where(rig, lambda status: status["active"] and not status["paused"], "resumed run"))
    assert rig.run(mcp.runtime_lockstep_stop()) == {"lockstep": "off"}
    rig.run(_status_where(rig, lambda status: not status["active"], "stopped run"))


def test_lockstep_start_for_seconds_pauses_through_a_job(rig: Rig) -> None:
    started = rig.run(mcp.runtime_lockstep_start(rtf=2.0, seconds=0.5))
    job = started["job"]
    assert job["kind"] == "runtime_lockstep"
    done = rig.run(JOBS.wait(job["id"], 10.0))
    assert done.state == "succeeded" and done.result == {"lockstep": "paused", "after_s": 0.5}
    status = rig.run(_status_where(rig, lambda status: status["paused"], "paused after the job"))
    assert status["active"] and status["target_rtf"] == 2.0 and not status["ungated"]
    rig.run(mcp.runtime_lockstep_stop())
    rig.run(_status_where(rig, lambda status: not status["active"], "stopped run"))


def test_despawn_evicts_the_env_and_cleanup_reports_removed(rig: Rig) -> None:
    env_id = rig.env.env_id
    with pytest.raises(ToolError, match=f"no env {env_id + 7}"):
        rig.run(mcp.runtime_despawn_env(env_id + 7))
    despawned = rig.run(mcp.runtime_despawn_env(timeout_s=20.0), timeout=30.0)
    assert despawned == {"env_id": env_id, "requested": True, "gone": True}
    assert rig.run(mcp.runtime_envs()) == []
    with pytest.raises(ToolError, match="no env is up"):
        rig.run(mcp.runtime_despawn_env())
    assert rig.run(mcp.runtime_cleanup_env(env_id)) == {"env_id": env_id, "removed": 0}


def test_spawn_is_refused_without_a_task_layer(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    from arena_runtime.env.task_layer import TASK_LAYER_GROUP

    real = importlib.metadata.entry_points
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda **kwargs: [] if kwargs.get("group") == TASK_LAYER_GROUP else real(**kwargs))
    with pytest.raises(ToolError, match=TASK_LAYER_GROUP):
        rig.run(mcp.runtime_spawn_env())


def test_spawn_is_refused_once_the_sim_is_dead(rig: Rig) -> None:
    rig.run(rig.arena._mark_sim_dead("killed by the mcp test"))
    state = rig.run(_dead_state(rig))
    assert state["healthy"] is False and state["sim"]["reason"] == "killed by the mcp test"
    with pytest.raises(ToolError, match="sim dead: killed by the mcp test"):
        rig.run(mcp.runtime_spawn_env({"robot": "jackal"}))
    started = time.monotonic()
    with pytest.raises(ToolError, match="sim dead"):
        rig.run(mcp.runtime_spawn_env(wait_ready=False))
    assert time.monotonic() - started < 5.0


async def _dead_state(rig: Rig) -> dict:
    holder: dict = {}

    async def dead() -> bool:
        holder["state"] = await mcp.runtime_state()
        return not holder["state"]["healthy"]

    await rig.until(dead, 5.0, "sim dead in the state")
    return holder["state"]


def test_server_shutdown_releases_only_its_own_holds(rig: Rig) -> None:
    from arena_runtime_msgs.srv import LifecycleHold

    rig.run(mcp.runtime_hold("left_over"))
    rig.run(_other_hold(rig, LifecycleHold.Request.ACQUIRE))
    caller = rig.run(mcp._caller())
    assert rig.arena._holds.has(caller, "left_over")
    rig.run(ROS.shutdown())
    assert not rig.arena._holds.has(caller, "left_over")
    assert rig.arena._holds.has("/mcp_other_holder", "foreign")


def test_server_shutdown_keeps_a_context_it_did_not_start(rig: Rig) -> None:
    import rclpy

    rig.run(ROS.node())
    rig.run(ROS.shutdown())
    assert rclpy.ok()


def test_server_shutdown_ends_a_context_it_started(tmp_path) -> None:
    code = "import asyncio\nimport rclpy\nfrom arena_mcp.core import ROS\nasync def main():\n    await ROS.node()\n    assert rclpy.ok()\n    await ROS.shutdown()\n    assert not rclpy.ok()\nasyncio.run(main())\n"
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr
