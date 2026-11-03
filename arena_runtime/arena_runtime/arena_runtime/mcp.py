from __future__ import annotations

import asyncio
import collections
import contextlib
import time
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field

from arena_mcp.core import JOBS, LATCHED, ROS, Domain, ToolError, cli, env_table, msgs
from arena_mcp.core.envs import NO_RUNTIME, RUNTIME, tg
from arena_mcp.core.jobs import FAILED, RUNNING, JobHandle

if TYPE_CHECKING:
    from arena_runtime_msgs.msg import HoldRegistry, LockstepStatus
    from arena_runtime_msgs.srv import LifecycleHold, SpawnEnv
    from std_srvs.srv import Trigger

domain = Domain(
    "runtime",
    """The Arena runtime (/arena): env table, spawn and despawn, sim health, pause holds, stepping, unpause windows and the lockstep scheduler.
    One runtime drives one simulator, so pause, step and lockstep act on every env of it at once.
    Holds owned by this server are the runtime's holders under its caller_id (its node name), released on request or when the server exits, never another caller's.""",
    requires=("rclpy", "arena_runtime_msgs"),
)

SIM_STATE = f"{RUNTIME}/state/sim"
HOLDERS = f"{RUNTIME}/state/holders"
SPAWN_ENV = f"{RUNTIME}/spawn_env"
DESPAWN_ENV = f"{RUNTIME}/despawn_env"
CLEANUP_ENV = f"{RUNTIME}/cleanup_env"
HOLD = f"{RUNTIME}/sim_lifecycle/hold"
STEP = f"{RUNTIME}/sim_lifecycle/step"
WINDOW = f"{RUNTIME}/sim_lifecycle/unpause_window"
LOCKSTEP = f"{RUNTIME}/sim_lifecycle/lockstep"


class _Own:
    def __init__(self) -> None:
        self.windows = 0
        self.lock = asyncio.Lock()


_OWN = _Own()


async def _caller() -> str:
    return (await ROS.node()).get_fully_qualified_name()


async def _latest[MsgT](msg_type: type[MsgT], name: str, timeout: float = 2.0) -> MsgT:
    return await ROS.latest(msg_type, f"{RUNTIME}/state/{name}", qos=LATCHED, timeout=timeout, hint=NO_RUNTIME)


async def _runtime_up(timeout: float = 2.0) -> bool:
    node = await ROS.node()
    deadline = time.monotonic() + timeout
    while node.count_publishers(SIM_STATE) == 0:
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.1)
    return True


async def _require(srv_type: type, name: str) -> None:
    if not await ROS.ready(srv_type, name, timeout=2.0):
        raise ToolError(f"service {name} is not available", hint=NO_RUNTIME)


def _holders(registry: HoldRegistry) -> list[dict[str, Any]]:
    return [msgs.to_dict(entry) for entry in registry.holds]


def _mine(registry: HoldRegistry, caller: str) -> collections.Counter[str]:
    return collections.Counter({entry.reason: entry.count for entry in registry.holds if entry.caller_id == caller})


async def _published(total: int) -> HoldRegistry:
    from arena_runtime_msgs.msg import HoldRegistry

    cache = await ROS.cache(HoldRegistry, HOLDERS, qos=LATCHED)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        arrived = cache.next()
        last = cache.last
        if isinstance(last, HoldRegistry) and sum(entry.count for entry in last.holds) == total:
            return last
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(arrived, deadline - time.monotonic())
    return await _latest(HoldRegistry, "holders")


def _lockstep_dict(status: LockstepStatus) -> dict[str, Any]:
    out = msgs.to_dict(status)
    out["tick"] = msgs.seconds(status.tick)
    return out


def _env_row(record: dict[str, Any]) -> dict[str, Any]:
    beat = record["last_heartbeat"]
    stamp = beat["sec"] + beat["nanosec"] * 1e-9
    return {
        "env_id": record["env_id"],
        "fqn": record["fqn"],
        "ns": record["fqn"].rsplit("/", 1)[0],
        "ready": record["ready"],
        "placed": record["placed"],
        "slot": {"reference": record["reference"], "size": record["slot_extent"], "prespawn": record["prespawn"], "extent": record["extent"]},
        "heartbeat_age_s": round(time.time() - stamp, 2) if stamp else None,
    }


async def _resolve(env_id: int | None) -> int:
    fqn = await tg(env_id)
    for record in await env_table():
        if record["fqn"] == fqn:
            return record["env_id"]
    raise ToolError(f"{fqn} is not registered with the runtime", hint="runtime_envs lists the registered envs")


async def _sim_now() -> float | None:
    node = await ROS.node()
    if node.sim_time.to_seconds() == 0.0:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(node.next_clock(), 1.0)
    now = node.sim_time.to_seconds()
    return now if now > 0.0 else None


async def _sim_reached(target: float) -> float | None:
    from arena_rclpy_mixins.Time import Time

    node = await ROS.node()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(node.await_sim_time(Time.from_float(target - 1e-6), freeze_timeout=1.0), 2.0)
    return await _sim_now()


@domain.tool(read_only=True)
async def runtime_state() -> dict[str, Any]:
    """Whether a runtime is up and healthy: sim alive or dead (terminal) with reason, paused (global for all envs), holders, lockstep, env counts and the runtime's parameters."""
    from arena_runtime_msgs.msg import EnvRegistry, HoldRegistry, LockstepStatus, SimState
    from std_msgs.msg import Bool

    if not await _runtime_up():
        return {"running": False, "hint": NO_RUNTIME}
    paused, sim, holders, lockstep, envs = await asyncio.gather(
        _latest(Bool, "paused"),
        _latest(SimState, "sim"),
        _latest(HoldRegistry, "holders"),
        _latest(LockstepStatus, "lockstep"),
        _latest(EnvRegistry, "envs"),
    )
    status = _lockstep_dict(lockstep)
    caller = await _caller()
    return {
        "running": True,
        "healthy": sim.alive,
        "sim": {"alive": sim.alive, "reason": sim.reason, "since_wall": msgs.seconds(sim.header.stamp)},
        "paused": paused.data,
        "holders": _holders(holders),
        "own": {"caller_id": caller, "holds": dict(_mine(holders, caller)), "windows": _OWN.windows},
        "lockstep": {key: status[key] for key in ("active", "paused", "ungated", "target_rtf", "measured_rtf", "tick", "waiting_on", "gated_ticks")},
        "envs": {"count": len(envs.envs), "ready": sum(1 for record in envs.envs if record.ready), "env_ids": sorted(record.env_id for record in envs.envs)},
        "params": await ROS.get_params(RUNTIME),
    }


@domain.tool(read_only=True)
async def runtime_envs() -> list[dict[str, Any]]:
    """The runtime's env table: env_id, task generator fqn, ns, ready, placed, slot, and seconds since the last heartbeat (or since registration before the first)."""
    return [_env_row(record) for record in await env_table()]


async def _spawn(request: SpawnEnv.Request, timeout_s: float) -> dict[str, Any]:
    from arena_runtime_msgs.srv import SpawnEnv

    await _require(SpawnEnv, SPAWN_ENV)
    try:
        response = await ROS.call(SpawnEnv, SPAWN_ENV, request, timeout=timeout_s)
    except ToolError as exc:
        raise ToolError(f"the env did not reach ACTIVE within {timeout_s:.0f}s", hint="the runtime keeps waiting for it, runtime_envs shows it and runtime_despawn_env stops it") from exc
    if not response.success:
        raise ToolError(f"spawn failed: {response.error_msg}", hint=f"env log: {response.log_path}" if response.log_path else None)
    return {"env_id": response.env_id, "fqn": f"/{response.ns.lstrip('/')}", "log_path": response.log_path, "ready": True}


@domain.tool()
async def runtime_spawn_env(
    launch_args: Annotated[dict[str, str] | None, Field(description="env.launch.py arguments without the := (e.g. {'robot': 'jackal'}), merged over the runtime's env_args; env.id, env.ns and sim are set by the runtime")] = None,
    wait_ready: Annotated[bool, Field(description="wait until the env is ACTIVE, else return at once with a job that finishes when it is")] = True,
    headless: Annotated[bool, Field(description="env output only to its log file, false also echoes it on the runtime's stdout")] = True,
    ns: Annotated[str, Field(description="explicit task generator node path, empty allocates arena/env_<id>/arena_env_node")] = "",
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Spawn one env as a child of the runtime and return its env_id, fqn and log_path."""
    from arena_runtime_msgs.srv import SpawnEnv

    request = SpawnEnv.Request(ns=ns, headless=headless, launch_args=[f"{key}:={value}" for key, value in (launch_args or {}).items()])
    if wait_ready:
        return await _spawn(request, timeout_s)
    await _require(SpawnEnv, SPAWN_ENV)
    before = {record["env_id"] for record in await env_table()}

    async def body(handle: JobHandle) -> dict[str, Any]:
        result = await _spawn(request, timeout_s)
        handle.log(f"env {result['env_id']} ACTIVE, log {result['log_path']}")
        return result

    job = JOBS.start_task("runtime_spawn", f"spawn env {launch_args or {}}", body)
    deadline = time.monotonic() + 10.0
    fresh: list[dict[str, Any]] = []
    while not fresh and job.state == RUNNING and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
        fresh = [record for record in await env_table() if record["env_id"] not in before]
    if job.state == FAILED:
        raise ToolError(job.error or "spawn failed", hint="jobs_log shows the details")
    out: dict[str, Any] = {"env_id": fresh[0]["env_id"] if fresh else None, "fqn": fresh[0]["fqn"] if fresh else None, "ready": False, "job": job.as_dict()}
    if job.result is not None:
        out.update(job.result)
    return out


@domain.tool(destructive=True)
async def runtime_despawn_env(
    env_id: int | None = None,
    wait: Annotated[bool, Field(description="wait until the env has left the env table")] = True,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Ask a registered env to shut itself down through its lifecycle, the runtime then evicts it and cleans up its sim entities."""
    from arena_runtime_msgs.srv import DespawnEnv

    env_id = await _resolve(env_id)
    response = await ROS.call(DespawnEnv, DESPAWN_ENV, DespawnEnv.Request(env_id=env_id), hint=NO_RUNTIME)
    if not response.success:
        raise ToolError(response.error_msg, hint="runtime_envs lists the registered envs")
    gone = False
    deadline = time.monotonic() + timeout_s
    while wait and time.monotonic() < deadline:
        if env_id not in {record["env_id"] for record in await env_table()}:
            gone = True
            break
        await asyncio.sleep(0.2)
    return {"env_id": env_id, "requested": True, "gone": gone}


@domain.tool(destructive=True)
async def runtime_cleanup_env(env_id: int) -> dict[str, Any]:
    """Delete every sim entity under env_id's namespace prefix, also for envs that are gone; the env's registration is untouched."""
    from arena_runtime_msgs.srv import CleanupEnv

    response = await ROS.call(CleanupEnv, CLEANUP_ENV, CleanupEnv.Request(env_id=env_id), timeout=30.0, hint=NO_RUNTIME)
    if not response.success:
        raise ToolError(response.error_msg)
    return {"env_id": env_id, "removed": response.removed}


async def _hold_call(action: int, caller: str, reason: str, *, timeout: float = 10.0, ready: float = 2.0) -> LifecycleHold.Response:
    from arena_runtime_msgs.srv import LifecycleHold

    return await ROS.call(LifecycleHold, HOLD, LifecycleHold.Request(action=action, caller_id=caller, reason=reason), timeout=timeout, ready=ready, hint=NO_RUNTIME)


@domain.tool()
async def runtime_hold(reason: Annotated[str, Field(description="label of the hold, e.g. inspect; holds are counted per caller and reason")]) -> dict[str, Any]:
    """Acquire a pause hold owned by this server; the sim stays paused for every env while any caller holds one."""
    from arena_runtime_msgs.srv import LifecycleHold

    if not reason.strip():
        raise ToolError("reason must not be empty")
    caller = await _caller()
    async with _OWN.lock:
        response = await _hold_call(LifecycleHold.Request.ACQUIRE, caller, reason)
        registry = await _published(response.hold_count)
    return {"caller_id": caller, "paused": response.paused, "hold_count": response.hold_count, "own": dict(_mine(registry, caller)), "holders": _holders(registry)}


@domain.tool()
async def runtime_release(reason: Annotated[str | None, Field(description="release one count of this reason, None releases every hold this server owns")] = None) -> dict[str, Any]:
    """Release holds owned by this server, never another caller's; the sim unpauses once no caller holds any."""
    from arena_runtime_msgs.msg import HoldRegistry
    from arena_runtime_msgs.srv import LifecycleHold
    from std_msgs.msg import Bool

    caller = await _caller()
    async with _OWN.lock:
        registry = await _latest(HoldRegistry, "holders")
        mine = _mine(registry, caller)
        if reason is None:
            targets = list(mine.elements())
        elif mine[reason] > 0:
            targets = [reason]
        else:
            raise ToolError(f"this server holds no {reason!r}", hint=f"own holds: {dict(mine) or 'none'}, runtime_state lists every holder")
        paused = None
        hold_count = sum(entry.count for entry in registry.holds)
        for target in targets:
            response = await _hold_call(LifecycleHold.Request.RELEASE, caller, target)
            registry = await _published(response.hold_count)
            hold_count, paused = response.hold_count, response.paused
    if paused is None:
        paused = (await _latest(Bool, "paused")).data
    return {"caller_id": caller, "released": targets, "paused": paused, "hold_count": hold_count, "own": dict(_mine(registry, caller)), "holders": _holders(registry)}


@domain.tool()
async def runtime_step(
    seconds: Annotated[float, Field(gt=0, description="sim seconds to advance, quantized to physics steps")],
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """Advance a held sim (every env) by seconds of sim time and return the sim time before and after; needs an active hold and no open unpause window."""
    from arena_runtime_msgs.srv import LifecycleStep

    before = await _sim_now()
    response = await ROS.call(LifecycleStep, STEP, LifecycleStep.Request(seconds=float(seconds)), timeout=timeout_s, hint=NO_RUNTIME)
    if not response.success:
        raise ToolError(f"step refused: {response.error_msg}")
    after = await _sim_reached(before + response.advanced) if before is not None else await _sim_now()
    return {"requested": seconds, "advanced": response.advanced, "sim_time_before": before, "sim_time_after": after}


async def _window_call(action: int, caller: str, *, timeout: float = 10.0, ready: float = 2.0) -> None:
    from arena_runtime_msgs.srv import LifecycleUnpauseWindow

    response = await ROS.call(LifecycleUnpauseWindow, WINDOW, LifecycleUnpauseWindow.Request(action=action, caller_id=caller), timeout=timeout, ready=ready, hint=NO_RUNTIME)
    if not response.success:
        raise ToolError(f"unpause window refused: {response.error_msg}")


@domain.tool()
async def runtime_unpause_window(seconds: Annotated[float, Field(gt=0, le=120, description="wall seconds the sim runs freely")]) -> dict[str, Any]:
    """Let the sim (every env) run freely for seconds of wall time even while holds exist, then restore the pause the holds imply; returns sim time before and after."""
    from arena_runtime_msgs.srv import LifecycleUnpauseWindow
    from std_msgs.msg import Bool

    caller = await _caller()
    before = await _sim_now()
    await _window_call(LifecycleUnpauseWindow.Request.ACQUIRE, caller)
    _OWN.windows += 1
    try:
        await asyncio.sleep(seconds)
    finally:
        await asyncio.shield(_window_call(LifecycleUnpauseWindow.Request.RELEASE, caller))
        _OWN.windows -= 1
    after = await _sim_now()
    return {"seconds": seconds, "sim_time_before": before, "sim_time_after": after, "paused": (await _latest(Bool, "paused")).data}


async def _trigger(name: str) -> Trigger.Response:
    from std_srvs.srv import Trigger

    return await ROS.call(Trigger, f"{LOCKSTEP}/{name}", hint=NO_RUNTIME)


async def _lockstep_pause() -> None:
    await _trigger("pause")


@domain.tool()
async def runtime_lockstep_start(
    rtf: Annotated[float, Field(ge=0, description="target real-time factor, 0 = flat out")] = 0.0,
    gated: Annotated[bool, Field(description="wait on the registered hard channels every tick, false free-steps")] = True,
    seconds: Annotated[float | None, Field(gt=0, description="pause the run after this many wall seconds as a job (arena lockstep run for:=), None keeps it running (arena lockstep on)")] = None,
) -> dict[str, Any]:
    """Start or reconfigure the lockstep scheduler, which holds the sim (every env) and steps it tick by tick, gated on hard channels."""
    from arena_runtime_msgs.srv import LockstepStart

    response = await ROS.call(LockstepStart, f"{LOCKSTEP}/start", LockstepStart.Request(target_rtf=float(rtf), ungated=not gated), hint=NO_RUNTIME)
    if not response.success:
        raise ToolError(f"lockstep start refused: {response.error_msg}")
    out: dict[str, Any] = {"lockstep": "on", "target_rtf": rtf, "gated": gated}
    if seconds is not None:
        wait = float(seconds)

        async def body(handle: JobHandle) -> dict[str, Any]:
            await asyncio.sleep(wait)
            await _lockstep_pause()
            handle.log(f"lockstep paused after {wait:g}s")
            return {"lockstep": "paused", "after_s": wait}

        out["job"] = JOBS.start_task("runtime_lockstep", f"lockstep run for {wait:g}s", body, on_cancel=_lockstep_pause).as_dict()
    return out


@domain.tool()
async def runtime_lockstep_gate(
    channels: Annotated[list[str], Field(description="preset names (engine, peds) or 'name|topic|type|period_s|role' with role hard or soft, {env} in a topic expands per env; empty clears")],
) -> dict[str, Any]:
    """Replace the lockstep gate channels of the cli registration, global to all envs, through `arena lockstep gate`."""
    done = await asyncio.to_thread(cli.run, "lockstep", "gate", *channels, timeout=90.0)
    return {"command": done["command"], "output": done["stdout"].strip()}


@domain.tool(idempotent=True)
async def runtime_lockstep_stop() -> dict[str, Any]:
    """Stop the lockstep run and drop its hold, the sim free-runs unless other holds remain."""
    await _trigger("stop")
    return {"lockstep": "off"}


@domain.tool(idempotent=True)
async def runtime_lockstep_pause() -> dict[str, Any]:
    """Freeze a running lockstep at the next tick boundary, keeping the run and its hold."""
    response = await _trigger("pause")
    if not response.success:
        raise ToolError(response.message, hint="runtime_lockstep_start starts it")
    return {"lockstep": "paused"}


@domain.tool(idempotent=True)
async def runtime_lockstep_resume() -> dict[str, Any]:
    """Continue a paused lockstep run."""
    response = await _trigger("resume")
    if not response.success:
        raise ToolError(response.message, hint="runtime_lockstep_start starts it")
    return {"lockstep": "resumed"}


@domain.tool(read_only=True)
async def runtime_lockstep_status() -> dict[str, Any]:
    """The scheduler's latched status: active, paused, ungated, target and measured rtf, gated tick (sim s), registrations, channels it waits on, per-channel counters."""
    from arena_runtime_msgs.msg import LockstepStatus

    return _lockstep_dict(await _latest(LockstepStatus, "lockstep"))


@domain.resource("arena://runtime/state", mime_type="application/json", description="Runtime health: sim, pause, holders, lockstep, envs, params")
async def state_resource() -> dict[str, Any]:
    return await runtime_state()


@domain.resource("arena://runtime/envs", mime_type="application/json", description="The runtime's env table")
async def envs_resource() -> list[dict[str, Any]]:
    return await runtime_envs()


async def _release_own() -> None:
    from arena_runtime_msgs.msg import HoldRegistry
    from arena_runtime_msgs.srv import LifecycleHold, LifecycleUnpauseWindow

    if not ROS.started:
        return
    caller = await _caller()
    for _ in range(_OWN.windows):
        with contextlib.suppress(ToolError):
            await _window_call(LifecycleUnpauseWindow.Request.RELEASE, caller, timeout=1.0, ready=0.5)
    _OWN.windows = 0
    if (await ROS.node()).count_publishers(HOLDERS) == 0:
        return
    try:
        registry = await _latest(HoldRegistry, "holders", timeout=1.0)
    except ToolError:
        return
    for reason in list(_mine(registry, caller).elements()):
        with contextlib.suppress(ToolError):
            await _hold_call(LifecycleHold.Request.RELEASE, caller, reason, timeout=1.0, ready=0.5)


ROS.on_shutdown(_release_own)
