from __future__ import annotations

from typing import Any

from arena_bringup.viz_backends import env_idx_from_ns

from .errors import ToolError
from .msgs import to_dict
from .ros import LATCHED, ROS

RUNTIME = "/arena"
ENVS_TOPIC = f"{RUNTIME}/state/envs"
SUBMIT_TASK_TIMEOUT_S = 150.0
NO_RUNTIME = "no Arena runtime is up, system_launch starts one"


async def env_table(*, timeout: float = 2.0) -> list[dict[str, Any]]:
    """The runtime's env registry: env_id, task generator fqn, slot and readiness per env."""
    from arena_runtime_msgs.msg import EnvRegistry

    registry = await ROS.latest(EnvRegistry, ENVS_TOPIC, qos=LATCHED, timeout=timeout, hint=NO_RUNTIME)
    return [to_dict(record) for record in registry.envs]


async def _unmanaged() -> list[str]:
    graph = await ROS.graph()
    return [name for name in graph["nodes"] if name.rsplit("/", 1)[-1] == "arena_env_node"]


async def tg(env_id: int | None = None) -> str:
    """The task generator node of env_id, or of the only env when env_id is None."""
    try:
        table = await env_table()
    except ToolError:
        table = []
    if table:
        if env_id is None:
            if len(table) == 1:
                return table[0]["fqn"]
            raise ToolError(f"{len(table)} envs are up, pass env_id", hint=f"env_ids: {sorted(record['env_id'] for record in table)}")
        for record in table:
            if record["env_id"] == env_id:
                return record["fqn"]
        raise ToolError(f"no env {env_id}", hint=f"env_ids: {sorted(record['env_id'] for record in table)}")
    nodes = await _unmanaged()
    if not nodes:
        raise ToolError("no env is up", hint=NO_RUNTIME)
    if env_id is None and len(nodes) == 1:
        return nodes[0]
    matches = [node for node in nodes if env_id is not None and env_idx_from_ns(node) == env_id]
    if len(matches) == 1:
        return matches[0]
    raise ToolError(f"cannot pick an env among {nodes}", hint="pass env_id")


async def env_ns(env_id: int | None = None) -> str:
    """The env namespace that holds the task generator, e.g. /arena/env_0."""
    return (await tg(env_id)).rsplit("/", 1)[0]


async def fleet(env_id: int | None = None) -> list[dict[str, Any]]:
    """The live robots of an env: name, model, namespace, frames and caps."""
    from arena_runtime_msgs.msg import RobotFleet

    node = await tg(env_id)
    msg = await ROS.latest(RobotFleet, f"{node}/state/robots", qos=LATCHED, hint="the env has not finished its first reset")
    out = []
    for state in msg.robots:
        entry = to_dict(state.descriptor)
        entry["caps"] = [to_dict(cap) for cap in state.caps]
        entry["params"] = {kv.key: kv.value for kv in state.params}
        out.append(entry)
    return out


async def robot_ns(env_id: int | None = None, robot: str | None = None) -> str:
    """The namespace of robot in env_id, or of the only robot when robot is None."""
    robots = await fleet(env_id)
    if not robots:
        raise ToolError("the env has no robots", hint="entities_spawn_robot adds one")
    if robot is None:
        if len(robots) == 1:
            return robots[0]["ns"]
        raise ToolError(f"{len(robots)} robots in the env, pass robot", hint=f"robots: {[entry['name'] for entry in robots]}")
    for entry in robots:
        if robot in (entry["name"], entry["ns"]):
            return entry["ns"]
    raise ToolError(f"no robot {robot!r}", hint=f"robots: {[entry['name'] for entry in robots]}")
