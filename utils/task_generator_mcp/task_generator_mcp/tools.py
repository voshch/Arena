from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import rclpy.parameter
from arena_runtime_msgs.srv import LifecycleHold
from mcp.types import CallToolRequestParams, CallToolResult, TextContent, Tool
from rcl_interfaces.msg import Parameter, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from std_srvs.srv import Empty
from task_generator.constants import Constants
from task_generator_msgs.action import RunEpisode
from task_generator_msgs.srv import (
    DespawnRobot,
    GetTaskModes,
    QueryDynamicObstacles,
    QueryEnvironments,
    QueryParametrizeds,
    QueryRobots,
    QueryScenarios,
    QueryStaticObstacles,
    QueryWorlds,
    QueueEpisode,
    ResetEpisode,
    SpawnDynamic,
    SpawnRobot,
    SpawnStatic,
)

from .params import EPISODE_PARAMS, STATIC_CONFIG_PARAMS

if TYPE_CHECKING:
    from .ros_bridge import RosBridge

# Enum value lists compiled once from the Python source of truth.
TM_ROBOTS_VALUES: list[str] = [m.value for m in Constants.TaskMode.TM_Robots] + [""]
TM_OBSTACLES_VALUES: list[str] = [m.value for m in Constants.TaskMode.TM_Obstacles] + [""]
TM_MODULE_VALUES: list[str] = [m.value for m in Constants.TaskMode.TM_Module]

_RUN_EPISODE_STATE_NAMES: dict[int, str] = {
    RunEpisode.Result.QUEUED: "QUEUED",
    RunEpisode.Result.RUNNING: "RUNNING",
    RunEpisode.Result.SUCCESS: "SUCCESS",
    RunEpisode.Result.FAILED: "FAILED",
    RunEpisode.Result.SKIPPED: "SKIPPED",
    RunEpisode.Result.FATAL: "FATAL",
}

_POSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "position": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}},
        },
        "orientation": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"},
                "w": {"type": "number"},
            },
        },
        "frame_id": {"type": "string"},
    },
}


def _record_to_dict(record: object) -> dict[str, object]:
    return {
        "episode_id": record.episode_id,
        "world": record.world,
        "seed": record.seed,
        "tm_robots": record.tm_robots,
        "tm_obstacles": record.tm_obstacles,
        "tm_modules": list(record.tm_modules),
        "outcome_state": record.outcome_state,
        "outcome_info": record.outcome_info,
        "goal_uuid": record.goal_uuid,
        "integrity": record.integrity,
        "obstacles_params": [{"name": p.name, "type": p.value.type, "value": _param_value_to_python(p.value)} for p in record.obstacles_params],
        "robots_params": [{"name": p.name, "type": p.value.type, "value": _param_value_to_python(p.value)} for p in record.robots_params],
    }


def _param_value_to_python(pv: ParameterValue) -> object:
    type_map = {
        rclpy.parameter.Parameter.Type.BOOL: pv.bool_value,
        rclpy.parameter.Parameter.Type.INTEGER: pv.integer_value,
        rclpy.parameter.Parameter.Type.DOUBLE: pv.double_value,
        rclpy.parameter.Parameter.Type.STRING: pv.string_value,
        rclpy.parameter.Parameter.Type.BOOL_ARRAY: list(pv.bool_array_value),
        rclpy.parameter.Parameter.Type.INTEGER_ARRAY: list(pv.integer_array_value),
        rclpy.parameter.Parameter.Type.DOUBLE_ARRAY: list(pv.double_array_value),
        rclpy.parameter.Parameter.Type.STRING_ARRAY: list(pv.string_array_value),
    }
    return type_map.get(rclpy.parameter.Parameter.Type(pv.type))


def _python_to_param_value(name: str, value: object) -> Parameter:
    p = Parameter()
    p.name = name
    pv = ParameterValue()
    if isinstance(value, bool):
        pv.type = rclpy.parameter.Parameter.Type.BOOL.value
        pv.bool_value = value
    elif isinstance(value, int):
        pv.type = rclpy.parameter.Parameter.Type.INTEGER.value
        pv.integer_value = value
    elif isinstance(value, float):
        pv.type = rclpy.parameter.Parameter.Type.DOUBLE.value
        pv.double_value = value
    elif isinstance(value, str):
        pv.type = rclpy.parameter.Parameter.Type.STRING.value
        pv.string_value = value
    p.value = pv
    return p


def build_tools_list() -> list[Tool]:
    """Build the full tool list served on tools/list."""
    return [
        Tool(
            name="lifecycle_reset_episode",
            description="Advance to a new episode. Specify a world id to switch worlds; leave empty to inherit the current world. Pass seed >= 0 for bit-perfect replay of a historical episode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "world": {"type": "string", "default": "", "description": "World id for the new episode. Empty inherits current."},
                    "seed": {"type": "integer", "default": -1, "description": "Seed for RNG. -1 derives automatically; non-negative replays exactly."},
                },
            },
        ),
        Tool(
            name="lifecycle_pause",
            description="Pause the simulation clock.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="lifecycle_unpause",
            description="Unpause / resume the simulation clock.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="lifecycle_toggle_pause",
            description="Toggle the simulation pause state. Returns the resulting paused flag.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="lifecycle_wait_for_world",
            description="Block until the active world is fully loaded.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="lifecycle_run_episode",
            description="Queue and run a single episode to completion via the action server. Returns outcome state, reason, and episode_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "world": {"type": "string", "default": "", "description": "World to load before the episode; empty keeps current."},
                },
            },
        ),
        Tool(
            name="query_worlds",
            description="List all available world ids.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="query_scenarios",
            description="List scenario shortnames. Pass world to restrict to that world; empty uses current world.",
            inputSchema={
                "type": "object",
                "properties": {
                    "world": {"type": "string", "default": "", "description": "World to restrict results; empty = current world."},
                },
            },
        ),
        Tool(
            name="query_robots",
            description="List available robot model shortnames.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="query_static_obstacles",
            description="List available static obstacle model shortnames.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="query_dynamic_obstacles",
            description="List available dynamic obstacle (pedestrian/dynamic entity) model shortnames.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="query_environments",
            description="List available environment configuration shortnames.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="query_parametrizeds",
            description="List available parametrized obstacle configuration shortnames.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="query_episode",
            description="Get the current episode and the queued (next) episode.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="config_queue_episode",
            description="Queue episode configuration. Empty string for any field preserves the prior queued value. tm_modules replaces the current set unless keep_modules=true.",
            inputSchema={
                "type": "object",
                "properties": {
                    "tm_robots": {"type": "string", "enum": TM_ROBOTS_VALUES, "default": "", "description": "Robot goal-dispatch mode .value string."},
                    "tm_obstacles": {"type": "string", "enum": TM_OBSTACLES_VALUES, "default": "", "description": "Obstacle population mode .value string."},
                    "tm_modules": {
                        "type": "array",
                        "items": {"type": "string", "enum": TM_MODULE_VALUES},
                        "description": "Cross-cutting module .value strings. Replaces current set unless keep_modules=true.",
                    },
                    "keep_modules": {"type": "boolean", "default": False, "description": "When true, tm_modules is ignored and the prior queued module set is preserved."},
                    "world": {"type": "string", "default": "", "description": "World id for the queued episode. Empty keeps prior queued value."},
                },
            },
        ),
        Tool(
            name="config_get_task_modes",
            description="Get the currently active task modes (tm_robots, tm_obstacles, tm_modules) as .value strings.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="config_set_episode_params",
            description="Set one or more episode-shaping parameters. Only supplied (non-null) fields are written.",
            inputSchema={
                "type": "object",
                "properties": {
                    "timeout": {"type": "number", "description": "Episode timeout in seconds."},
                    "goal_tolerance_radius": {"type": "number", "description": "Distance from goal within which the episode is considered successful."},
                    "robot_safe_dist": {"type": "number", "description": "Minimum safe distance between robot and obstacles."},
                    "auto_reset": {"type": "boolean", "description": "When true the node auto-advances on episode terminal."},
                    "episodes": {"type": "integer", "description": "Number of episodes to run before shutdown. -1 = unlimited."},
                },
            },
        ),
        Tool(
            name="config_get_episode_params",
            description="Read current values of all episode-shaping parameters.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="config_get_static_config",
            description="Read the startup configuration parameters (sim, human, mobile/arm adapters).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="runtime_spawn_static",
            description="Spawn a static obstacle by model shortname. Omit pose for random placement via the active task mode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Static obstacle model shortname (see query_static_obstacles)."},
                    "pose": {**_POSE_SCHEMA, "description": "Optional spawn pose. Omit for random placement."},
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="runtime_spawn_dynamic",
            description="Spawn a dynamic obstacle / pedestrian by model shortname. Omit pose for random placement.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Dynamic obstacle model shortname (see query_dynamic_obstacles)."},
                    "pose": {**_POSE_SCHEMA, "description": "Optional spawn pose. Omit for random placement."},
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="runtime_spawn_robot",
            description="Spawn a robot by model shortname. Experimental. Omit name for auto-generated name. Omit pose for placement by the active task mode.",
            inputSchema={
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "Robot model shortname (see query_robots)."},
                    "name": {"type": "string", "default": "", "description": "Robot name; empty auto-generates."},
                    "pose": {**_POSE_SCHEMA, "description": "Optional spawn pose. Omit for placement by active task mode."},
                    "immediate": {"type": "boolean", "default": False, "description": "Provision into the live world now (idle), else commit on the next reset."},
                },
                "required": ["model"],
            },
        ),
        Tool(
            name="runtime_despawn_robot",
            description="Despawn a robot by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Robot name to despawn."},
                },
                "required": ["name"],
            },
        ),
    ]


async def dispatch_tool_call(params: CallToolRequestParams, bridge: RosBridge) -> CallToolResult:
    """Dispatch a tools/call request and wrap the outcome."""
    result = await _dispatch(params.name, params.arguments or {}, bridge)
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(result, default=str))],
        isError=isinstance(result, dict) and "error" in result,
    )


async def _dispatch(name: str, args: dict[str, object], bridge: RosBridge) -> object:
    if name == "lifecycle_reset_episode":
        req = ResetEpisode.Request()
        req.world = args.get("world", "")
        req.seed = int(args.get("seed", -1))
        resp = await bridge.client_reset_episode.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {"success": resp.success, "error_msg": resp.error_msg}

    if name in ("lifecycle_pause", "lifecycle_unpause", "lifecycle_toggle_pause"):
        req = LifecycleHold.Request()
        req.caller_id = bridge.get_fully_qualified_name()
        req.reason = "mcp_external_pause"
        if name == "lifecycle_pause":
            req.action = LifecycleHold.Request.ACQUIRE
        elif name == "lifecycle_unpause":
            req.action = LifecycleHold.Request.RELEASE
        else:
            req.action = LifecycleHold.Request.RELEASE if bridge.arena_paused else LifecycleHold.Request.ACQUIRE
        resp = await bridge.client_pause.call_timeout(req)
        if resp is None:
            return {"success": False}
        return {"success": True, "paused": resp.paused}

    if name == "lifecycle_wait_for_world":
        resp = await bridge.client_wait_for_world.call_timeout(Empty.Request())
        return {"success": resp is not None}

    if name == "lifecycle_run_episode":
        goal = RunEpisode.Goal()
        goal.world = args.get("world", "")
        wrapped = await bridge.action_run_episode.send_and_await(goal)
        if wrapped is None:
            return {"error": "timeout or rejected"}
        result = wrapped.result
        return {
            "state": _RUN_EPISODE_STATE_NAMES.get(result.state, str(result.state)),
            "info": result.info,
            "episode_id": result.episode_id,
        }

    if name == "query_worlds":
        resp = await bridge.client_query_worlds.call_timeout(QueryWorlds.Request())
        if resp is None:
            return {"error": "timeout"}
        return list(resp.ids)

    if name == "query_scenarios":
        req = QueryScenarios.Request()
        req.world = args.get("world", "")
        resp = await bridge.client_query_scenarios.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return list(resp.ids)

    if name == "query_robots":
        resp = await bridge.client_query_robots.call_timeout(QueryRobots.Request())
        if resp is None:
            return {"error": "timeout"}
        return list(resp.ids)

    if name == "query_static_obstacles":
        resp = await bridge.client_query_static_obstacles.call_timeout(QueryStaticObstacles.Request())
        if resp is None:
            return {"error": "timeout"}
        return list(resp.ids)

    if name == "query_dynamic_obstacles":
        resp = await bridge.client_query_dynamic_obstacles.call_timeout(QueryDynamicObstacles.Request())
        if resp is None:
            return {"error": "timeout"}
        return list(resp.ids)

    if name == "query_environments":
        resp = await bridge.client_query_environments.call_timeout(QueryEnvironments.Request())
        if resp is None:
            return {"error": "timeout"}
        return list(resp.ids)

    if name == "query_parametrizeds":
        resp = await bridge.client_query_parametrizeds.call_timeout(QueryParametrizeds.Request())
        if resp is None:
            return {"error": "timeout"}
        return list(resp.ids)

    if name == "query_episode":
        cur = bridge.current_episode
        que = bridge.queued_episode
        return {
            "current": _record_to_dict(cur) if cur is not None else None,
            "queued": _record_to_dict(que) if que is not None else None,
        }

    if name == "config_queue_episode":
        req = QueueEpisode.Request()
        req.action = QueueEpisode.Request.MERGE
        req.tm_robots = args.get("tm_robots", "")
        req.tm_obstacles = args.get("tm_obstacles", "")
        modules = args.get("tm_modules")
        req.tm_modules = list(modules) if modules is not None else []
        req.keep_modules = bool(args.get("keep_modules", False))
        req.world = args.get("world", "")
        resp = await bridge.client_queue_episode.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {"success": resp.success, "error_msg": resp.error_msg}

    if name == "config_get_task_modes":
        resp = await bridge.client_get_task_modes.call_timeout(GetTaskModes.Request())
        if resp is None:
            return {"error": "timeout"}
        return {
            "tm_robots": resp.tm_robots,
            "tm_obstacles": resp.tm_obstacles,
            "tm_modules": list(resp.tm_modules),
        }

    if name == "config_set_episode_params":
        params: list[Parameter] = []
        for key in EPISODE_PARAMS:
            val = args.get(key)
            if val is not None:
                params.append(_python_to_param_value(key, val))
        if not params:
            return {"success": True, "error_msg": "nothing to set"}
        req = SetParameters.Request()
        req.parameters = params
        resp = await bridge.client_set_parameters.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        results = [{"name": p.name, "successful": r.successful, "reason": r.reason} for p, r in zip(params, resp.results, strict=True)]
        return {"results": results}

    if name == "config_get_episode_params":
        req = GetParameters.Request()
        req.names = list(EPISODE_PARAMS)
        resp = await bridge.client_get_parameters.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {name_: _param_value_to_python(v) for name_, v in zip(EPISODE_PARAMS, resp.values, strict=True)}

    if name == "config_get_static_config":
        req = GetParameters.Request()
        req.names = list(STATIC_CONFIG_PARAMS)
        resp = await bridge.client_get_parameters.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {name_: _param_value_to_python(v) for name_, v in zip(STATIC_CONFIG_PARAMS, resp.values, strict=True)}

    if name == "runtime_spawn_static":
        req = SpawnStatic.Request()
        req.model = args["model"]
        req.pose, req.use_pose = bridge.pose_dict_to_msg(args.get("pose"))
        resp = await bridge.client_spawn_static.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {"id": resp.id, "success": resp.success, "error_msg": resp.error_msg}

    if name == "runtime_spawn_dynamic":
        req = SpawnDynamic.Request()
        req.model = args["model"]
        req.pose, req.use_pose = bridge.pose_dict_to_msg(args.get("pose"))
        resp = await bridge.client_spawn_dynamic.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {"id": resp.id, "success": resp.success, "error_msg": resp.error_msg}

    if name == "runtime_spawn_robot":
        req = SpawnRobot.Request()
        req.model = args["model"]
        req.name = args.get("name", "")
        req.immediate = bool(args.get("immediate", False))
        req.pose, req.use_pose = bridge.pose_dict_to_msg(args.get("pose"))
        resp = await bridge.client_spawn_robot.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {"name": resp.name, "success": resp.success, "error_msg": resp.error_msg}

    if name == "runtime_despawn_robot":
        req = DespawnRobot.Request()
        req.name = args["name"]
        resp = await bridge.client_despawn_robot.call_timeout(req)
        if resp is None:
            return {"error": "timeout"}
        return {"success": resp.success, "error_msg": resp.error_msg}

    return {"error": f"unknown tool: {name}"}
