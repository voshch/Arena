from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Annotated, Literal, cast

from pydantic import Field

from arena_mcp.core import JOBS, LATCHED, ROS, Domain, ToolError, msgs, robot_ns, tg
from arena_mcp.core.envs import SUBMIT_TASK_TIMEOUT_S
from arena_mcp.core.ros import python_to_param
from arena_runtime.env.task_layer import declare_mode_schemas, decode_modes, encode_modes, task_layer, task_layer_name

if TYPE_CHECKING:
    from arena_runtime_msgs.msg import EpisodeRecord
    from rcl_interfaces.msg import Parameter, ParameterDescriptor
    from std_msgs.msg import Bool, String

    from arena_mcp.core import JobHandle
    from arena_mcp.core.ros import TopicCache
    from arena_runtime.env.task_layer import TaskLayer, TaskModes

domain = Domain(
    "episode",
    """Episodes of one env's task generator: reset, run, wait, queue the next episode's task modes and params, records and history, catalogs, parameters.
    Every tool takes env_id, optional when exactly one env runs.
    episode_run returns a job (jobs_wait, jobs_cancel). Pausing the sim is a runtime hold, spawning and moving things is the entities domain.""",
    requires=("arena_runtime_msgs",),
)

HISTORY = 200
RESET_TIMEOUT = 120.0
BUSY = "RunEpisode is single-flight: wait for the running episode (episode_wait until=terminal) or cancel the job that runs it, an env with task.episode.auto_reset true runs its own episodes and is always busy"

CATALOGS = {
    "worlds": ("QueryWorlds", "query/worlds"),
    "scenarios": ("QueryScenarios", "query/scenarios"),
    "robots": ("QueryRobots", "query/robots"),
    "static_obstacles": ("QueryStaticObstacles", "query/static_obstacles"),
    "dynamic_obstacles": ("QueryDynamicObstacles", "query/dynamic_obstacles"),
    "environments": ("QueryEnvironments", "query/environments"),
    "parametrizeds": ("QueryParametrizeds", "query/parametrizeds"),
}

EnvId = Annotated[int | None, Field(description="env to act on, optional when exactly one env runs")]
ParamValues = dict[str, bool | int | float | str | list[bool] | list[int] | list[float] | list[str]]


def _states() -> dict[int, str]:
    from arena_runtime_msgs.msg import EpisodeRecord

    return msgs.constants(EpisodeRecord)


def _state(name: str) -> int:
    return {value: key for key, value in _states().items()}[name]


def _terminal() -> frozenset[int]:
    from arena_runtime_msgs.msg import EpisodeRecord

    return frozenset((EpisodeRecord.SUCCESS, EpisodeRecord.FAILED, EpisodeRecord.SKIPPED, EpisodeRecord.FATAL))


def _json(text: str, empty: object) -> object:
    if not text:
        return empty
    try:
        return json.loads(text)
    except ValueError:
        return text


def _params(params: Iterable[Parameter]) -> dict[str, object]:
    from rclpy.parameter import parameter_value_to_python

    return {param.name: msgs.jsonable(parameter_value_to_python(param.value)) for param in params}


def record(msg: EpisodeRecord) -> dict[str, object]:
    """An EpisodeRecord with its outcome name, start time in seconds, modes as axis to names, params as name to value and phases and conditions parsed."""
    out = msgs.to_dict(msg)
    out["outcome"] = _states().get(msg.outcome_state, str(msg.outcome_state))
    out["start_time"] = msgs.seconds(msg.start_time)
    out["modes"] = decode_modes(msg.modes)
    out["params"] = _params(msg.params)
    out["human_params"] = _params(msg.human_params)
    out["conditions"] = _json(msg.conditions, [])
    out["phases"] = _json(msg.phases, {})
    return out


async def _episodes(node: str) -> TopicCache:
    from arena_runtime_msgs.msg import EpisodeRecord

    return await ROS.cache(EpisodeRecord, f"{node}/state/episode", qos=LATCHED, depth=HISTORY)


async def _latched(msg_type: type, topic: str) -> TopicCache:
    return await ROS.cache(msg_type, topic, qos=LATCHED)


async def _fresh(cache: TopicCache, previous: object | None, timeout: float = 1.0) -> object | None:
    deadline = time.monotonic() + timeout
    while cache.last is previous and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    return cache.last


async def _first(cache: TopicCache, timeout: float = 1.0) -> object | None:
    if cache.last is not None:
        return cache.last
    try:
        return await asyncio.wait_for(cache.next(), timeout)
    except TimeoutError:
        return cache.last


async def _resetting(node: str) -> TopicCache:
    from std_msgs.msg import Bool

    cache = await _latched(Bool, f"{node}/state/resetting")
    await _first(cache)
    return cache


def _flag(cache: TopicCache) -> bool:
    last = cast("Bool | None", cache.last)
    return last is not None and last.data


def _last(cache: TopicCache) -> EpisodeRecord | None:
    return cast("EpisodeRecord | None", cache.last)


def _by_id(cache: TopicCache) -> dict[int, EpisodeRecord]:
    latest: dict[int, EpisodeRecord] = {}
    for msg in tuple(cache.history):
        latest[msg.episode_id] = msg
    return latest


async def _until[T](check: Callable[[], T | None], timeout: float, what: str, hint: str) -> T:
    deadline = time.monotonic() + timeout
    while True:
        found = check()
        if found is not None:
            return found
        if time.monotonic() >= deadline:
            raise ToolError(f"{what} within {timeout:.0f}s", hint=hint)
        await asyncio.sleep(0.05)


def _summary(msg: EpisodeRecord | None) -> str:
    if msg is None:
        return "no episode record yet"
    return f"episode {msg.episode_id} is {_states().get(msg.outcome_state, msg.outcome_state)}"


@domain.tool()
async def episode_reset(
    env_id: EnvId = None,
    world: Annotated[str, Field(description="world to switch to, empty keeps the current one")] = "",
    seed: Annotated[int, Field(description="seed for an exact replay, -1 derives one")] = -1,
    wait: Annotated[bool, Field(description="block until the new episode runs")] = True,
    timeout_s: Annotated[float, Field(description="upper bound of the wait in seconds")] = RESET_TIMEOUT,
) -> dict[str, object]:
    """Start a new episode, ending the running one as SKIPPED; with wait, return the new episode's record once it runs."""
    from arena_runtime_msgs.srv import ResetEpisode

    node = await tg(env_id)
    episodes = await _episodes(node)
    resetting = await _resetting(node)
    before = cast("EpisodeRecord | None", await _first(episodes))
    base = before.episode_id if before is not None else 0
    response = await ROS.call(ResetEpisode, f"{node}/lifecycle/reset_episode", ResetEpisode.Request(world=world, seed=seed))
    if not response.success:
        raise ToolError(f"reset rejected: {response.error_msg}")
    if not wait:
        return {"env": node, "requested": True, "previous_episode_id": base}
    running = _state("running")

    def started() -> EpisodeRecord | None:
        last = _last(episodes)
        if last is None or last.episode_id <= base or _flag(resetting):
            return None
        return last if last.outcome_state == running or last.outcome_state in _terminal() else None

    msg = await _until(started, timeout_s, f"no episode after {base} started", "episode_wait until=running keeps waiting, a world swap can take longer")
    if msg.outcome_state == _state("fatal"):
        raise ToolError(f"the reset failed: {msg.outcome_info}", hint=f"assets_find world {msg.world} shows what each resolver answers" if world else "episode_history shows the failed record")
    return record(msg)


@domain.tool()
async def episode_run(
    env_id: EnvId = None,
    world: Annotated[str, Field(description="world to load first, empty keeps the current one")] = "",
    seed: Annotated[int, Field(description="seed for an exact replay, -1 derives one")] = -1,
) -> dict[str, object]:
    """Run one episode of a managed env (task.episode.auto_reset false) through the RunEpisode action as a job with no result timeout; the result carries state, info, episode_id and the record, jobs_cancel cancels the goal."""
    from arena_runtime_msgs.action import RunEpisode

    from arena_runtime.env.constants.runtime import RUN_EPISODE_BUSY, RUN_EPISODE_SHUTTING_DOWN

    node = await tg(env_id)
    episodes = await _episodes(node)
    goal = RunEpisode.Goal(world=world, seed=seed)
    wrapper, handle = await ROS.send_goal(RunEpisode, f"{node}/lifecycle/run_episode", goal, hint=BUSY)
    states = msgs.constants(RunEpisode.Result)

    async def body(job: JobHandle) -> dict[str, object]:
        job.progress(goal_id=bytes(handle.goal_id.uuid).hex())
        response = await wrapper.await_result(handle)
        result = response.result
        if result.state == RunEpisode.Result.SKIPPED and result.info == RUN_EPISODE_BUSY:
            raise ToolError("the env was already running an episode", hint=BUSY)
        if result.state == RunEpisode.Result.SKIPPED and result.info == RUN_EPISODE_SHUTTING_DOWN:
            raise ToolError(f"{node} is shutting down and runs no more episodes", hint="runtime_envs lists the envs still up, runtime_spawn_env adds one")
        out: dict[str, object] = {"env": node, "state": states.get(result.state, str(result.state)), "info": result.info, "episode_id": result.episode_id}
        found = _by_id(episodes).get(result.episode_id)
        if found is not None:
            out["record"] = record(found)
        return out

    async def cancel() -> None:
        await wrapper.cancel(handle)

    return JOBS.start_task("episode", f"episode on {node}", body, on_cancel=cancel).as_dict()


@domain.tool(read_only=True)
async def episode_wait(
    env_id: EnvId = None,
    until: Annotated[
        Literal["running", "terminal", "idle"],
        Field(description="running: an episode is under way. terminal: the episode running at call time (or the next one) ended. idle: no episode runs and no reset is in progress"),
    ] = "terminal",
    timeout_s: Annotated[float, Field(description="at most 300 seconds")] = 60.0,
) -> dict[str, object]:
    """Wait for an episode state and return the record that matched, None for idle before any episode."""
    if not 0 < timeout_s <= 300:
        raise ToolError("timeout_s must be in (0, 300]", hint="call again to keep waiting")
    node = await tg(env_id)
    episodes = await _episodes(node)
    resetting = await _resetting(node)
    base = cast("EpisodeRecord | None", await _first(episodes))
    terminal = _terminal()
    running = _state("running")
    target = 0 if base is None else base.episode_id + (1 if base.outcome_state in terminal else 0)

    def check() -> tuple[EpisodeRecord | None] | None:
        if until == "terminal":
            for episode_id, msg in sorted(_by_id(episodes).items()):
                if episode_id >= target and msg.outcome_state in terminal:
                    return (msg,)
            return None
        if _flag(resetting):
            return None
        last = _last(episodes)
        if until == "running":
            return (last,) if last is not None and last.outcome_state == running else None
        return (last,) if last is None or last.outcome_state in terminal else None

    (msg,) = await _until(check, timeout_s, f"no {until} episode", _summary(_last(episodes)))
    return {"env": node, "until": until, "record": record(msg) if msg is not None else None}


@domain.tool(read_only=True)
async def episode_wait_for_world(
    env_id: EnvId = None,
    timeout_s: Annotated[float, Field(description="upper bound in seconds")] = 120.0,
) -> dict[str, object]:
    """Block until the env's active world is fully loaded, then return its name."""
    from std_msgs.msg import String
    from std_srvs.srv import Empty

    node = await tg(env_id)
    await ROS.call(Empty, f"{node}/lifecycle/wait_for_world", timeout=timeout_s)
    world = cast("String | None", await _first(await _latched(String, f"{node}/state/world")))
    return {"env": node, "world": world.data if world is not None else None}


@domain.tool(read_only=True)
async def episode_current(env_id: EnvId = None) -> dict[str, object]:
    """The current episode record, the queued next one and whether a reset is in progress."""
    node = await tg(env_id)
    current = cast("EpisodeRecord | None", await _first(await _episodes(node)))
    queued = _last(await _queue(node))
    resetting = await _resetting(node)
    return {
        "env": node,
        "resetting": _flag(resetting),
        "current": record(current) if current is not None else None,
        "queued": record(queued) if queued is not None else None,
    }


@domain.tool(read_only=True)
async def episode_history(
    env_id: EnvId = None,
    limit: Annotated[int, Field(description="newest records to return")] = 20,
) -> list[dict[str, object]]:
    """Episode records seen since the server first watched the env (the topic replays its last 20), newest first, one per episode with its final outcome."""
    node = await tg(env_id)
    episodes = await _episodes(node)
    await _first(episodes)
    latest = _by_id(episodes)
    return [record(latest[episode_id]) for episode_id in sorted(latest, reverse=True)[:limit]]


async def _queue(node: str) -> TopicCache:
    from arena_runtime_msgs.msg import EpisodeRecord

    cache = await _latched(EpisodeRecord, f"{node}/state/queue")
    await _first(cache)
    return cache


async def _active(node: str) -> TaskModes:
    from arena_runtime_msgs.srv import GetTaskModes

    return decode_modes((await ROS.call(GetTaskModes, f"{node}/config/get_task_modes")).modes)


async def _declared(node: str, prefixes: list[str]) -> list[str]:
    from rcl_interfaces.srv import ListParameters

    request = ListParameters.Request(prefixes=prefixes, depth=ListParameters.Request.DEPTH_RECURSIVE)
    return sorted((await ROS.call(ListParameters, f"{node}/list_parameters", request)).result.names)


@functools.cache
def _layer(name: str) -> TaskLayer:
    try:
        return task_layer(name)
    except LookupError as exc:
        raise ToolError(str(exc), hint="build the package that provides the task modes") from exc


async def _env_layer_name(node: str, name: str = "") -> str:
    """The task layer `name` resolves to, else the one the env's task_layer parameter names."""
    return task_layer_name(name or (await ROS.get_params(node, ["task_layer"]))["task_layer"])


async def _typed(node: str, selected: TaskModes, values: Mapping[str, object]) -> list[Parameter]:
    from rcl_interfaces.msg import Parameter
    from rcl_interfaces.srv import DescribeParameters

    spaces = _layer(await _env_layer_name(node)).mode_namespaces()
    prefixes = sorted({str(spaces[axis][mode]) for axis, names in selected.items() for mode in names if mode in spaces.get(axis, {})})
    declared = set(await _declared(node, prefixes)) if prefixes else set()
    unknown = sorted(set(values) - declared)
    if unknown:
        raise ToolError(f"{unknown} are not parameters of the selected task modes {selected}", hint=f"their parameters: {sorted(declared)}, episode_task_modes lists them with types")
    names = list(values)
    described = await ROS.call(DescribeParameters, f"{node}/describe_parameters", DescribeParameters.Request(names=names))
    return [Parameter(name=name, value=python_to_param(values[name], descriptor.type)) for name, descriptor in zip(names, described.descriptors, strict=True)]


def _untyped(values: Mapping[str, object]) -> list[Parameter]:
    from rcl_interfaces.msg import Parameter

    return [Parameter(name=name, value=python_to_param(value, 0)) for name, value in values.items()]


@domain.tool(idempotent=True)
async def episode_queue(
    env_id: EnvId = None,
    modes: Annotated[dict[str, str | list[str]] | None, Field(description="axis to mode name (a list for an axis with many), replacing that axis' queued selection, [] clears an axis with many modes like modules, an axis with one mode needs exactly one name, an axis left out keeps the queued one")] = None,
    world: Annotated[str | None, Field(description="world of the next episode, None keeps the queued one")] = None,
    params: Annotated[ParamValues | None, Field(description="full parameter name to value of the selected modes, e.g. {'task.scenario.file': 'default'}")] = None,
    human_params: Annotated[ParamValues | None, Field(description="'<backend namespace>.<param>' to value for the human simulator")] = None,
) -> dict[str, object]:
    """Stage the next episode: modes, world and params apply at the next reset, each omitted field keeps what is queued."""
    from arena_runtime_msgs.srv import QueueEpisode

    node = await tg(env_id)
    layer = _layer(await _env_layer_name(node))
    requested: TaskModes = {axis: [name for name in ([value] if isinstance(value, str) else value) if name] for axis, value in (modes or {}).items()}
    try:
        requested = layer.validate_modes(requested)
    except ValueError as e:
        raise ToolError(str(e)) from e
    queue = await _queue(node)
    queued = _last(queue)
    selected = {**(decode_modes(queued.modes) if queued is not None else await _active(node)), **requested}
    values: dict[str, object] = dict(params or {})
    request = QueueEpisode.Request(action=QueueEpisode.Request.MERGE, modes=encode_modes(requested), world=world or "")
    if values:
        request.params = await _typed(node, selected, values)
    if human_params:
        request.human_params = _untyped(human_params)
    response = await ROS.call(QueueEpisode, f"{node}/config/queue_episode", request)
    if not response.success:
        raise ToolError(f"queue rejected: {response.error_msg}")
    staged = cast("EpisodeRecord | None", await _fresh(queue, queued))
    return {"env": node, "queued": record(staged) if staged is not None else None}


def _entry(leaf: str, descriptor: ParameterDescriptor, value: object) -> dict[str, object]:
    from rclpy.parameter import Parameter as RclpyParameter

    entry: dict[str, object] = {"name": leaf, "type": RclpyParameter.Type(descriptor.type).name.lower(), "value": msgs.jsonable(value)}
    if descriptor.description:
        entry["description"] = descriptor.description
    if descriptor.additional_constraints:
        entry["constraints"] = descriptor.additional_constraints
    if descriptor.read_only:
        entry["read_only"] = True
    for bounds in (*descriptor.floating_point_range, *descriptor.integer_range):
        entry["range"] = [bounds.from_value, bounds.to_value, bounds.step]
    return entry


def _bucket(prefixes: Mapping[str, str], names: list[str], entries: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {key: [] for key in prefixes}
    for name, entry in zip(names, entries, strict=True):
        for key, prefix in prefixes.items():
            if name.startswith(f"{prefix}."):
                out[key].append({**entry, "name": name[len(prefix) + 1 :]})
    return out


@functools.cache
def _registry_modes(name: str) -> dict[str, object]:
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from rcl_interfaces.srv import ListParameters

    layer = _layer(name)
    node = ROSParamServer(f"arena_mcp_task_modes_{os.getpid()}", start_parameter_services=False)
    try:
        declare_mode_schemas(layer, node)
        out: dict[str, object] = {"source": "registry", "task_layer": name}
        for axis, prefixes in layer.mode_namespaces().items():
            names = sorted(node.list_parameters(list(prefixes.values()), ListParameters.Request.DEPTH_RECURSIVE).names)
            entries = [_entry(name, descriptor, param.value) for name, descriptor, param in zip(names, node.describe_parameters(names), node.get_parameters(names), strict=True)]
            out[axis] = _bucket(prefixes, names, entries)
        out["fleet_only_robots"] = layer.fleet_only_robots()
        return out
    finally:
        node.destroy_node()


async def _live_modes(node: str) -> dict[str, object]:
    from rcl_interfaces.srv import DescribeParameters, GetParameters
    from rclpy.parameter import parameter_value_to_python

    name = await _env_layer_name(node)
    layer = _layer(name)
    out: dict[str, object] = {"source": "live", "env": node, "task_layer": name}
    for axis, namespaces in layer.mode_namespaces().items():
        prefixes = {mode: str(namespace) for mode, namespace in sorted(namespaces.items())}
        names = await _declared(node, list(prefixes.values())) if prefixes else []
        if not names:
            out[axis] = {mode: [] for mode in prefixes}
            continue
        described = await ROS.call(DescribeParameters, f"{node}/describe_parameters", DescribeParameters.Request(names=names))
        values = await ROS.call(GetParameters, f"{node}/get_parameters", GetParameters.Request(names=names))
        entries = [_entry(name, descriptor, parameter_value_to_python(value)) for name, descriptor, value in zip(names, described.descriptors, values.values, strict=True)]
        out[axis] = _bucket(prefixes, names, entries)
    out["fleet_only_robots"] = layer.fleet_only_robots()
    return out


@domain.tool(read_only=True)
async def episode_task_modes(
    env_id: EnvId = None,
    source: Annotated[
        Literal["auto", "live", "registry"],
        Field(description="live: the env's registered modes and current values. registry: the code's registry with defaults, no env needed. auto: live when an env resolves"),
    ] = "auto",
    task_layer: Annotated[str, Field(description="task runtime of the registry, empty = $ARENA_TASK_LAYER, else task_generator, a live env answers for its own")] = "",
) -> dict[str, object]:
    """Selectable task modes per axis of the task runtime, each with its parameters: leaf name, type, value, description, constraints."""
    if source == "registry":
        await ROS.node()
        return _registry_modes(task_layer_name(task_layer))
    try:
        node = await tg(env_id)
    except ToolError as exc:
        if source == "live" or env_id is not None:
            raise
        await ROS.node()
        return {**_registry_modes(task_layer_name(task_layer)), "reason": exc.message}
    return await _live_modes(node)


@domain.tool(read_only=True)
async def episode_task_modes_active(env_id: EnvId = None) -> dict[str, object]:
    """The task modes the running episode uses, per axis."""
    node = await tg(env_id)
    return {"env": node, "modes": await _active(node)}


def _episode_params() -> dict[str, str]:
    from arena_runtime.env.constants.runtime import EPISODE_PARAMS

    return EPISODE_PARAMS


@domain.tool(read_only=True)
async def episode_params(
    env_id: EnvId = None,
    names: Annotated[list[str] | None, Field(description="task.episode.* names, None reads all")] = None,
) -> dict[str, object]:
    """The episode-shaping task.episode.* parameters, re-read at every reset, with value and description."""
    catalog = _episode_params()
    wanted = list(catalog) if names is None else names
    unknown = sorted(set(wanted) - set(catalog))
    if unknown:
        raise ToolError(f"{unknown} are not episode parameters", hint=f"episode parameters: {sorted(catalog)}")
    node = await tg(env_id)
    values = await ROS.get_params(node, wanted)
    return {name: {"value": msgs.jsonable(values[name]), "description": catalog[name]} for name in wanted}


@domain.tool(idempotent=True)
async def episode_set_params(
    values: Annotated[dict[str, bool | int | float | str], Field(description="task.episode.* name to value, coerced to the declared type")],
    env_id: EnvId = None,
) -> dict[str, object]:
    """Set episode-shaping task.episode.* parameters; they apply at the next reset."""
    catalog = _episode_params()
    if not values:
        raise ToolError("nothing to set", hint=f"episode parameters: {sorted(catalog)}")
    unknown = sorted(set(values) - set(catalog))
    if unknown:
        raise ToolError(f"{unknown} are not episode parameters", hint=f"episode parameters: {sorted(catalog)}")
    node = await tg(env_id)
    results = await ROS.set_params(node, dict(values))
    failed = {name: result["reason"] for name, result in results.items() if not result["ok"]}
    if failed:
        raise ToolError(f"rejected: {failed}", hint=f"applied: {sorted(set(results) - set(failed))}")
    current = await ROS.get_params(node, list(values))
    return {name: msgs.jsonable(value) for name, value in current.items()}


@functools.cache
def _config_names(layer: str) -> tuple[str, ...]:
    import rclpy
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from rcl_interfaces.srv import ListParameters

    from arena_runtime.env.constants.runtime import EPISODE_PARAMS, Configuration

    node = ROSParamServer(f"arena_mcp_config_{os.getpid()}", start_parameter_services=False, parameter_overrides=[rclpy.Parameter("world", value="")])
    try:
        before = set(node.list_parameters([], ListParameters.Request.DEPTH_RECURSIVE).names)
        Configuration(node)
        _layer(layer).declare_task_params(node)
        return tuple(sorted(set(node.list_parameters([], ListParameters.Request.DEPTH_RECURSIVE).names) - before - set(EPISODE_PARAMS)))
    finally:
        node.destroy_node()


@domain.tool(read_only=True)
async def episode_config(
    env_id: EnvId = None,
    task_layer: Annotated[str, Field(description="task runtime whose mode params are included, empty = the env's own")] = "",
) -> dict[str, object]:
    """The env's startup configuration: every Configuration parameter that is not a task.episode.* one (sim, human, acoustics, world, adapters, task modes, ...)."""
    node = await tg(env_id)
    values = await ROS.get_params(node, list(_config_names(await _env_layer_name(node, task_layer))))
    return {name: msgs.jsonable(value) for name, value in values.items()}


@domain.tool(read_only=True)
async def episode_catalog(
    kind: Literal["worlds", "scenarios", "robots", "static_obstacles", "dynamic_obstacles", "environments", "parametrizeds"],
    env_id: EnvId = None,
    world: Annotated[str, Field(description="for scenarios: the world to list, empty means the current one")] = "",
) -> list[str]:
    """Names the env's resolvers can load for one kind of asset, network buckets included for obstacles."""
    import arena_runtime_msgs.srv

    type_name, path = CATALOGS[kind]
    srv_type = getattr(arena_runtime_msgs.srv, type_name)
    node = await tg(env_id)
    request = srv_type.Request(world=world) if kind == "scenarios" else srv_type.Request()
    response = await ROS.call(srv_type, f"{node}/{path}", request, timeout=60.0)
    return sorted(response.ids)


@domain.tool()
async def episode_submit_task(
    task_yaml: Annotated[str, Field(description="a phases list or {phases, conditions} mapping in the scenario.yaml format, poses in the map frame, an empty list stops the robot")],
    env_id: EnvId = None,
    robot: Annotated[str | None, Field(description="robot name, optional when the env has one robot")] = None,
) -> dict[str, object]:
    """Hand a robot a task in the running episode, dropping its unfinished phases; returns the episode-wide phase indices it got."""
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime_msgs.srv import SubmitTask

    node = await tg(env_id)
    name = str(Namespace(await robot_ns(env_id, robot)).robot_ns)
    response = await ROS.call(SubmitTask, f"{node}/runtime/submit_task", SubmitTask.Request(robot=name, task_yaml=task_yaml), timeout=SUBMIT_TASK_TIMEOUT_S)
    if not response.success:
        raise ToolError(f"task rejected: {response.error_msg}")
    return {"env": node, "robot": name, "phase_indices": list(response.phase_indices)}


@domain.resource("arena://episode/{env_id}/current", mime_type="application/json", description="current and queued episode records of an env")
async def current_resource(env_id: str) -> str:
    return json.dumps(await episode_current(int(env_id)))


@domain.resource("arena://episode/{env_id}/history", mime_type="application/json", description="episode records of an env, newest first")
async def history_resource(env_id: str) -> str:
    return json.dumps(await episode_history(int(env_id), limit=HISTORY))
