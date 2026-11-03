from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import BaseModel, Field

from arena_mcp.core import LATCHED, RELIABLE, ROS, Domain, ToolError, env_ns, msgs, tg

if TYPE_CHECKING:
    from arena_humansim_msgs.msg import AgentMeta
    from arena_people_msgs.msg import Pedestrian
    from arena_runtime_msgs.msg import SemanticEntityState
    from geometry_msgs.msg import PoseStamped

domain = Domain(
    "entities",
    """Things in one env: obstacles, robots, sounds, semantics, the map server, pedestrians and humansim behavior.
    Spawn, move and pedestrian poses are in the env's map frame, humansim poses in the engine's world frame.
    Every tool takes env_id, optional when exactly one env runs. entities_ped_* need the env's human simulator to be humansim (human:=arena).""",
    requires=("arena_runtime_msgs", "arena_people_msgs", "arena_runtime"),
)

EnvId = Annotated[int | None, Field(description="env to act on, optional when exactly one env runs")]
OptionalX = Annotated[float | None, Field(description="x in the map frame, omit with y for placement by the task mode")]
OptionalY = Annotated[float | None, Field(description="y in the map frame, omit with x for placement by the task mode")]
Yaw = Annotated[float, Field(description="heading in radians")]


class PedestrianPose(BaseModel):
    """A pedestrian by name and a map-frame pose."""

    name: str
    x: float
    y: float
    yaw: float = 0.0


class PedestrianState(BaseModel):
    """One pedestrian's state for the stream: map-frame pose, velocity and animation state name."""

    name: str
    x: float
    y: float
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    animation_state: str = "walking"


class _Outcome(Protocol):
    @property
    def success(self) -> bool: ...
    @property
    def error_msg(self) -> str: ...


class _Done(Protocol):
    @property
    def success(self) -> bool: ...
    @property
    def message(self) -> str: ...


class Waypoint(BaseModel):
    """A humansim waypoint in the engine's world frame, radius 0 uses the engine default."""

    x: float
    y: float
    radius: float = 0.0


def _stamped(x: float, y: float, yaw: float, frame: str = "map", z: float = 0.0) -> PoseStamped:
    from geometry_msgs.msg import PoseStamped

    out = PoseStamped()
    out.header.frame_id = frame
    out.pose = msgs.pose(x, y, yaw, z)
    return out


def _placement(x: float | None, y: float | None, yaw: float) -> tuple[PoseStamped, bool]:
    if (x is None) != (y is None):
        raise ToolError("pass both x and y, or neither for placement by the task mode")
    if x is None or y is None:
        return _stamped(0.0, 0.0, 0.0), False
    return _stamped(x, y, yaw), True


def _check(response: _Outcome, what: str) -> None:
    if not response.success:
        raise ToolError(f"{what} failed: {response.error_msg}")


@domain.tool()
async def entities_spawn_static(model: str, env_id: EnvId = None, x: OptionalX = None, y: OptionalY = None, yaw: Yaw = 0.0) -> dict[str, object]:
    """Spawn a static obstacle model (episode_catalog kind=static_obstacles) into the running episode; returns its handle for entities_move."""
    from arena_runtime_msgs.srv import SpawnStatic

    node = await tg(env_id)
    pose, use_pose = _placement(x, y, yaw)
    response = await ROS.call(SpawnStatic, f"{node}/runtime/spawn_static", SpawnStatic.Request(model=model, pose=pose, use_pose=use_pose), timeout=60.0)
    _check(response, f"spawning {model}")
    return {"env": node, "id": response.id}


@domain.tool()
async def entities_spawn_dynamic(model: str, env_id: EnvId = None, x: OptionalX = None, y: OptionalY = None, yaw: Yaw = 0.0) -> dict[str, object]:
    """Spawn a dynamic obstacle or pedestrian model (episode_catalog kind=dynamic_obstacles) into the running episode."""
    from arena_runtime_msgs.srv import SpawnDynamic

    node = await tg(env_id)
    pose, use_pose = _placement(x, y, yaw)
    response = await ROS.call(SpawnDynamic, f"{node}/runtime/spawn_dynamic", SpawnDynamic.Request(model=model, pose=pose, use_pose=use_pose), timeout=60.0)
    _check(response, f"spawning {model}")
    return {"env": node, "id": response.id}


@domain.tool()
async def entities_spawn_robot(
    model: str,
    env_id: EnvId = None,
    name: Annotated[str | None, Field(description="instance name, None generates one")] = None,
    x: OptionalX = None,
    y: OptionalY = None,
    yaw: Yaw = 0.0,
    immediate: Annotated[bool, Field(description="provision into the live world now (idle) instead of on the next reset")] = False,
    args: Annotated[dict[str, str | int | float | bool] | None, Field(description="robot args forwarded to Robot.parse, e.g. {'mobile': 'nav2', 'mobile.local_planner': 'dwb'}")] = None,
) -> dict[str, object]:
    """Add a robot model (episode_catalog kind=robots) to the env's fleet; returns the instance name."""
    from arena_runtime_msgs.srv import SpawnRobot
    from diagnostic_msgs.msg import KeyValue

    node = await tg(env_id)
    pose, use_pose = _placement(x, y, yaw)
    pairs = [KeyValue(key=key, value=str(value).lower() if isinstance(value, bool) else str(value)) for key, value in (args or {}).items()]
    request = SpawnRobot.Request(model=model, name=name or "", pose=pose, use_pose=use_pose, immediate=immediate, args=pairs)
    response = await ROS.call(SpawnRobot, f"{node}/runtime/spawn_robot", request, timeout=120.0)
    _check(response, f"spawning robot {model}")
    return {"env": node, "name": response.name, "immediate": immediate}


@domain.tool(destructive=True)
async def entities_despawn_robot(name: str, env_id: EnvId = None) -> dict[str, object]:
    """Toggle a robot's removal: stage a live robot for teardown on the next reset, or undo a staged spawn or despawn."""
    from arena_runtime_msgs.srv import DespawnRobot

    node = await tg(env_id)
    response = await ROS.call(DespawnRobot, f"{node}/runtime/despawn_robot", DespawnRobot.Request(name=name))
    _check(response, f"despawning {name}")
    return {"env": node, "name": name}


@domain.tool(idempotent=True)
async def entities_move(
    name: Annotated[str, Field(description="a robot name or the handle a runtime spawn returned")],
    x: float,
    y: float,
    env_id: EnvId = None,
    yaw: Yaw = 0.0,
) -> dict[str, object]:
    """Teleport a robot or a runtime-spawned obstacle or sound to a pose in the env's map frame."""
    from arena_runtime_msgs.srv import MoveEntity

    node = await tg(env_id)
    response = await ROS.call(MoveEntity, f"{node}/runtime/move", MoveEntity.Request(entity=name, pose=_stamped(x, y, yaw)), timeout=30.0)
    _check(response, f"moving {name}")
    return {"env": node, "name": name, "x": x, "y": y, "yaw": yaw}


@domain.tool(idempotent=True)
async def entities_require_map(env_id: EnvId = None) -> dict[str, object]:
    """Start the env's map server if it is not running yet."""
    from std_srvs.srv import Trigger

    node = await tg(env_id)
    response = await ROS.call(Trigger, f"{node}/runtime/require_map", timeout=60.0)
    if not response.success:
        raise ToolError(f"require_map failed: {response.message}")
    return {"env": node, "map_server": True}


def _semantic(state: SemanticEntityState) -> dict[str, object]:
    return {
        "entity": state.entity,
        "kind": state.kind,
        "discrete": dict(zip(state.discrete_names, state.discrete_values, strict=True)),
        "continuous": {name: msgs.jsonable(value) for name, value in zip(state.continuous_names, state.continuous_values, strict=True)},
        "predicates": dict(zip(state.predicate_names, state.predicate_values, strict=True)),
        "members": list(state.members),
    }


@domain.tool(read_only=True)
async def entities_semantics(env_id: EnvId = None) -> dict[str, object]:
    """Semantic state of every annotated entity (doors, elevators, signals, schedules, sounds, ...): discrete, continuous and predicate fields and members."""
    from arena_runtime_msgs.msg import SemanticSnapshot

    node = await tg(env_id)
    snapshot = await ROS.latest(SemanticSnapshot, f"{node}/state/semantics", qos=LATCHED, hint="the env has not finished its first reset")
    return {"env": node, "env_id": snapshot.env_id, "world": snapshot.world, "stamp": msgs.seconds(snapshot.stamp), "entities": [_semantic(state) for state in snapshot.entities]}


@domain.tool(idempotent=True)
async def entities_set_semantic(
    entity: Annotated[str, Field(description="env-prefixed entity name as entities_semantics lists it")],
    field: Annotated[str, Field(description="a writable state or predicate name of that entity")],
    value: Annotated[str | bool | float, Field(description="a token, a boolean or a number")],
    env_id: EnvId = None,
) -> dict[str, object]:
    """Write one semantic field of an annotated entity."""
    from arena_runtime_msgs.srv import SetSemantic

    node = await tg(env_id)
    text = str(value).lower() if isinstance(value, bool) else str(value)
    response = await ROS.call(SetSemantic, f"{node}/semantics/set", SetSemantic.Request(entity=entity, field=field, value=text))
    _check(response, f"setting {entity}.{field}")
    return {"env": node, "entity": entity, "field": field, "value": text}


SOUNDS_HINT = "runtime sounds need the sounds module: episode_queue modes={'modules': [..., 'sounds']} then episode_reset"


@domain.tool()
async def entities_spawn_sound(
    kind: Annotated[str, Field(description="environment sound kind, e.g. music")],
    x: float,
    y: float,
    env_id: EnvId = None,
    z: float = 0.0,
    yaw: Yaw = 0.0,
    frame: Annotated[str, Field(description="frame of the pose, map or a robot or object frame")] = "map",
    attach_to_frame: Annotated[bool, Field(description="follow the frame every tick instead of staying where it was spawned")] = False,
    asset_id: Annotated[str, Field(description="a specific sound asset, empty takes the kind's default")] = "",
    source_volume_db: Annotated[float | None, Field(description="source level in dB, required when asset_id, loop or initially_active deviate from the defaults")] = None,
    loop: bool = True,
    initially_active: bool = True,
) -> dict[str, object]:
    """Spawn an environment sound at a pose; returns the entity name entities_remove_sound and entities_move take."""
    from arena_runtime_msgs.srv import SpawnSound

    customize = bool(asset_id) or source_volume_db is not None or not loop or not initially_active
    if customize and source_volume_db is None:
        raise ToolError("customized playback needs source_volume_db", hint="pass the source level in dB, or keep asset_id empty and loop and initially_active true")
    node = await tg(env_id)
    request = SpawnSound.Request(
        pose=_stamped(x, y, yaw, frame, z),
        kind=kind,
        customize_playback=customize,
        asset_id=asset_id,
        source_volume_db=float(source_volume_db or 0.0),
        loop=loop,
        initially_active=initially_active,
        attach_to_frame=attach_to_frame,
    )
    response = await ROS.call(SpawnSound, f"{node}/runtime/spawn_sound", request, hint=SOUNDS_HINT)
    _check(response, f"spawning sound {kind}")
    return {"env": node, "entity": response.entity}


@domain.tool(destructive=True)
async def entities_remove_sound(entity: Annotated[str, Field(description="the entity name entities_spawn_sound returned")], env_id: EnvId = None) -> dict[str, object]:
    """Remove a runtime-spawned sound."""
    from arena_runtime_msgs.srv import RemoveSound

    node = await tg(env_id)
    response = await ROS.call(RemoveSound, f"{node}/runtime/remove_sound", RemoveSound.Request(entity=entity), hint=SOUNDS_HINT)
    _check(response, f"removing sound {entity}")
    return {"env": node, "entity": entity}


def _animation_states() -> dict[int, str]:
    from arena_people_msgs.msg import Pedestrian

    return msgs.constants(Pedestrian)


def _pedestrian(ped: Pedestrian) -> dict[str, object]:
    return {
        "name": ped.name,
        "id": ped.id,
        **msgs.pose_dict(ped.pose),
        "vx": ped.twist.linear.x,
        "vy": ped.twist.linear.y,
        "animation_state": _animation_states().get(ped.animation_state, str(ped.animation_state)),
        "gestures": [msgs.to_dict(gesture) for gesture in ped.gestures],
        "agent_type": ped.agent_type,
        "interaction_id": ped.interaction_id,
    }


@domain.tool(read_only=True)
async def entities_pedestrians(
    env_id: EnvId = None,
    full: Annotated[bool, Field(description="every message field, joint states included")] = False,
) -> dict[str, object]:
    """The env's pedestrians as the human simulator publishes them: name, id, map-frame pose, velocity, animation state, gestures."""
    from arena_people_msgs.msg import Pedestrians

    ns = await env_ns(env_id)
    msg = await ROS.latest(Pedestrians, f"{ns}/arena_peds", qos=RELIABLE, timeout=3.0, hint="no human simulator publishes pedestrians in this env")
    peds = [msgs.to_dict(ped) if full else _pedestrian(ped) for ped in msg.pedestrians]
    return {"env": ns, "frame": msg.header.frame_id, "stamp": msgs.seconds(msg.header.stamp), "pedestrians": peds}


@domain.tool(idempotent=True)
async def entities_move_pedestrians(moves: list[PedestrianPose], env_id: EnvId = None) -> dict[str, object]:
    """Teleport pedestrians to map-frame poses; returns which names were found."""
    from arena_people_msgs.msg import Pedestrian
    from arena_people_msgs.srv import MovePedestrians

    if not moves:
        raise ToolError("nothing to move", hint="entities_pedestrians lists the names")
    ns = await env_ns(env_id)
    request = MovePedestrians.Request(pedestrians=[Pedestrian(name=move.name, pose=msgs.pose(move.x, move.y, move.yaw)) for move in moves])
    response = await ROS.call(MovePedestrians, f"{ns}/human/move", request, timeout=30.0, hint="the env runs no human simulator")
    names = msgs.constants(MovePedestrians.Response)
    results = {move.name: names.get(code, str(code)) for move, code in zip(moves, response.results, strict=True)}
    if all(result == "not_found" for result in results.values()):
        raise ToolError(f"no pedestrian moved: {sorted(results)} not found", hint="entities_pedestrians lists the names, before the first reset none are tracked")
    return {"env": ns, "results": results}


@domain.tool()
async def entities_stream_pedestrians(states: list[PedestrianState], env_id: EnvId = None) -> dict[str, object]:
    """Publish one human/stream message stamped with the sim time: the named pedestrians follow these states until the stream goes quiet."""
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from builtin_interfaces.msg import Time

    codes = {name: value for value, name in _animation_states().items()}
    unknown = sorted({state.animation_state for state in states} - set(codes))
    if unknown:
        raise ToolError(f"unknown animation states {unknown}", hint=f"animation states: {sorted(codes)}")
    ns = await env_ns(env_id)
    now = await ROS.sim_time()
    msg = Pedestrians()
    msg.header.frame_id = "map"
    msg.header.stamp = Time(sec=int(now), nanosec=int((now - int(now)) * 1e9))
    peds: list[Pedestrian] = []
    for state in states:
        ped = Pedestrian(name=state.name, pose=msgs.pose(state.x, state.y, state.yaw), animation_state=codes[state.animation_state])
        ped.twist.linear.x, ped.twist.linear.y = state.vx, state.vy
        peds.append(ped)
    msg.pedestrians = peds
    await ROS.publish(Pedestrians, f"{ns}/human/stream", msg)
    return {"env": ns, "stamp": now, "published": [state.name for state in states]}


async def _humansim(env_id: int | None) -> str:
    from arena_runtime.env.constants import Constants

    node = await tg(env_id)
    human = (await ROS.get_params(node, ["human"]))["human"]
    if human != Constants.HumanSimulator.ARENA.value:
        raise ToolError(f"the env runs human simulator {human!r}, entities_ped_* calls need humansim", hint=f"launch the env with human:={Constants.HumanSimulator.ARENA.value}")
    return node


def _done(response: _Done, what: str) -> None:
    if not response.success:
        raise ToolError(f"{what} failed: {response.message}")


async def _agents(node: str) -> AgentMeta:
    from arena_humansim_msgs.msg import AgentMeta

    return await ROS.latest(AgentMeta, f"{node}/agent_meta", qos=LATCHED, hint="humansim has not published its agents yet")


@domain.tool(read_only=True)
async def entities_ped_agents(env_id: EnvId = None) -> dict[str, object]:
    """Humansim's own agents: id, name, agent type and handedness, plus the registered local planner policies."""
    node = await _humansim(env_id)
    meta = await _agents(node)
    agents = [{"agent_id": agent_id, "name": name, "agent_type": agent_type, "handedness": hand} for agent_id, name, agent_type, hand in zip(meta.agent_id, meta.name, meta.agent_type, meta.handedness, strict=False)]
    return {"env": node, "agents": agents, "policies": list(meta.policies)}


@domain.tool(idempotent=True)
async def entities_ped_waypoints(
    ped: Annotated[str | int, Field(description="humansim agent name or id (entities_ped_agents)")],
    waypoints: Annotated[list[Waypoint], Field(description="the new waypoint list, empty clears the agent's goals")],
    env_id: EnvId = None,
    mode: Literal["repeat", "reverse", "once", "random"] = "repeat",
) -> dict[str, object]:
    """Replace a humansim agent's waypoint list and cycling mode."""
    from arena_humansim_msgs.msg import Waypoint as WaypointMsg
    from arena_humansim_msgs.msg import Waypoints
    from arena_humansim_msgs.srv import SetWaypoints
    from geometry_msgs.msg import Pose2D

    node = await _humansim(env_id)
    modes = {name: value for value, name in msgs.constants(Waypoints, "MODE_").items()}
    points = Waypoints(points=[WaypointMsg(pose=Pose2D(x=point.x, y=point.y, theta=0.0), radius=point.radius) for point in waypoints], mode=modes[mode])
    request = SetWaypoints.Request(agent_id=ped if isinstance(ped, int) else 0, name="" if isinstance(ped, int) else ped, waypoints=points)
    _done(await ROS.call(SetWaypoints, f"{node}/set_waypoints", request), f"setting waypoints of {ped}")
    return {"env": node, "ped": ped, "waypoints": len(waypoints), "mode": mode}


@domain.tool()
async def entities_ped_flow(
    sources: Annotated[list[dict[str, object]], Field(description="arena_humansim_msgs/SourceConfig fields per source (name, pose {x, y, theta}, shape, type, rate_profile, max_concurrent, max_total, agent)")],
    sinks: Annotated[list[dict[str, object]], Field(description="arena_humansim_msgs/SinkConfig fields per sink (name, pose, shape, absorption_radius, capacity)")],
    env_id: EnvId = None,
) -> dict[str, object]:
    """Replace humansim's flow: the sources that emit pedestrians and the sinks that absorb them, poses in the engine's world frame."""
    from arena_humansim_msgs.msg import FlowConfig
    from arena_humansim_msgs.srv import SetFlow

    node = await _humansim(env_id)
    flow = msgs.fill(FlowConfig, {"sources": sources, "sinks": sinks})
    _done(await ROS.call(SetFlow, f"{node}/set_flow", SetFlow.Request(flow=flow)), "setting the flow")
    return {"env": node, "sources": [source.name for source in flow.sources], "sinks": [sink.name for sink in flow.sinks]}


@domain.tool()
async def entities_ped_add_source(
    source: Annotated[dict[str, object], Field(description="arena_humansim_msgs/SourceConfig fields")],
    env_id: EnvId = None,
) -> dict[str, object]:
    """Add one humansim flow source; returns the name it got."""
    from arena_humansim_msgs.msg import SourceConfig
    from arena_humansim_msgs.srv import AddSource

    node = await _humansim(env_id)
    response = await ROS.call(AddSource, f"{node}/add_source", AddSource.Request(source=msgs.fill(SourceConfig, source)))
    _done(response, "adding a source")
    return {"env": node, "name": response.name}


@domain.tool(destructive=True)
async def entities_ped_remove_source(name: str, env_id: EnvId = None) -> dict[str, object]:
    """Remove a humansim flow source by name."""
    from arena_humansim_msgs.srv import RemoveSource

    node = await _humansim(env_id)
    _done(await ROS.call(RemoveSource, f"{node}/remove_source", RemoveSource.Request(name=name)), f"removing source {name}")
    return {"env": node, "name": name}


@domain.tool()
async def entities_ped_add_sink(
    sink: Annotated[dict[str, object], Field(description="arena_humansim_msgs/SinkConfig fields")],
    env_id: EnvId = None,
) -> dict[str, object]:
    """Add one humansim flow sink; returns the name it got."""
    from arena_humansim_msgs.msg import SinkConfig
    from arena_humansim_msgs.srv import AddSink

    node = await _humansim(env_id)
    response = await ROS.call(AddSink, f"{node}/add_sink", AddSink.Request(sink=msgs.fill(SinkConfig, sink)))
    _done(response, "adding a sink")
    return {"env": node, "name": response.name}


@domain.tool(destructive=True)
async def entities_ped_remove_sink(name: str, env_id: EnvId = None) -> dict[str, object]:
    """Remove a humansim flow sink by name."""
    from arena_humansim_msgs.srv import RemoveSink

    node = await _humansim(env_id)
    _done(await ROS.call(RemoveSink, f"{node}/remove_sink", RemoveSink.Request(name=name)), f"removing sink {name}")
    return {"env": node, "name": name}


@domain.tool()
async def entities_ped_stimulus(
    stimulus: Annotated[str, Field(description="the need the stimulus drives")],
    intensity: Annotated[float, Field(description="0 to 1")],
    env_id: EnvId = None,
    ped: Annotated[str | int | None, Field(description="humansim agent name or id, None reaches every agent")] = None,
) -> dict[str, object]:
    """Notify humansim agents of a stimulus."""
    from arena_humansim_msgs.srv import NotifyStimulus

    if not 0.0 <= intensity <= 1.0:
        raise ToolError("intensity must be in [0, 1]")
    node = await _humansim(env_id)
    if ped is None:
        agent_id = -1
    elif isinstance(ped, int):
        agent_id = ped
    else:
        meta = await _agents(node)
        ids = dict(zip(meta.name, meta.agent_id, strict=False))
        if ped not in ids:
            raise ToolError(f"no humansim agent named {ped!r}", hint=f"named agents: {sorted(name for name in ids if name)}")
        agent_id = ids[ped]
    request = NotifyStimulus.Request(agent_id=agent_id, stimulus=stimulus, intensity=intensity)
    _done(await ROS.call(NotifyStimulus, f"{node}/notify_stimulus", request), f"notifying {stimulus}")
    return {"env": node, "agent_id": agent_id, "stimulus": stimulus, "intensity": intensity}


@domain.tool(read_only=True)
async def entities_ped_profile(env_id: EnvId = None, reset: Annotated[bool, Field(description="zero the counters after reading")] = False) -> dict[str, object]:
    """Humansim's tick profile: mean and p95 milliseconds per phase, ticks and agents counted."""
    from arena_humansim_msgs.srv import GetProfile

    node = await _humansim(env_id)
    response = await ROS.call(GetProfile, f"{node}/get_profile", GetProfile.Request(reset=reset))
    phases = {name: {"mean_ms": mean, "p95_ms": p95} for name, mean, p95 in zip(response.phase_names, response.phase_means_ms, response.phase_p95s_ms, strict=True)}
    return {"env": node, "phases": phases, "ticks": response.n_ticks, "agents": response.n_agents}
