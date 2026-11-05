from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

from pydantic import Field

from arena_mcp.core import JOBS, LATCHED, RELIABLE, ROS, SENSOR, Domain, JobHandle, ToolError, media, msgs

if TYPE_CHECKING:
    from nav_msgs.msg import OccupancyGrid
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage, Image

    from arena_mcp.core.msgs import RosMessage
    from arena_mcp.core.ros import ActionType, ServiceType

    class FeedbackMessage(Protocol):
        feedback: object


domain = Domain(
    "ros",
    """Generic ROS 2 access for what no Arena domain covers: the graph with types, interface definitions, any service call, topic echo and publish, parameters, any action as a job.
    Types are package/kind/Name or package/Name. Prefer the Arena domains where one covers the task, they resolve envs and validate input.""",
    requires=("rclpy", "rosidl_runtime_py"),
)

QOS = (SENSOR, RELIABLE, LATCHED)
KINDS = ("msg", "srv", "action")
IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage", "nav_msgs/msg/OccupancyGrid")


def resolve(type_name: str, kind: str | None = None) -> tuple[str, str]:
    """The full package/kind/Name of an interface and its kind, inferring the kind from the package's interfaces when type_name is package/Name."""
    from rosidl_runtime_py import get_interfaces

    parts = type_name.strip("/").split("/")
    if len(parts) not in (2, 3) or not all(parts) or (len(parts) == 3 and parts[1] not in KINDS):
        raise ToolError(f"{type_name!r} is not an interface name", hint="use package/msg/Name, package/srv/Name, package/action/Name or package/Name")
    package, name = parts[0], parts[-1]
    try:
        listed = get_interfaces([package])[package]
    except (LookupError, ValueError) as exc:
        raise ToolError(f"no interface package {package!r} is installed", hint="ros_graph shows the types in use") from exc
    wanted_kind = parts[1] if len(parts) == 3 else kind
    if kind is not None and wanted_kind != kind:
        raise ToolError(f"{type_name} is a {wanted_kind}, a {kind} is needed here")
    matches = [entry for entry in listed if entry.rsplit("/", 1)[-1] == name and (wanted_kind is None or entry.split("/", 1)[0] == wanted_kind)]
    if len(matches) != 1:
        options = [f"{package}/{entry}" for entry in listed if wanted_kind is None or entry.startswith(f"{wanted_kind}/")]
        raise ToolError(f"{'ambiguous' if matches else 'no'} interface {type_name!r}", hint=f"{package} has: {', '.join(options[:80])}")
    return f"{package}/{matches[0]}", matches[0].split("/", 1)[0]


def load(full: str, kind: str) -> type:
    """The Python class of a resolved interface."""
    from rosidl_runtime_py.utilities import get_action, get_message, get_service

    return {"msg": get_message, "srv": get_service, "action": get_action}[kind](full)


def _fields(cls: type, kind: str) -> dict[str, Any]:
    parts = {"msg": {}, "srv": {"request": "Request", "response": "Response"}, "action": {"goal": "Goal", "result": "Result", "feedback": "Feedback"}}[kind]
    if not parts:
        return dict(cls.get_fields_and_field_types())
    return {key: dict(getattr(cls, attr).get_fields_and_field_types()) for key, attr in parts.items()}


@domain.tool(read_only=True)
def ros_interface(type_name: Annotated[str, Field(description="package/msg/Name, package/srv/Name, package/action/Name or package/Name")]) -> dict[str, Any]:
    """The definition text and field types of a message, service or action, what `ros2 interface show` prints."""
    from rosidl_runtime_py import get_interface_path

    full, kind = resolve(type_name)
    path = get_interface_path(full)
    return {"type": full, "kind": kind, "path": path, "definition": Path(path).read_text(), "fields": _fields(load(full, kind), kind)}


async def _settled() -> Node:
    node = await ROS.node()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 2.0
    last = -1
    while loop.time() < deadline:
        count = len(node.get_node_names_and_namespaces())
        if count == last and count > 1:
            break
        last = count
        await asyncio.sleep(0.3)
    return node


def _hidden(name: str) -> bool:
    return any(part.startswith("_") for part in name.split("/") if part)


async def graph_types(prefix: str | None = None, hidden: bool = False) -> dict[str, Any]:
    """Nodes plus topics, services and actions mapped to their types."""
    from rclpy.action.graph import get_action_names_and_types

    node = await _settled()

    def keep(name: str) -> bool:
        return (prefix is None or name.startswith(prefix)) and (hidden or not _hidden(name))

    nodes = sorted(f"{namespace.rstrip('/')}/{name}" for name, namespace in node.get_node_names_and_namespaces())
    return {
        "nodes": [name for name in nodes if keep(name)],
        "topics": {name: types for name, types in sorted(node.get_topic_names_and_types()) if keep(name)},
        "services": {name: types for name, types in sorted(node.get_service_names_and_types()) if keep(name)},
        "actions": {name: types for name, types in sorted(get_action_names_and_types(node)) if keep(name)},
    }


@domain.tool(read_only=True)
async def ros_graph(
    prefix: Annotated[str | None, Field(description="only names starting with this, e.g. /arena/env_0")] = None,
    hidden: Annotated[bool, Field(description="include hidden names (a segment starting with _), e.g. action internals")] = False,
) -> dict[str, Any]:
    """Live nodes, and topics, services and actions with their types."""
    return await graph_types(prefix, hidden)


async def _graph_type(section: str, name: str) -> str:
    types = (await graph_types(hidden=True))[section].get(name)
    if not types:
        raise ToolError(f"no {section[:-1]} {name} in the ROS graph", hint=f"pass type_name, or check ros_graph for the {section} that exist")
    if len(types) > 1:
        raise ToolError(f"{name} carries several types: {', '.join(types)}", hint="pass type_name")
    return types[0]


async def _typed(section: str, kind: str, name: str, type_name: str | None) -> tuple[str, type]:
    full, _ = resolve(type_name or await _graph_type(section, name), kind)
    return full, load(full, kind)


@domain.tool(destructive=True)
async def ros_call(
    service: str,
    type_name: Annotated[str | None, Field(description="the service type, taken from the graph when None")] = None,
    request: Annotated[dict[str, Any] | None, Field(description="request fields, nested messages as nested objects")] = None,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Call any service and return every field of its response."""
    full, srv_cls = await _typed("services", "srv", service, type_name)
    srv = cast("ServiceType[RosMessage, object]", srv_cls)
    response = await ROS.call(srv, service, msgs.fill(srv.Request, request), timeout=timeout_s, hint="ros_graph lists the services that are up")
    return {"service": service, "type": full, "response": msgs.to_dict(response)}


def clip(value: object, limit: int, path: str, clipped: list[str]) -> object:
    """value with every list longer than limit cut to limit items, the cut paths appended to clipped."""
    if isinstance(value, dict):
        return {key: clip(item, limit, f"{path}.{key}" if path else key, clipped) for key, item in value.items()}
    if isinstance(value, list):
        if len(value) > limit:
            clipped.append(f"{path} ({len(value)} items)")
            value = value[:limit]
        return [clip(item, limit, f"{path}[]", clipped) for item in value]
    return value


def _image(full: str, msg: object) -> object:
    if full == "sensor_msgs/msg/Image":
        return media.image_msg(cast("Image", msg))
    if full == "sensor_msgs/msg/CompressedImage":
        return media.compressed_msg(cast("CompressedImage", msg))
    return media.occupancy_png(cast("OccupancyGrid", msg))


@domain.tool(read_only=True)
async def ros_echo(
    topic: str,
    type_name: Annotated[str | None, Field(description="the message type, taken from the graph when None")] = None,
    count: Annotated[int, Field(description="messages to return, at most 50")] = 1,
    timeout_s: Annotated[float, Field(description="seconds to wait for them, at most 60")] = 5.0,
    qos: Annotated[str, Field(description="sensor (best effort, hears every publisher), reliable, or latched (transient local, for state topics)")] = SENSOR,
    max_items: Annotated[int, Field(description="lists longer than this are cut, the cut paths are listed under clipped")] = 100,
) -> object:
    """The next count messages on a topic with every field, images, compressed images and occupancy grids also as image content."""
    if not 1 <= count <= 50:
        raise ToolError("count must be in [1, 50]")
    if not 0 < timeout_s <= 60:
        raise ToolError("timeout_s must be in (0, 60]")
    if qos not in QOS:
        raise ToolError(f"unknown qos {qos!r}", hint=f"qos: {', '.join(QOS)}")
    full, msg_type = await _typed("topics", "msg", topic, type_name)
    received = await ROS.collect(msg_type, topic, qos=qos, count=count, timeout=timeout_s)
    if not received:
        raise ToolError(f"nothing received on {topic} within {timeout_s:.0f}s", hint="ros_graph shows whether it has publishers, a state topic needs qos='latched'")
    clipped: list[str] = []
    images = [_image(full, msg) for msg in received] if full in IMAGE_TYPES else []
    messages = []
    for msg in received:
        data = msgs.to_dict(msg)
        if images:
            data.pop("data", None)
        messages.append(clip(data, max_items, "", clipped))
    envelope = {"topic": topic, "type": full, "qos": qos, "messages": messages}
    if clipped:
        envelope["clipped"] = sorted(set(clipped))
    return [envelope, *images] if images else envelope


@domain.tool(destructive=True)
async def ros_publish(
    topic: str,
    type_name: str,
    message: Annotated[dict[str, Any], Field(description="message fields, nested messages as nested objects")],
    qos: Annotated[str, Field(description="reliable, sensor (best effort) or latched (transient local, kept for late subscribers)")] = RELIABLE,
    match_s: Annotated[float, Field(description="seconds to wait for the discovered subscribers to match before publishing")] = 1.0,
) -> dict[str, Any]:
    """Publish one message on a topic, after the subscribers already in the graph have matched."""
    if qos not in QOS:
        raise ToolError(f"unknown qos {qos!r}", hint=f"qos: {', '.join(QOS)}")
    full, _ = resolve(type_name, "msg")
    msg_type = load(full, "msg")
    msg = msgs.fill(msg_type, message)
    publisher = await ROS.publisher(msg_type, topic, qos=qos)
    await ROS.matched(publisher, topic, match_s)
    publisher.publish(msg)
    return {"topic": topic, "type": full, "qos": qos, "subscribers": publisher.get_subscription_count()}


def _node(name: str) -> str:
    return "/" + name.strip("/")


@domain.tool(read_only=True)
async def ros_params(node: str, names: Annotated[list[str] | None, Field(description="parameter names, every declared one when None")] = None) -> dict[str, Any]:
    """Parameter values of a node."""
    fqn = _node(node)
    return {"node": fqn, "params": await ROS.get_params(fqn, names)}


@domain.tool(destructive=True)
async def ros_set_params(node: str, values: Annotated[dict[str, Any], Field(description="{name: value}, each coerced to the type the node declared")]) -> dict[str, Any]:
    """Set parameters of a node, returning per parameter whether the node accepted it."""
    if not values:
        raise ToolError("values is empty")
    fqn = _node(node)
    return {"node": fqn, "results": await ROS.set_params(fqn, values)}


@domain.tool(destructive=True)
async def ros_action(
    action: str,
    goal: Annotated[dict[str, Any] | None, Field(description="goal fields, nested messages as nested objects")] = None,
    type_name: Annotated[str | None, Field(description="the action type, taken from the graph when None")] = None,
) -> dict[str, Any]:
    """Send a goal to any action server as a job: feedback lands in the job's progress, jobs_cancel cancels the goal, the result is the job result."""
    from action_msgs.msg import GoalStatus

    full, action_cls = await _typed("actions", "action", action, type_name)
    action_type = cast("ActionType[RosMessage]", action_cls)
    goal_msg = msgs.fill(action_type.Goal, goal)
    loop = asyncio.get_running_loop()
    seen: dict[str, Any] = {"count": 0}
    handles: list[JobHandle] = []

    def record(feedback: object) -> None:
        seen["count"] += 1
        seen["last"] = msgs.to_dict(cast("FeedbackMessage", feedback).feedback)
        for handle in handles:
            handle.progress(feedback=seen["last"], feedback_count=seen["count"])

    def on_feedback(feedback: object) -> None:
        loop.call_soon_threadsafe(record, feedback)

    wrapper, goal_handle = await ROS.send_goal(action_type, action, goal_msg, feedback=on_feedback, hint="ros_graph lists the action servers that are up")
    names = msgs.constants(GoalStatus, "STATUS_")

    async def body(handle: JobHandle) -> dict[str, Any]:
        handles.append(handle)
        if seen["count"]:
            handle.progress(feedback=seen["last"], feedback_count=seen["count"])
        response = await wrapper.await_result(goal_handle)
        status = names.get(response.status, str(response.status))
        result = msgs.to_dict(response.result)
        if status != "succeeded":
            raise ToolError(f"goal {status}: {result}")
        return {"status": status, "result": result}

    async def cancel() -> None:
        await wrapper.cancel(goal_handle)

    job = JOBS.start_task("action", f"{action} {full}", body, on_cancel=cancel)
    return {**job.as_dict(), "type": full}
