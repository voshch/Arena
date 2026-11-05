from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import os
import threading
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

from .errors import ToolError

if TYPE_CHECKING:
    from arena_rclpy_mixins.Async import ActionClientWrapper, AsyncNode
    from rcl_interfaces.msg import ParameterValue
    from rclpy.action.client import ClientGoalHandle
    from rclpy.parameter import Parameter
    from rclpy.publisher import Publisher
    from rclpy.qos import QoSProfile


class ServiceType[RequestT, ResponseT](Protocol):
    Request: type[RequestT]
    Response: type[ResponseT]


class ActionType[GoalT](Protocol):
    __name__: str
    Goal: type[GoalT]


LATCHED = "latched"
RELIABLE = "reliable"
SENSOR = "sensor"
SETTLE_S = 3.5
PEER_MATCH_S = 0.3
ACK_S = 1.0


def profile(kind: str, depth: int) -> QoSProfile:
    """The arena_rclpy_mixins QoS profile of a qos name: latched, reliable or sensor (best effort)."""
    from arena_rclpy_mixins import qos

    if kind == LATCHED:
        return qos.latched(depth)
    if kind == RELIABLE:
        return qos.reliable(depth)
    if kind == SENSOR:
        return qos.best_effort(depth)
    raise ValueError(f"unknown qos {kind!r}, use latched, reliable or sensor")


def type_name(interface: object) -> str:
    """package/kind/Name of a generated rosidl message, service or action class."""
    cls = cast(type, interface)
    package, kind = cls.__module__.split(".")[:2]
    return f"{package}/{kind}/{cls.__name__}"


class TopicCache:
    """The latest messages on one topic, kept from the first time a tool asks for it."""

    def __init__(self, loop: asyncio.AbstractEventLoop, depth: int) -> None:
        self._loop = loop
        self.history: collections.deque[Any] = collections.deque(maxlen=depth)
        self._waiters: list[asyncio.Future[Any]] = []
        self._lock = threading.Lock()

    def on_message(self, msg: object) -> None:
        with self._lock:
            self.history.append(msg)
            waiters, self._waiters = self._waiters, []
        for waiter in waiters:
            self._loop.call_soon_threadsafe(_resolve, waiter, msg)

    def grow(self, depth: int) -> None:
        with self._lock:
            if self.history.maxlen is not None and depth > self.history.maxlen:
                self.history = collections.deque(self.history, maxlen=depth)

    @property
    def last(self) -> object | None:
        with self._lock:
            return self.history[-1] if self.history else None

    def snapshot(self) -> list[Any]:
        with self._lock:
            return list(self.history)

    def next(self) -> asyncio.Future[Any]:
        future = self._loop.create_future()
        with self._lock:
            self._waiters.append(future)
        return future


def _resolve(future: asyncio.Future[Any], value: object) -> None:
    if not future.done():
        future.set_result(value)


class Ros:
    """The server's one ROS node, created on first use and shared by every domain, settle_discovery holding its first use for one DDS announcement period."""

    def __init__(self) -> None:
        self.settle_discovery = False
        self._node: Any = None
        self._executor: Any = None
        self._thread: threading.Thread | None = None
        self._start = asyncio.Lock()
        self._actions: dict[tuple[str, str], Any] = {}
        self._topics: dict[str, tuple[Any, TopicCache, Any]] = {}
        self._publishers: dict[tuple[str, type, str], Publisher] = {}
        self._shutdown_hooks: list[Callable[[], Awaitable[None]]] = []
        self._closing = False
        self._owns_context = False
        self._crash: Exception | None = None

    async def node(self) -> AsyncNode:
        if self._crash is not None:
            raise ToolError(f"the server's ROS executor stopped: {type(self._crash).__name__}: {self._crash}", hint="restart the MCP server")
        async with self._start:
            if self._node is None:
                try:
                    import rclpy
                    from arena_rclpy_mixins.Async import AsyncNode
                    from arena_rclpy_mixins.spin import create_executor
                    from rclpy.signals import SignalHandlerOptions
                except ImportError as exc:
                    raise ToolError(f"ROS is not available in this environment ({exc})", hint="serve from a sourced Arena container") from exc
                if not rclpy.ok():
                    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
                    self._owns_context = True
                self._closing = False
                self._node = AsyncNode(f"arena_mcp_{os.getpid()}")
                self._executor = create_executor()
                self._executor.add_node(self._node)
                self._thread = threading.Thread(target=self._spin, name="arena_mcp_ros", daemon=True)
                self._thread.start()
                if self.settle_discovery:
                    await asyncio.sleep(SETTLE_S)
            return self._node

    def _spin(self) -> None:
        from rclpy.executors import ExternalShutdownException

        try:
            self._executor.spin()
        except Exception as exc:
            if self._closing:
                return
            if not isinstance(exc, ExternalShutdownException):
                logging.getLogger(__name__).exception("the ROS executor stopped")
            self._crash = exc

    @property
    def started(self) -> bool:
        return self._node is not None

    def on_shutdown(self, hook: Callable[[], Awaitable[None]]) -> None:
        self._shutdown_hooks.append(hook)

    async def shutdown(self) -> None:
        for hook in reversed(self._shutdown_hooks):
            with contextlib.suppress(Exception):
                await asyncio.wait_for(hook(), 5.0)
        if self._node is None:
            return
        import rclpy
        from rclpy.duration import Duration

        if self.settle_discovery:
            for publisher in self._publishers.values():
                with contextlib.suppress(Exception):
                    publisher.wait_for_all_acked(Duration(seconds=ACK_S))
        self._closing = True
        with contextlib.suppress(Exception):
            self._executor.shutdown(timeout_sec=2.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        with contextlib.suppress(Exception):
            self._node.destroy_node()
        with contextlib.suppress(Exception):
            if self._owns_context and rclpy.ok():
                rclpy.shutdown()
        self._owns_context = False
        self._node = None
        self._crash = None
        self._actions.clear()
        self._topics.clear()
        self._publishers.clear()

    async def served(self, name: str, *, timeout: float = 2.0, srv_type: str | None = None) -> bool:
        """Whether another node serves the service name, of srv_type when given, read from the graph without creating a client."""
        node = await self.node()
        own = node.get_fully_qualified_name()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            for node_name, namespace in node.get_node_names_and_namespaces():
                if f"{namespace.rstrip('/')}/{node_name}" == own:
                    continue
                with contextlib.suppress(Exception):
                    if any(service == name and (srv_type is None or srv_type in types) for service, types in node.get_service_names_and_types_by_node(node_name, namespace)):
                        return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.1)

    async def ready(self, srv_type: object, name: str, timeout: float = 2.0) -> bool:
        """Whether a server of srv_type for name is up, without creating a client."""
        return await self.served(name, timeout=timeout, srv_type=type_name(srv_type))

    async def call[RequestT, ResponseT](self, srv_type: ServiceType[RequestT, ResponseT], name: str, request: RequestT | None = None, *, timeout: float = 10.0, ready: float = 2.0, hint: str | None = None) -> ResponseT:
        """Call a service, failing at once when no server answers within ready seconds."""
        if not await self.served(name, timeout=ready):
            raise ToolError(f"service {name} is not available", hint=hint or "is the runtime or env that serves it up? runtime_envs lists live envs")
        node = await self.node()
        wrapper = node.create_client_wrapper(cast(type, srv_type), name)
        try:
            if not wrapper.client.service_is_ready() and not await wrapper.ensure(timeout_sec=max(ready, 2.0)):
                raise ToolError(f"service {name} is not available", hint=hint or "is the runtime or env that serves it up? runtime_envs lists live envs")
            response = await wrapper.call_timeout(request if request is not None else srv_type.Request(), timeout_sec=timeout)
        finally:
            node.destroy_client(wrapper.client)
        if response is None:
            raise ToolError(f"service {name} did not answer within {timeout:.0f}s")
        return cast(ResponseT, response)

    async def action[GoalT](self, action_type: ActionType[GoalT], name: str, *, ready: float = 2.0, hint: str | None = None) -> ActionClientWrapper:
        """An action client for name, created only once a server is in the graph and dropped when the server is gone."""
        node = await self.node()
        key = (name, action_type.__name__)
        wrapper = self._actions.get(key)
        if wrapper is not None and not wrapper.client.server_is_ready():
            self._actions.pop(key)
            with contextlib.suppress(Exception):
                wrapper.client.destroy()
            wrapper = None
        if wrapper is None:
            if not await self.served(f"{name}/_action/send_goal", timeout=ready):
                raise ToolError(f"action server {name} is not available", hint=hint)
            wrapper = node.create_action_client_wrapper(cast(type, action_type), name)
            if not wrapper.client.server_is_ready() and not await wrapper.ensure(timeout_sec=max(ready, 2.0)):
                with contextlib.suppress(Exception):
                    wrapper.client.destroy()
                raise ToolError(f"action server {name} is not available", hint=hint)
            self._actions[key] = wrapper
        return wrapper

    async def send_goal[GoalT](self, action_type: ActionType[GoalT], name: str, goal: GoalT, *, feedback: Callable[[object], None] | None = None, timeout: float = 10.0, hint: str | None = None) -> tuple[ActionClientWrapper, ClientGoalHandle]:
        """Send a goal and return (wrapper, goal handle) once accepted."""
        wrapper = await self.action(action_type, name, hint=hint)
        try:
            handle = await asyncio.wait_for(wrapper.send_goal(goal, feedback_callback=feedback), timeout)
        except TimeoutError as exc:
            raise ToolError(f"action server {name} did not accept the goal within {timeout:.0f}s") from exc
        except RuntimeError as exc:
            raise ToolError(str(exc), hint=hint) from exc
        return wrapper, handle

    async def cache[MsgT](self, msg_type: type[MsgT], topic: str, *, qos: str = LATCHED, depth: int = 1) -> TopicCache:
        node = await self.node()
        entry = self._topics.get(topic)
        if entry is None:
            cache = TopicCache(asyncio.get_running_loop(), depth)
            subscription = node.create_subscription(msg_type, topic, cache.on_message, profile(qos, max(depth, 1)))
            entry = (msg_type, cache, subscription)
            self._topics[topic] = entry
        elif entry[0] is not msg_type:
            raise ToolError(f"topic {topic} is already watched as {entry[0].__name__}, not {msg_type.__name__}")
        else:
            entry[1].grow(depth)
        return entry[1]

    async def latest[MsgT](self, msg_type: type[MsgT], topic: str, *, qos: str = LATCHED, timeout: float = 2.0, depth: int = 1, hint: str | None = None) -> MsgT:
        """The last message on topic, waiting up to timeout for the first one."""
        cache = await self.cache(msg_type, topic, qos=qos, depth=depth)
        if cache.last is not None:
            return cast(MsgT, cache.last)
        try:
            return cast(MsgT, await asyncio.wait_for(cache.next(), timeout))
        except TimeoutError as exc:
            raise ToolError(f"nothing received on {topic} within {timeout:.0f}s", hint=hint or "is the node that publishes it up?") from exc

    async def next[MsgT](self, msg_type: type[MsgT], topic: str, *, qos: str = SENSOR, timeout: float = 5.0) -> MsgT:
        """The next message published on topic after this call."""
        cache = await self.cache(msg_type, topic, qos=qos)
        try:
            return cast(MsgT, await asyncio.wait_for(cache.next(), timeout))
        except TimeoutError as exc:
            raise ToolError(f"no new message on {topic} within {timeout:.0f}s") from exc

    async def collect[MsgT](self, msg_type: type[MsgT], topic: str, *, qos: str, count: int, timeout: float) -> list[MsgT]:
        """Up to count messages received on topic within timeout, through a subscription that lives only for the call."""
        node = await self.node()
        loop = asyncio.get_running_loop()
        cache = TopicCache(loop, count)
        subscription = node.create_subscription(msg_type, topic, cache.on_message, profile(qos, max(count, 1)))
        deadline = loop.time() + timeout
        try:
            while len(cache.snapshot()) < count and loop.time() < deadline:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(cache.next(), deadline - loop.time())
        finally:
            node.destroy_subscription(subscription)
        return cast(list[MsgT], cache.snapshot())

    async def publisher[MsgT](self, msg_type: type[MsgT], topic: str, *, qos: str = RELIABLE) -> Publisher:
        """The server's publisher of msg_type on topic with this qos, created on first use."""
        node = await self.node()
        key = (topic, msg_type, qos)
        publisher = self._publishers.get(key)
        if publisher is None:
            publisher = node.create_publisher(msg_type, topic, profile(qos, 10))
            self._publishers[key] = publisher
        return publisher

    async def matched(self, publisher: Publisher, topic: str, timeout: float = 1.0) -> None:
        """Wait up to timeout for the subscribers already in the graph to match publisher."""
        node = await self.node()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout, 0.0)
        while publisher.get_subscription_count() < node.count_subscribers(topic) and loop.time() < deadline:
            await asyncio.sleep(0.05)
        if self.settle_discovery:
            await asyncio.sleep(PEER_MATCH_S)

    async def publish[MsgT](self, msg_type: type[MsgT], topic: str, msg: MsgT, *, qos: str = RELIABLE) -> None:
        fresh = (topic, msg_type, qos) not in self._publishers
        publisher = await self.publisher(msg_type, topic, qos=qos)
        if fresh:
            await self.matched(publisher, topic)
        publisher.publish(msg)

    async def graph(self) -> dict[str, list[str]]:
        """Live node, service, action and topic names."""
        node = await self.node()
        nodes = sorted(f"{ns.rstrip('/')}/{name}" for name, ns in node.get_node_names_and_namespaces())
        services = sorted(name for name, _ in node.get_service_names_and_types())
        topics = sorted(name for name, _ in node.get_topic_names_and_types())
        actions = sorted({name[: -len("/_action/status")] for name in topics if name.endswith("/_action/status")})
        return {"nodes": nodes, "services": services, "actions": actions, "topics": topics}

    async def sim_time(self) -> float:
        node = await self.node()
        return node.sim_time.to_seconds()

    async def get_params(self, node_name: str, names: list[str] | None = None) -> dict[str, Any]:
        from rcl_interfaces.srv import GetParameters, ListParameters
        from rclpy.parameter import parameter_value_to_python

        if names is None:
            listed = await self.call(ListParameters, f"{node_name}/list_parameters", hint=f"node {node_name} is not up")
            names = sorted(listed.result.names)
        response = await self.call(GetParameters, f"{node_name}/get_parameters", GetParameters.Request(names=names), hint=f"node {node_name} is not up")
        if len(response.values) != len(names):
            await self._undeclared(node_name, names)
        return {name: parameter_value_to_python(value) for name, value in zip(names, response.values, strict=True)}

    async def _undeclared(self, node_name: str, names: list[str]) -> None:
        from rcl_interfaces.srv import ListParameters

        listed = await self.call(ListParameters, f"{node_name}/list_parameters", hint=f"node {node_name} is not up")
        missing = sorted(set(names) - set(listed.result.names))
        raise ToolError(f"{node_name} does not declare {', '.join(missing) or 'some of ' + ', '.join(names)}", hint=f"ros_params {node_name} lists what it declares")

    async def set_params(self, node_name: str, values: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Set parameters, coercing each value to the type the node declared."""
        from rcl_interfaces.msg import Parameter
        from rcl_interfaces.srv import DescribeParameters, SetParameters

        names = list(values)
        described = await self.call(DescribeParameters, f"{node_name}/describe_parameters", DescribeParameters.Request(names=names), hint=f"node {node_name} is not up")
        if len(described.descriptors) != len(names):
            await self._undeclared(node_name, names)
        params = []
        for name, descriptor in zip(names, described.descriptors, strict=True):
            params.append(Parameter(name=name, value=python_to_param(values[name], descriptor.type)))
        response = await self.call(SetParameters, f"{node_name}/set_parameters", SetParameters.Request(parameters=params), hint=f"node {node_name} is not up")
        return {name: {"ok": result.successful, "reason": result.reason} for name, result in zip(names, response.results, strict=True)}


def python_to_param(value: object, declared: int) -> ParameterValue:
    """A ParameterValue of the declared type, or of the value's own type when the parameter is untyped."""
    from rcl_interfaces.msg import ParameterType
    from rclpy.parameter import Parameter

    try:
        kind = Parameter.Type.from_parameter_value(value) if declared == ParameterType.PARAMETER_NOT_SET else Parameter.Type(declared)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"cannot send {value!r} as a ROS parameter") from exc
    try:
        return Parameter("value", kind, _coerce(value, kind)).get_parameter_value()
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{value!r} is not a valid {kind.name.lower()} value") from exc


def _coerce(value: object, kind: Parameter.Type) -> bool | int | float | str | list[Any]:
    from rclpy.parameter import Parameter

    scalars: dict[Parameter.Type, Callable[[Any], bool | int | float | str]] = {
        Parameter.Type.BOOL: _bool,
        Parameter.Type.INTEGER: _int,
        Parameter.Type.DOUBLE: float,
        Parameter.Type.STRING: str,
    }
    arrays: dict[Parameter.Type, Callable[[Any], Any]] = {
        Parameter.Type.BYTE_ARRAY: lambda item: bytes([int(item)]),
        Parameter.Type.BOOL_ARRAY: bool,
        Parameter.Type.INTEGER_ARRAY: int,
        Parameter.Type.DOUBLE_ARRAY: float,
        Parameter.Type.STRING_ARRAY: str,
    }
    if kind in scalars:
        return scalars[kind](_scalar(value))
    if kind in arrays:
        return [arrays[kind](_scalar(item)) for item in _items(value)]
    raise TypeError(f"unsupported parameter type {kind.name}")


def _bool(scalar: str | float) -> bool:
    if isinstance(scalar, str):
        lowered = scalar.strip().lower()
        if lowered not in ("true", "false"):
            raise ValueError(scalar)
        return lowered == "true"
    return bool(scalar)


def _int(scalar: str | float) -> int:
    if isinstance(scalar, float) and not scalar.is_integer():
        raise ValueError(scalar)
    return int(scalar)


def _scalar(value: object) -> bool | int | float | str:
    if isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"{type(value).__name__} is not a scalar")


def _items(value: object) -> list[object]:
    if isinstance(value, list | tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not a list")


ROS = Ros()
