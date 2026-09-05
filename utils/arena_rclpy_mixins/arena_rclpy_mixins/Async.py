from __future__ import annotations

import asyncio
import functools
import inspect
import traceback
import typing
import warnings

import launch
import rclpy.action
import rclpy.action.client
import rclpy.client
import rclpy.node
import rclpy.qos

from arena_rclpy_mixins.Time import TimeNode

T = typing.TypeVar('T')


class LaunchHandle:
    """Handle to a running LaunchService, with graceful teardown.

    Cancelling the run_async task only abandons the run loop (LaunchService
    breaks out without calling _shutdown), leaving child processes orphaned.
    shutdown() emits the Shutdown event so they are terminated properly.
    """

    def __init__(self, service: launch.LaunchService, task: asyncio.Task) -> None:
        self.task = task
        self._service = service
        self._shutdown_started = False

    def __await__(self) -> typing.Generator[typing.Any, None, int]:
        """Await the launch to completion (e.g. until its process exits)."""
        return self.task.__await__()

    async def shutdown(self) -> None:
        if not self._shutdown_started:
            self._shutdown_started = True
            coro = self._service.shutdown()
            if coro is not None:
                await coro
        await asyncio.gather(self.task, return_exceptions=True)


class AsyncLaunchManager:
    def __init__(self):
        self.active_launches: set[LaunchHandle] = set()

    async def launch_description(self, description: launch.LaunchDescription) -> LaunchHandle:
        ls = launch.LaunchService(noninteractive=True)
        ls.include_launch_description(description)
        task = asyncio.create_task(ls.run_async())
        handle = LaunchHandle(ls, task)
        self.active_launches.add(handle)
        task.add_done_callback(lambda _: self.active_launches.discard(handle))
        return handle

    async def kill_all(self):
        if not self.active_launches:
            return
        await asyncio.gather(*(h.shutdown() for h in list(self.active_launches)), return_exceptions=True)


class AsyncNode(TimeNode, rclpy.node.Node):
    """Async utils for rclpy nodes."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.__loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self._launch_manager = AsyncLaunchManager()

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        return self.__loop

    @event_loop.setter
    def event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.__loop = loop

    async def wait_for_service_async(
        self,
        client: rclpy.client.Client,
        timeout: float | None = None,
        interval: float = 0.1,
    ) -> bool:
        start_time = self.wall_time
        while True:
            if client.wait_for_service(timeout_sec=interval):
                return True
            if timeout is not None:
                if (self.wall_time - start_time).to_seconds() >= timeout:
                    return False
            await asyncio.sleep(interval)

    async def do_launch(self, launch_description: launch.LaunchDescription) -> None:
        async def _launcher():
            await self._launch_manager.launch_description(launch_description)

        asyncio.run_coroutine_threadsafe(_launcher(), self.__loop)

    async def do_launch_tracked(self, launch_description: launch.LaunchDescription) -> LaunchHandle:
        """Like do_launch, but returns a handle so the caller can gracefully shut down just
        this launch group (terminating its child processes) on demand."""
        return await self._launch_manager.launch_description(launch_description)

    async def kill_launches(self) -> None:
        await self._launch_manager.kill_all()

    def wait_for(self, future: typing.Awaitable[T]) -> T:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.event_loop:
            warnings.warn(
                "wait_for invoked from the node's own event loop thread; this will block the loop on itself. Await the coroutine directly instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            traceback.print_stack()

        async def coro() -> T:
            return await future

        return asyncio.run_coroutine_threadsafe(coro(), self.event_loop).result()

    def sync_wrap(self, fn: typing.Callable[..., typing.Awaitable[T]]) -> typing.Callable[..., T]:
        @functools.wraps(fn)
        def sync_fn(*args: object, **kwargs: object) -> T:
            return self.wait_for(fn(*args, **kwargs))

        return sync_fn

    def syncify(self, fn: typing.Callable[..., T] | typing.Callable[..., typing.Awaitable[T]]) -> typing.Callable[..., T]:
        if inspect.iscoroutinefunction(fn):
            return self.sync_wrap(fn)

        @functools.wraps(fn)
        def sync_fn(*args: object, **kwargs: object) -> T:
            result = fn(*args, **kwargs)
            if inspect.isawaitable(result):
                return self.wait_for(result)
            return result

        return sync_fn

    async def await_ros(self, ros_future: asyncio.Future[T]) -> T:
        """Wraps a ROS Future into an Asyncio Future so it can be awaited."""
        aio_future = self.__loop.create_future()

        def on_ros_done(fut: asyncio.Future[T]) -> None:
            if aio_future.cancelled():
                return

            try:
                result = fut.result()
                self.__loop.call_soon_threadsafe(aio_future.set_result, result)
            except Exception as e:
                self.__loop.call_soon_threadsafe(aio_future.set_exception, e)

        ros_future.add_done_callback(on_ros_done)

        return await aio_future

    def create_subscription(
        self,
        msg_type: type,
        topic: str,
        callback: typing.Callable[[rclpy.node.MsgType], None] | typing.Callable[[rclpy.node.MsgType], typing.Awaitable[None]],
        qos_profile: rclpy.client.QoSProfile | int,
        *,
        callback_group: rclpy.client.CallbackGroup | None = None,
        event_callbacks: rclpy.node.SubscriptionEventCallbacks | None = None,
        qos_overriding_options: rclpy.node.QoSOverridingOptions | None = None,
        raw: bool = False,
    ) -> rclpy.node.Subscription:
        callback = self.syncify(callback)
        return super().create_subscription(msg_type, topic, callback, qos_profile, callback_group=callback_group, event_callbacks=event_callbacks, qos_overriding_options=qos_overriding_options, raw=raw)

    def create_service(
        self,
        srv_type: type,
        srv_name: str,
        callback: typing.Callable[[rclpy.node.SrvTypeRequest, rclpy.node.SrvTypeResponse], rclpy.node.SrvTypeResponse] | typing.Callable[[rclpy.node.SrvTypeRequest, rclpy.node.SrvTypeResponse], typing.Awaitable[rclpy.node.SrvTypeResponse]],
        *,
        qos_profile: rclpy.client.QoSProfile = rclpy.qos.qos_profile_services_default,
        callback_group: rclpy.client.CallbackGroup | None = None,
    ) -> rclpy.node.Service:
        callback = self.syncify(callback)
        return super().create_service(srv_type, srv_name, callback, qos_profile=qos_profile, callback_group=callback_group)

    def create_client_wrapper(self, srv_type: type, srv_name: str, timeout: float = 60.0, *, qos_profile: rclpy.client.QoSProfile = rclpy.qos.qos_profile_services_default, callback_group: rclpy.client.CallbackGroup | None = None) -> ClientWrapper:
        return ClientWrapper(
            self,
            self.create_client(srv_type, srv_name, qos_profile=qos_profile, callback_group=callback_group),
            timeout=timeout,
        )

    def create_action_client_wrapper(self, action_type: type, action_name: str, timeout: float = 60.0, *, callback_group: rclpy.client.CallbackGroup | None = None) -> ActionClientWrapper:
        return ActionClientWrapper(
            self,
            rclpy.action.ActionClient(self, action_type, action_name, callback_group=callback_group),
            timeout=timeout,
        )


class AsyncUtil:
    @classmethod
    async def timeout(cls, coro: typing.Awaitable[T], timeout_sec: float) -> T | None:
        try:
            return await asyncio.wait_for(coro, timeout=timeout_sec)
        except TimeoutError:
            return None


ServiceT = typing.TypeVar('ServiceT')


class ClientWrapper(typing.Generic[ServiceT]):
    """Async wrapper around rclpy.client.Client."""

    def __init__(self, node: AsyncNode, client: rclpy.client.Client, timeout: float = 60.0):
        self._node: AsyncNode = node
        self._client: rclpy.client.Client = client
        self._timeout: float = timeout

    @property
    def client(self) -> rclpy.client.Client:
        return self._client

    async def call_timeout(
        self,
        request: ServiceT.Request,
        timeout_sec: float | None = None,
    ) -> ServiceT.Response | None:
        if timeout_sec is None:
            timeout_sec = self._timeout
        ros_future = self._client.call_async(request)
        res = await AsyncUtil.timeout(self._node.await_ros(ros_future), timeout_sec=timeout_sec)
        if res is None:
            self._client.remove_pending_request(ros_future)
            self._node.get_logger().warning(f"Service call to {self._client.srv_name} timed out after {timeout_sec} seconds")
        return res

    def call_timeout_sync(
        self,
        request: ServiceT.Request,
        timeout_sec: float | None = None,
    ) -> ServiceT.Response | None:
        if timeout_sec is None:
            timeout_sec = self._timeout

        return self._node.wait_for(
            self.call_timeout(
                request,
                timeout_sec=timeout_sec,
            )
        )

    async def call_forever(
        self,
        request: ServiceT.Request,
        what: str | None = None,
    ) -> ServiceT.Response:
        """Ensure + await the response with no timeout, narrating the wait.
        The caller owns cancellation (e.g. via shutdown)."""
        await self.ensure()
        return await self._node.await_forever(
            self._node.await_ros(self._client.call_async(request)),
            what if what is not None else f"response from {self._client.srv_name}",
        )

    async def ensure(self, timeout_sec: float | None = None) -> bool:
        if timeout_sec is None:
            await self._node.poll(self._client.service_is_ready, f"service {self._client.srv_name}")
            return True
        return await self._node.wait_for_service_async(self._client, timeout=timeout_sec)


ActionT = typing.TypeVar('ActionT')


class ActionClientWrapper(typing.Generic[ActionT]):
    """Async wrapper around rclpy.action.ActionClient."""

    def __init__(self, node: AsyncNode, action_client: rclpy.action.ActionClient, timeout: float = 60.0):
        self._node: AsyncNode = node
        self._client: rclpy.action.ActionClient = action_client
        self._timeout: float = timeout

    @property
    def client(self) -> rclpy.action.ActionClient:
        return self._client

    async def send_goal(
        self,
        goal: ActionT.Goal,
        *,
        feedback_callback: typing.Callable[[typing.Any], None] | None = None,
    ) -> rclpy.action.client.ClientGoalHandle:
        send_future = self._client.send_goal_async(goal, feedback_callback=feedback_callback)
        goal_handle = await self._node.await_ros(send_future)
        if not goal_handle.accepted:
            raise RuntimeError(f"goal was rejected by action server {self._client._action_name!r}")
        return goal_handle

    async def send_goal_timeout(
        self,
        goal: ActionT.Goal,
        timeout_sec: float | None = None,
        *,
        feedback_callback: typing.Callable[[typing.Any], None] | None = None,
    ) -> rclpy.action.client.ClientGoalHandle | None:
        if timeout_sec is None:
            timeout_sec = self._timeout
        res = await AsyncUtil.timeout(
            self.send_goal(goal, feedback_callback=feedback_callback),
            timeout_sec=timeout_sec,
        )
        if res is None:
            self._node.get_logger().warning(f"send_goal to {self._client._action_name} timed out after {timeout_sec} seconds")
        return res

    async def await_result(
        self,
        goal_handle: rclpy.action.client.ClientGoalHandle,
    ) -> object:
        return await self._node.await_ros(goal_handle.get_result_async())

    async def await_result_timeout(
        self,
        goal_handle: rclpy.action.client.ClientGoalHandle,
        timeout_sec: float | None = None,
    ) -> object | None:
        if timeout_sec is None:
            timeout_sec = self._timeout
        res = await AsyncUtil.timeout(
            self.await_result(goal_handle),
            timeout_sec=timeout_sec,
        )
        if res is None:
            self._node.get_logger().warning(f"await_result on {self._client._action_name} timed out after {timeout_sec} seconds")
        return res

    async def send_and_await(
        self,
        goal: ActionT.Goal,
        timeout_sec: float | None = None,
        *,
        feedback_callback: typing.Callable[[typing.Any], None] | None = None,
    ) -> object | None:
        if timeout_sec is None:
            timeout_sec = self._timeout
        goal_handle = await self.send_goal_timeout(goal, timeout_sec=timeout_sec, feedback_callback=feedback_callback)
        if goal_handle is None:
            return None
        return await self.await_result_timeout(goal_handle, timeout_sec=timeout_sec)

    async def cancel(self, goal_handle: rclpy.action.client.ClientGoalHandle) -> object:
        return await self._node.await_ros(goal_handle.cancel_goal_async())

    async def ensure(self, timeout_sec: float | None = None) -> bool:
        if timeout_sec is None:
            await self._node.poll(self._client.server_is_ready, f"action server {self._client._action_name}")
            return True
        start_time = self._node.wall_time
        interval = 0.1
        while not self._client.server_is_ready():
            if (self._node.wall_time - start_time).to_seconds() >= timeout_sec:
                return False
            await asyncio.sleep(interval)
        return True
