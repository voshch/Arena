import asyncio
import os

import lifecycle_msgs.msg
import lifecycle_msgs.srv
import rclpy.node

from arena_rclpy_mixins.Async import AsyncNode
from arena_rclpy_mixins.Time import TimeNode


class LifecycleClient(TimeNode, rclpy.node.Node):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

    def get_lifecycle_state(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> lifecycle_msgs.msg.State:
        """
        Get state of lifecycle node
        """
        cli = self.create_client(
            lifecycle_msgs.srv.GetState,
            name := os.path.join(node_name, 'get_state'),
            **kwargs,
        )
        try:
            if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
                raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
            return cli.call(lifecycle_msgs.srv.GetState.Request()).current_state
        finally:
            # Per-call client must be freed; wait_for_lifecycle_state loops this every
            # 0.5s, so a leak here grows the executor's waitable set without bound.
            self.destroy_client(cli)

    def get_available_lifecycle_states(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> list[lifecycle_msgs.msg.State]:
        """
        Get available lifecycle states
        """
        cli = self.create_client(
            lifecycle_msgs.srv.GetAvailableStates,
            name := os.path.join(node_name, 'get_available_states'),
            **kwargs,
        )

        try:
            if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
                raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
            return cli.call(lifecycle_msgs.srv.GetAvailableStates.Request()).available_states
        finally:
            self.destroy_client(cli)

    def get_available_lifecycle_transitions(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> list[lifecycle_msgs.msg.Transition]:
        """
        Get available lifecycle transitions
        """
        cli = self.create_client(
            lifecycle_msgs.srv.GetAvailableTransitions,
            name := os.path.join(node_name, 'get_available_transitions'),
            **kwargs,
        )

        try:
            if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
                raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
            return cli.call(lifecycle_msgs.srv.GetAvailableTransitions.Request()).available_transitions
        finally:
            self.destroy_client(cli)

    def change_lifecycle_state(self, node_name: str, transition: lifecycle_msgs.msg.Transition | int, *, timeout: float | None = None, **kwargs: object) -> bool:
        """
        Set state of lifecycle node
        """
        if isinstance(transition, int):
            transition = lifecycle_msgs.msg.Transition(id=transition)
        cli = self.create_client(
            lifecycle_msgs.srv.ChangeState,
            name := os.path.join(node_name, 'change_state'),
            **kwargs,
        )

        try:
            if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
                raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
            return cli.call(lifecycle_msgs.srv.ChangeState.Request(transition=transition)).success
        finally:
            self.destroy_client(cli)

    async def wait_for_lifecycle_state(self, node_name: str, desired_state: lifecycle_msgs.msg.State | int, *, check_interval: float = 0.5, timeout: float | None = None, **kwargs: object) -> bool:
        """
        Asynchronously wait for lifecycle node to reach desired state
        """
        if isinstance(desired_state, int):
            desired_state = lifecycle_msgs.msg.State(id=desired_state)
        start_time = self.wall_time
        while True:
            current_state = self.get_lifecycle_state(node_name, timeout=timeout, **kwargs)
            if current_state.id == desired_state.id:
                return True
            if timeout is not None:
                if (self.wall_time - start_time).to_seconds() >= timeout:
                    return False
            await asyncio.sleep(check_interval)


class AsyncLifecycleClient(AsyncNode):
    async def _call_lifecycle(self, srv_type: type, sub_path: str, node_name: str, request: object, timeout: float | None, **kwargs: object) -> object:
        cli = self.create_client_wrapper(
            srv_type,
            srv_name := os.path.join(node_name, sub_path),
            **kwargs,
        )
        # A fresh client is created per call; it MUST be destroyed on every exit path.
        # Otherwise each call leaks an rclpy client (and its waitables) onto the executor.
        # Under a down/slow lifecycle node the retry loops here fire twice a second, so the
        # leak grows the executor's waitable set without bound and _wait_for_ready_callbacks
        # saturates a core — the task_generator "busy-spin" hang during startup resets.
        try:
            if not await cli.ensure(timeout_sec=timeout):
                raise TimeoutError(f'timed out waiting for {srv_name} after {timeout}s')
            res = await cli.call_timeout(request, timeout_sec=timeout)
            if res is None:
                raise TimeoutError(f'service call {srv_name} timed out after {timeout}s')
            return res
        finally:
            self.destroy_client(cli.client)

    async def get_lifecycle_state_async(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> lifecycle_msgs.msg.State:
        """
        Asynchronously get state of lifecycle node
        """
        res = await self._call_lifecycle(
            lifecycle_msgs.srv.GetState,
            'get_state',
            node_name,
            lifecycle_msgs.srv.GetState.Request(),
            timeout,
            **kwargs,
        )
        return res.current_state

    async def get_available_lifecycle_states_async(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> list[lifecycle_msgs.msg.State]:
        """
        Asynchronously get available lifecycle states
        """
        res = await self._call_lifecycle(
            lifecycle_msgs.srv.GetAvailableStates,
            'get_available_states',
            node_name,
            lifecycle_msgs.srv.GetAvailableStates.Request(),
            timeout,
            **kwargs,
        )
        return res.available_states

    async def get_available_lifecycle_transitions_async(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> list[lifecycle_msgs.msg.Transition]:
        """
        Asynchronously get available lifecycle transitions
        """
        res = await self._call_lifecycle(
            lifecycle_msgs.srv.GetAvailableTransitions,
            'get_available_transitions',
            node_name,
            lifecycle_msgs.srv.GetAvailableTransitions.Request(),
            timeout,
            **kwargs,
        )
        return res.available_transitions

    async def change_lifecycle_state_async(self, node_name: str, transition: lifecycle_msgs.msg.Transition | int, *, timeout: float | None = None, **kwargs: object) -> bool:
        """
        Asynchronously set state of lifecycle node
        """
        if isinstance(transition, int):
            transition = lifecycle_msgs.msg.Transition(id=transition)
        res = await self._call_lifecycle(
            lifecycle_msgs.srv.ChangeState,
            'change_state',
            node_name,
            lifecycle_msgs.srv.ChangeState.Request(transition=transition),
            timeout,
            **kwargs,
        )
        return res.success

    async def wait_for_lifecycle_state_async(self, node_name: str, desired_state: lifecycle_msgs.msg.State | int, *, check_interval: float = 0.5, timeout: float | None = None, **kwargs: object) -> bool:
        """
        Asynchronously wait for lifecycle node to reach desired state
        """

        if isinstance(desired_state, int):
            desired_state = lifecycle_msgs.msg.State(id=desired_state)
        start_time = self.wall_time
        while True:
            current_state = await self.get_lifecycle_state_async(node_name, timeout=timeout, **kwargs)
            if current_state.id == desired_state.id:
                return True
            if timeout is not None:
                if (self.wall_time - start_time).to_seconds() >= timeout:
                    return False
            await asyncio.sleep(check_interval)
