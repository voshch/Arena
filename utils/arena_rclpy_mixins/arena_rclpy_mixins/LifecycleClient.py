import asyncio
import os

import lifecycle_msgs.msg
import lifecycle_msgs.srv
import rclpy.node

from arena_rclpy_mixins.Async import AsyncNode, ClientWrapper
from arena_rclpy_mixins.Time import TimeNode


class LifecycleClient(TimeNode, rclpy.node.Node):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

    def get_lifecycle_state(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> lifecycle_msgs.msg.State:
        cli = self.create_client(
            lifecycle_msgs.srv.GetState,
            name := os.path.join(node_name, 'get_state'),
            **kwargs,
        )
        if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
            raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
        return cli.call(lifecycle_msgs.srv.GetState.Request()).current_state

    def get_available_lifecycle_states(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> list[lifecycle_msgs.msg.State]:
        cli = self.create_client(
            lifecycle_msgs.srv.GetAvailableStates,
            name := os.path.join(node_name, 'get_available_states'),
            **kwargs,
        )

        if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
            raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
        return cli.call(lifecycle_msgs.srv.GetAvailableStates.Request()).available_states

    def get_available_lifecycle_transitions(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> list[lifecycle_msgs.msg.Transition]:
        cli = self.create_client(
            lifecycle_msgs.srv.GetAvailableTransitions,
            name := os.path.join(node_name, 'get_available_transitions'),
            **kwargs,
        )

        if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
            raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
        return cli.call(lifecycle_msgs.srv.GetAvailableTransitions.Request()).available_transitions

    def change_lifecycle_state(self, node_name: str, transition: lifecycle_msgs.msg.Transition | int, *, timeout: float | None = None, **kwargs: object) -> bool:
        if isinstance(transition, int):
            transition = lifecycle_msgs.msg.Transition(id=transition)
        cli = self.create_client(
            lifecycle_msgs.srv.ChangeState,
            name := os.path.join(node_name, 'change_state'),
            **kwargs,
        )

        if not cli.wait_for_service(timeout_sec=timeout):  # type: ignore
            raise RuntimeError(f'timed out waiting for {name} after {timeout} secs')
        return cli.call(lifecycle_msgs.srv.ChangeState.Request(transition=transition)).success

    async def wait_for_lifecycle_state(self, node_name: str, desired_state: lifecycle_msgs.msg.State | int, *, check_interval: float = 0.5, timeout: float | None = None, **kwargs: object) -> bool:
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
    async def _call_wrapper(self, cli: ClientWrapper, request: object, timeout: float | None) -> object:
        srv_name = cli.client.srv_name
        if timeout is None:
            return await cli.call_forever(request)
        if not await cli.ensure(timeout_sec=timeout):
            raise TimeoutError(f'timed out waiting for {srv_name} after {timeout}s')
        res = await cli.call_timeout(request, timeout_sec=timeout)
        if res is None:
            raise TimeoutError(f'service call {srv_name} timed out after {timeout}s')
        return res

    async def _call_lifecycle(self, srv_type: type, sub_path: str, node_name: str, request: object, timeout: float | None, **kwargs: object) -> object:
        cli = self.create_client_wrapper(
            srv_type,
            os.path.join(node_name, sub_path),
            **kwargs,
        )
        try:
            return await self._call_wrapper(cli, request, timeout)
        finally:
            self.destroy_client(cli.client)

    async def get_lifecycle_state_async(self, node_name: str, *, timeout: float | None = None, **kwargs: object) -> lifecycle_msgs.msg.State:
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
        if isinstance(desired_state, int):
            desired_state = lifecycle_msgs.msg.State(id=desired_state)
        cli = self.create_client_wrapper(
            lifecycle_msgs.srv.GetState,
            os.path.join(node_name, 'get_state'),
            **kwargs,
        )
        start_time = self.wall_time
        try:
            while True:
                try:
                    res = await self._call_wrapper(cli, lifecycle_msgs.srv.GetState.Request(), timeout)
                except TimeoutError:
                    pass
                else:
                    if res.current_state.id == desired_state.id:
                        return True
                if timeout is not None:
                    if (self.wall_time - start_time).to_seconds() >= timeout:
                        return False
                await asyncio.sleep(check_interval)
        finally:
            self.destroy_client(cli.client)
