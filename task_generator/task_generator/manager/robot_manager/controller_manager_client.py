from __future__ import annotations

import asyncio
import typing

import controller_manager_msgs.srv

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from arena_rclpy_mixins.Async import AsyncNode, ClientWrapper


class ControllerManagerClient:
    """Async client for one robot's controller_manager services."""

    def __init__(self, node: AsyncNode, cm_namespace: str, *, call_timeout: float) -> None:
        self._list: ClientWrapper = node.create_client_wrapper(
            controller_manager_msgs.srv.ListControllers,
            f"{cm_namespace}/list_controllers",
            timeout=call_timeout,
        )
        self._load: ClientWrapper = node.create_client_wrapper(
            controller_manager_msgs.srv.LoadController,
            f"{cm_namespace}/load_controller",
            timeout=call_timeout,
        )
        self._configure: ClientWrapper = node.create_client_wrapper(
            controller_manager_msgs.srv.ConfigureController,
            f"{cm_namespace}/configure_controller",
            timeout=call_timeout,
        )
        self._switch: ClientWrapper = node.create_client_wrapper(
            controller_manager_msgs.srv.SwitchController,
            f"{cm_namespace}/switch_controller",
            timeout=call_timeout,
        )
        self._node = node

    async def ensure(self, timeout_sec: float | None = None) -> bool:
        ready = await asyncio.gather(*(c.ensure(timeout_sec=timeout_sec) for c in (self._list, self._load, self._configure, self._switch)))
        return all(ready)

    async def states(self) -> dict[str, str] | None:
        """name -> lifecycle label, None on no reply."""
        resp = await self._list.call_timeout(controller_manager_msgs.srv.ListControllers.Request())
        if resp is None:
            return None
        return {c.name: c.state for c in resp.controller}

    async def load(self, name: str) -> bool | None:
        resp = await self._load.call_timeout(controller_manager_msgs.srv.LoadController.Request(name=name))
        return None if resp is None else bool(resp.ok)

    async def configure(self, name: str) -> bool | None:
        resp = await self._configure.call_timeout(controller_manager_msgs.srv.ConfigureController.Request(name=name))
        return None if resp is None else bool(resp.ok)

    async def activate(self, names: Sequence[str], *, switch_timeout_sec: float, timeout_sec: float) -> tuple[bool, str] | None:
        """One STRICT grouped switch, client wait never shorter than the CM-side timeout."""
        req = controller_manager_msgs.srv.SwitchController.Request()
        req.activate_controllers = list(names)
        req.strictness = controller_manager_msgs.srv.SwitchController.Request.STRICT
        req.activate_asap = True
        req.timeout.sec = int(switch_timeout_sec)
        req.timeout.nanosec = int((switch_timeout_sec - int(switch_timeout_sec)) * 1e9)
        resp = await self._switch.call_timeout(req, timeout_sec=max(timeout_sec, switch_timeout_sec + 1.0))
        return None if resp is None else (bool(resp.ok), str(resp.message))

    def close(self) -> None:
        for wrapper in (self._list, self._load, self._configure, self._switch):
            self._node.destroy_client(wrapper.client)
