"""Hung node detection for one env: a tally of missed pings and the round that feeds it."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import attrs
from lifecycle_msgs.msg import State
from lifecycle_msgs.srv import GetState
from rcl_interfaces.srv import GetParameters

if TYPE_CHECKING:
    from arena_rclpy_mixins.Async import AsyncNode, ClientWrapper

_PARAM_SERVICE = "/get_parameters"
_STATE_SERVICE = "/get_state"
_HELPER_SUFFIX = "_rclcpp_node"


@attrs.define
class MissTally:
    """Pings each node has missed in a row, tracked once it has answered at least once."""

    limit: int
    _misses: dict[str, int] = attrs.field(factory=dict, init=False)

    def forget_absent(self, present: set[str]) -> None:
        self._misses = {name: count for name, count in self._misses.items() if name in present}

    def record(self, answered: set[str]) -> str | None:
        """Count one round and return a node that reached the limit, a round most nodes missed counts for nobody."""
        known = self._misses.keys() | answered
        if 2 * len(answered) < len(known):
            return None
        hung = None
        for name in known:
            self._misses[name] = 0 if name in answered else self._misses[name] + 1
            if self._misses[name] >= self.limit:
                hung = name
                self._misses[name] = 0
        return hung


class NodePinger:
    """Pings every node under ``root``: lifecycle nodes must report active, any other node must answer on its parameter service."""

    def __init__(self, node: AsyncNode, root: str, limit: int) -> None:
        self._node = node
        self._root = root
        self._tally = MissTally(limit)
        self._clients: dict[str, ClientWrapper] = {}

    def _targets(self) -> dict[str, str]:
        """Each node under the root and the service to ping it on."""
        services = {name for name, _ in self._node.get_service_names_and_types() if name.startswith(f"{self._root}/")}
        nodes = {name.removesuffix(_PARAM_SERVICE) for name in services if name.endswith(_PARAM_SERVICE)} - {self._node.get_fully_qualified_name()}
        nodes = {name for name in nodes if not name.endswith(_HELPER_SUFFIX)}
        return {name: name + _STATE_SERVICE if name + _STATE_SERVICE in services else name + _PARAM_SERVICE for name in nodes}

    async def _alive(self, service: str, timeout: float) -> bool:
        if service.endswith(_STATE_SERVICE):
            reply = await self._clients[service].call_timeout(GetState.Request(), timeout)
            return reply is not None and reply.current_state.id == State.PRIMARY_STATE_ACTIVE
        return await self._clients[service].call_timeout(GetParameters.Request(), timeout) is not None

    async def round(self, timeout: float, *, new_episode: bool) -> str | None:
        """Ping once and return a node that stopped answering or left its active state, or None."""
        targets = self._targets()
        if new_episode:
            self._tally.forget_absent(set(targets))
        for service in self._clients.keys() - targets.values():
            self._node.destroy_client(self._clients.pop(service).client)
        for service in targets.values() - self._clients.keys():
            self._clients[service] = self._node.create_client_wrapper(GetState if service.endswith(_STATE_SERVICE) else GetParameters, service, timeout=timeout)
        names = sorted(targets)
        alive = await asyncio.gather(*(self._alive(targets[name], timeout) for name in names))
        return self._tally.record({name for name, up in zip(names, alive, strict=True) if up})
