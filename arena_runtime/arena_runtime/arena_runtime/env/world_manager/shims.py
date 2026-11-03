"""World-manager shim decorators for adapter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arena_runtime.env.adapters import Adapter


def requires_map_server(cls: type[Adapter]) -> type[Adapter]:
    """Class decorator: chains map_server bringup into the adapter's ensure_services."""
    parent_ensure_services = cls.ensure_services

    async def ensure_services(self: Adapter) -> None:
        await parent_ensure_services(self)
        await self.rm.node._world_manager.require_map_server()

    cls.ensure_services = ensure_services
    return cls
