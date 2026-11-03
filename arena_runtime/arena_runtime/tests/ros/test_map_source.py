from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import yaml

try:
    from arena_simulation_setup.tree.World import WorldIdentifier
    from rclpy.parameter import Parameter

    from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

WORLD = "hospital_1_bare"


class _Logger:
    def __getattr__(self, _name):
        return lambda *a, **kw: None


async def _noop(*_args, **_kwargs) -> None:
    return None


def _manager() -> tuple[WorldManagerROS, list[bytes]]:
    loads: list[bytes] = []

    async def load(png: bytes, _yaml_text: str) -> None:
        loads.append(png)

    manager = WorldManagerROS.__new__(WorldManagerROS)
    manager._NodeInterface__node = SimpleNamespace(
        conf=SimpleNamespace(Arena=SimpleNamespace(KG_DERIVE=SimpleNamespace(value="off"))),
        get_parameters_by_prefix=lambda _prefix: {"map_source": Parameter("debug.map_source", value="disk")},
        get_logger=lambda: _Logger(),
    )
    manager._map_server_present = False
    manager._map_server_lock = asyncio.Lock()
    manager._cli = SimpleNamespace(ensure=_noop)
    manager._env_log_spec = lambda: ""
    manager.launch_map_server = _noop
    manager.ensure_map_server = _noop
    manager._world_name = WORLD
    manager._mechanism_map_enabled = False
    manager._mechanism_blocked = frozenset()
    manager._load_png_to_server = load
    manager._publish_door_mask = lambda *_args: None
    return manager, loads


def test_first_map_server_bringup_serves_the_disk_map() -> None:
    world_view = WorldIdentifier(WORLD).resolve_sync()
    map_yaml = world_view.map_yaml(world_view.load().level_ids[0])
    png = map_yaml.parent / yaml.safe_load(map_yaml.read_text())["image"]
    manager, loads = _manager()
    asyncio.run(manager.require_map_server())
    assert loads == [png.read_bytes()]
