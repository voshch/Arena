"""Opt-in mechanism map: blocked-door logic, PNG painting, WorldManagerROS push flow (no map_server)."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest

try:
    import PIL.Image
    from arena_runtime.sim._semantics import SemanticEntitySnapshot
    from arena_simulation_setup.shared import Door
    from arena_simulation_setup.tree.World.Map import Map
    from arena_simulation_setup.tree.World import Level, World, Zone
    from arena_simulation_setup.utils.geometry import Position as GeoPosition

    from arena_runtime.env.world_manager import mechanism_map
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")


def _door(name: str, *, activation: float = 3.0, gate: dict | None = None, start=(4.0, 1.5), end=(4.0, 2.5)) -> Door:
    semantics = [{"preset": "gate", "params": gate}] if gate is not None else []
    return Door(name=name, start=GeoPosition(*start, 0.0), end=GeoPosition(*end, 0.0), activation_distance=activation, semantics=semantics)


def _snap(entity: str, kind: str, *, state: str | None = None, locked: bool | None = None) -> SemanticEntitySnapshot:
    discrete = {} if state is None else {"state": state}
    predicates = {} if locked is None else {"locked": locked}
    return SemanticEntitySnapshot(entity, kind, discrete, {}, predicates)


# blocked_doors


def test_plain_door_never_blocked():
    assert mechanism_map.blocked_doors([("d", _door("d"))], [_snap("d", "door", state="closed")], ["jackal"]) == frozenset()


def test_zero_activation_blocked_until_held_open():
    doors = [("d", _door("d", activation=0.0))]
    assert mechanism_map.blocked_doors(doors, [_snap("d", "door", state="closed")], ["jackal"]) == {"d"}
    assert mechanism_map.blocked_doors(doors, [], ["jackal"]) == {"d"}
    assert mechanism_map.blocked_doors(doors, [_snap("d", "door", state="open")], ["jackal"]) == frozenset()
    assert mechanism_map.blocked_doors(doors, [_snap("d", "door", state="opening")], ["jackal"]) == frozenset()


def test_locked_gate_blocks_unauthorized_robot_only():
    doors = [("env0/g", _door("g", gate={"authorized": ["jackal"], "locked": False}))]
    locked = [_snap("env0/g", "door", state="closed"), _snap("env0/g", "gate", locked=True)]
    assert mechanism_map.blocked_doors(doors, locked, ["jackal"]) == frozenset()
    assert mechanism_map.blocked_doors(doors, locked, ["env0/jackal"]) == frozenset()
    assert mechanism_map.blocked_doors(doors, locked, ["burger", "env0/burger"]) == {"env0/g"}
    unlocked = [_snap("env0/g", "door", state="closed"), _snap("env0/g", "gate", locked=False)]
    assert mechanism_map.blocked_doors(doors, unlocked, ["burger"]) == frozenset()


def test_gate_falls_back_to_authored_initial_lock():
    doors = [("g", _door("g", gate={"authorized": [], "locked": True}))]
    assert mechanism_map.blocked_doors(doors, [], ["jackal"]) == {"g"}
    doors = [("g", _door("g", gate={"authorized": [], "locked": False}))]
    assert mechanism_map.blocked_doors(doors, [], ["jackal"]) == frozenset()


def test_gate_config_reads_preset():
    is_gate, authorized, locked = mechanism_map.gate_config(_door("g", gate={"authorized": ["a", "b"], "unlock_on": "alarm", "locked": True}))
    assert is_gate and authorized == {"a", "b"} and locked is True
    assert mechanism_map.gate_config(_door("p"))[0] is False


# paint_segments


def _blank_map(w: int = 100, h: int = 60, origin=(-1.0, -1.0), resolution: float = 0.1) -> tuple[bytes, str]:
    buf = io.BytesIO()
    PIL.Image.new("L", (w, h), 255).save(buf, format="PNG")
    return buf.getvalue(), Map.generate_map_yaml(resolution=resolution, filename="map.png", origin=origin)


def _pixel(png: bytes, yaml_text: str, x: float, y: float) -> int:
    import yaml

    meta = yaml.safe_load(yaml_text)
    img = PIL.Image.open(io.BytesIO(png))
    col = int((x - meta["origin"][0]) / meta["resolution"])
    row = img.size[1] - 1 - int((y - meta["origin"][1]) / meta["resolution"])
    return img.getpixel((col, row))


def test_paint_marks_segment_occupied_in_nav2_image_frame():
    png, yaml_text = _blank_map()
    painted = mechanism_map.paint_segments(png, yaml_text, [((4.0, 1.5), (4.0, 2.5), 0.1)])
    assert _pixel(painted, yaml_text, 4.0, 2.0) == 0
    assert _pixel(painted, yaml_text, 4.0, 1.55) == 0
    assert _pixel(painted, yaml_text, 4.0, 2.45) == 0
    assert _pixel(painted, yaml_text, 6.0, 2.0) == 255
    assert _pixel(painted, yaml_text, 4.0, 3.5) == 255
    assert _pixel(png, yaml_text, 4.0, 2.0) == 255


def test_paint_rgb_image():
    buf = io.BytesIO()
    PIL.Image.new("RGB", (100, 60), (255, 255, 255)).save(buf, format="PNG")
    yaml_text = Map.generate_map_yaml(resolution=0.1, filename="map.png", origin=(0.0, 0.0))
    painted = mechanism_map.paint_segments(buf.getvalue(), yaml_text, [((1.0, 1.0), (3.0, 3.0), 0.1)])
    assert PIL.Image.open(io.BytesIO(painted)).getpixel((20, 59 - 20)) == (0, 0, 0)


# door mask


def test_door_mask_leaves_out_the_doors_served_as_blocked():
    locked, plain = _door("g", activation=0.0), _door("p", start=(6.0, 1.5), end=(6.0, 2.5))
    world_doors = [("env0/g/1", locked), ("env0/p/1", plain)]
    assert mechanism_map.mask_doors([locked, plain], world_doors, frozenset({"env0/g/1"})) == [plain]
    assert mechanism_map.mask_doors([locked, plain], world_doors, frozenset()) == [locked, plain]


# WorldManagerROS flow


def _wm_ros(doors: list[Door], level_origin=(10.0, 0.0)):
    from arena_runtime.env.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS
    from arena_rclpy_mixins.Time import Time
    from arena_runtime.shared import Position
    import numpy as np

    class _Logger:
        def get_child(self, _n):
            return self

        def __getattr__(self, _n):
            return lambda *a, **kw: None

    snapshots: list = []
    wm = WorldManagerROS.__new__(WorldManagerROS)
    wm._NodeInterface__node = SimpleNamespace(
        _realizer=SimpleNamespace(prefix=lambda name, level: f"env0/{name}/{level}"),
        _simulator=SimpleNamespace(semantics_snapshot=lambda: list(snapshots)),
        get_logger=lambda: _Logger(),
    )
    zone = Zone(name="z", polygon=[GeoPosition(0, 0), GeoPosition(8, 0), GeoPosition(8, 4), GeoPosition(0, 4)])
    wm._world = World(levels={"1": Level(id="1", zones=[zone], doors=doors)})
    wm._map = WorldMap(
        occupancy=WorldLayers(walls=WorldOccupancy(np.full((1, 1), WorldOccupancy.EMPTY, dtype=np.uint8))),
        origin=Position(0.0, 0.0),
        resolution=0.1,
        time=Time(),
        level_origins={"1": level_origin},
    )
    wm._map_server_present = True
    wm._map_server_lock = asyncio.Lock()
    wm._served_map = None
    wm._mechanism_map_enabled = False
    wm._mechanism_agents = ()
    wm._mechanism_blocked = frozenset()
    wm._mechanism_served = frozenset()
    wm._mechanism_task = None
    loads: list[bytes] = []

    async def _load(png, yaml_text):
        loads.append(png)

    wm._load_png_to_server = _load
    wm.masks = []
    wm._publish_door_mask = lambda *a, **kw: wm.masks.append(wm._mechanism_served)
    return wm, snapshots, loads


def test_door_mask_leaves_ramps_to_the_sensors():
    from arena_simulation_setup.shared import Stairs
    from builtin_interfaces.msg import Time as TimeMsg

    wm, _snapshots, _loads = _wm_ros([])
    del wm.__dict__["_publish_door_mask"]
    sent: list = []
    wm._door_mask_pub = SimpleNamespace(publish=sent.append)
    wm._NodeInterface__node.sim_time = SimpleNamespace(to_msg=TimeMsg)
    wm._environment_manager = SimpleNamespace(realize=lambda position: position)
    slab = Stairs(name="slab", kind="ramp", position=GeoPosition(4.0, 1.0), size=[4.0, 1.0], rise=0.5)
    scan = Stairs(name="scan", kind="ramp", position=GeoPosition(4.0, 3.0), size=[4.0, 1.0], rise=0.5, floor=False)
    png, yaml_text = _blank_map(w=100, h=50, origin=(0.0, 0.0))

    wm._publish_door_mask(Level(id="1", stairs=[slab, scan]), png, yaml_text)

    cell = lambda x, y: sent[-1].data[int(y / sent[-1].info.resolution) * sent[-1].info.width + int(x / sent[-1].info.resolution)]
    assert (cell(4.0, 1.0), cell(4.0, 3.0), cell(8.0, 1.0)) == (-1, -1, -1)


def test_world_manager_ros_mechanism_map_follows_semantics():
    wm, snapshots, loads = _wm_ros([_door("g", gate={"authorized": ["jackal"], "locked": False})])
    png, yaml_text = _blank_map(w=200, h=60, origin=(0.0, 0.0))

    async def scenario():
        await wm._load_map_to_server(png, yaml_text, None)
        assert loads == [png]
        snapshots[:] = [_snap("env0/g/1", "gate", locked=True)]
        wm._update_mechanism_map(snapshots)
        assert wm._mechanism_task is None and len(loads) == 1

        wm.set_mechanism_map(True, ["burger", "env0/burger"])
        await wm._mechanism_task
        assert wm._mechanism_served == {"env0/g/1"} and len(loads) == 2
        assert wm.masks[-1] == {"env0/g/1"}
        assert _pixel(loads[-1], yaml_text, 14.0, 2.0) == 0
        assert _pixel(loads[-1], yaml_text, 4.0, 2.0) == 255

        wm._update_mechanism_map(snapshots)
        assert wm._mechanism_task.done() and len(loads) == 2

        snapshots[:] = [_snap("env0/g/1", "gate", locked=False)]
        wm._update_mechanism_map(snapshots)
        await wm._mechanism_task
        assert wm._mechanism_served == frozenset() and loads[-1] == png
        assert wm.masks[-1] == frozenset()

        snapshots[:] = [_snap("env0/g/1", "gate", locked=True)]
        wm._update_mechanism_map(snapshots)
        await wm._mechanism_task
        await wm._load_map_to_server(png, yaml_text, None)
        assert _pixel(loads[-1], yaml_text, 14.0, 2.0) == 0

        wm.set_mechanism_map(True, ["jackal"])
        await wm._mechanism_task
        assert loads[-1] == png
        snapshots[:] = [_snap("env0/g/1", "gate", locked=True)]
        wm.set_mechanism_map(True, ["burger"])
        await wm._mechanism_task
        wm.set_mechanism_map(False, [])
        await wm._mechanism_task
        assert loads[-1] == png and wm._mechanism_served == frozenset()

    asyncio.run(scenario())


def test_mechanism_map_without_map_server_is_noop():
    wm, snapshots, loads = _wm_ros([_door("d", activation=0.0)])
    wm._map_server_present = False
    wm.set_mechanism_map(True, ["jackal"])
    assert wm._mechanism_blocked == {"env0/d/1"} and wm._mechanism_task is None and loads == []
