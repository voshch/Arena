import asyncio
import io
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import arena_bringup.extensions.NodeLogLevelExtension as NodeLogLevelExtension
import arena_runtime_msgs.msg
import arena_runtime_msgs.srv
import attrs
import geometry_msgs.msg
import launch
import launch.actions
import launch.launch_description_sources
import launch_ros.actions
import lifecycle_msgs.msg
import nav2_msgs.srv
import nav_msgs.msg
import numpy as np
import PIL.Image
import rclpy.logging
import rclpy.qos
import visualization_msgs.msg
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins.Async import ClientWrapper, LaunchHandle
from arena_rclpy_mixins.shared import FrameNamespace
from arena_simulation_setup.shared import Door, Position, Wall
from arena_simulation_setup.tree.World import Level, World, WorldIdentifier, WorldView
from arena_simulation_setup.tree.World.model import render_door_polygons, render_elevator_polygons
from arena_simulation_setup.tree.World.view import point_world_assets

from arena_runtime._node import NodeInterface
from arena_runtime.env.environment_manager import EnvironmentManager
from arena_runtime.env.realizer import Realizer
from arena_runtime.flags import MapSource, map_source
from arena_runtime.humans.utils import ObstacleLayer

from . import markers, mechanism_map
from .utils import MultiLevelMap, WorldLayers, WorldMap, WorldOccupancy, door_mask
from .world_manager import WorldManager

if TYPE_CHECKING:
    from arena_runtime.env.node import EnvNode
    from arena_runtime.sim._semantics import SemanticEntitySnapshot

_DEFAULT_RESOLUTION = 0.05
_DOOR_MASK_BUFFER_M = 0.2


def map_server_launch_description(namespace: str, log_spec: str) -> launch.LaunchDescription:
    """The map_server bringup with its lifecycle manager under namespace, logging by the rule set log_spec."""
    return launch.LaunchDescription(
        [
            launch.actions.GroupAction(
                [
                    launch_ros.actions.PushRosNamespace(namespace),
                    NodeLogLevelExtension.SetGlobalLogLevelAction(log_spec),
                    launch.actions.IncludeLaunchDescription(launch.launch_description_sources.PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('arena_bringup'), 'launch/utils/map_server.launch.py'))),
                ]
            ),
        ]
    )


class MapServerHandler(NodeInterface):
    """Handler functions for the map server lifecycle."""

    _map_server_launch: LaunchHandle | None

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._map_server_launch = None

    async def launch_map_server(self, log_spec: str) -> None:
        """Replace any running map_server bringup with a fresh one in the node namespace."""
        if self._map_server_launch is not None:
            await self._map_server_launch.shutdown()
        self._map_server_launch = await self.node.do_launch_tracked(map_server_launch_description(self.node.get_fully_qualified_name(), log_spec))

    def _env_log_spec(self) -> str:
        return self.node.conf.General.LOG_LEVEL.value or rclpy.logging.get_logger_effective_level(self.node.get_logger().name).name.lower()

    async def ensure_map_server(self):
        """Restart the map server if it is not active."""

        wait_interval = 15.0

        while not await self.node.wait_for_lifecycle_state_async(
            self.node.service_namespace('map_server'),
            lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE,
            timeout=wait_interval,
        ):
            wait_interval = min(wait_interval * 2, 60.0)

            self._logger.warn('shutting down map server...')

            try:
                await self.node.change_lifecycle_state_async(self.node.service_namespace('map_server'), lifecycle_msgs.msg.Transition.TRANSITION_DESTROY, timeout=wait_interval)
            except TimeoutError:
                self._logger.warn('map server unresponsive to destroy.')
            else:
                self._logger.warn('map server shut down.')

            self._logger.warn('relaunching map server...')

            await self.launch_map_server(self._env_log_spec())

        self._logger.info('map server launched.')


class WorldManagerROS(MapServerHandler, WorldManager):
    """Initialize the WorldManager."""

    @property
    def node(self) -> "EnvNode":
        return super().node

    _environment_manager: EnvironmentManager

    _cli: ClientWrapper | None
    _cli_confirm_world: ClientWrapper
    _world_name: str
    _world_mtime: float
    _world_path: Path
    _world_sources: list[Path]
    _map_server_present: bool
    _map_render_memo: dict[str, tuple[bytes, str]]
    _static_markers: list[visualization_msgs.msg.Marker] | None
    _served_map: tuple[bytes, str, Level] | None
    _mechanism_map_enabled: bool
    _mechanism_agents: tuple[str, ...]
    _mechanism_blocked: frozenset[str]
    _mechanism_served: frozenset[str]
    _mechanism_task: asyncio.Task | None

    def _load_multi_level_map(self, world_view: WorldView, level_ids: Sequence[str]) -> MultiLevelMap | None:
        maps: dict[str, WorldMap] = {}
        for level_id in level_ids:
            level_yaml = world_view.map_yaml(level_id)
            if not level_yaml.is_file():
                continue
            try:
                maps[level_id] = WorldMap.from_map_files(level_yaml)
            except Exception as exc:
                self._logger.warn(f'failed to load level map {level_yaml}: {exc!r}')

        return MultiLevelMap(maps=maps) if maps else None

    def _shift_map(self, png_bytes: bytes, map_yaml_text: str) -> tempfile.TemporaryDirectory:
        """Write the rendered map into a tempdir with its origin shifted into this env's frame."""
        map_tmpdir = tempfile.TemporaryDirectory()

        map_yaml = yaml.safe_load(map_yaml_text)
        assert isinstance(map_yaml, dict), "map.yaml must be a dictionary"
        origin = list(map_yaml.get('origin', [0, 0, 0]))
        shifted_origin = self._environment_manager.realize(
            Position(
                x=origin[0],
                y=origin[1],
            )
        )
        origin[0] = shifted_origin.x
        origin[1] = shifted_origin.y
        map_yaml['origin'] = origin
        map_yaml['image'] = 'map.png'
        level_origins = map_yaml.get('origins')
        if isinstance(level_origins, dict):
            realizer: Realizer = self.node._realizer
            base_config = realizer.get_config()
            shifted_level_origins: dict[str, list[float]] = {}
            for level_id, offset in level_origins.items():
                if not isinstance(offset, (list, tuple)) or len(offset) < 2:
                    self._logger.warn(f"Skipping invalid level origin for {level_id!r}: {offset!r}")
                    continue
                ox = float(offset[0]) + base_config.x
                oy = float(offset[1]) + base_config.y
                shifted_level_origins[str(level_id)] = [ox, oy]
            if shifted_level_origins:
                map_yaml['origins'] = shifted_level_origins

        tmp = Path(map_tmpdir.name)
        (tmp / 'map.png').write_bytes(png_bytes)
        with open(tmp / 'map.yaml', 'w') as f:
            yaml.safe_dump(map_yaml, f)

        return map_tmpdir

    def _publish_anchor_tf(self, ref_x: float, ref_y: float) -> None:
        """Publish the global `map` -> `<prefix>/map` static transform that anchors the env's local frame."""
        prefix = self.node.rosparam[str].get('prefix', '')
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = FrameNamespace(prefix).tf('map')
        t.transform.translation.x = ref_x
        t.transform.translation.y = ref_y
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.node._static_tf_broadcaster.sendTransform(t)

    async def _render_world_map(
        self,
        world_name: str,
        description: Level,
        level_origins: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[bytes, str]:
        """Render (map.png bytes, map.yaml text) for a world, memoized per world for this process."""
        cached = self._map_render_memo.get(world_name)
        if cached is None:
            # rasterizing a large world is CPU-heavy, keep it off the event loop
            cached = await asyncio.to_thread(
                description.render_map_files,
                level_origins=level_origins,
                resolution=_DEFAULT_RESOLUTION,
                asset_color="black",
            )
            self._map_render_memo[world_name] = cached
        return cached

    async def apply_world(self, world_name: str) -> bool:
        """Synchronously swap to `world_name`. Must be called inside `_run_reset_cycle`'s hold."""
        bare_name, level_filter = WorldIdentifier.split_filter(world_name)
        world_view = await WorldIdentifier(bare_name).resolve()
        mtime = world_view.mtime(self._world_sources if world_name == self._world_name else ())

        if world_name == self._world_name and mtime == self._world_mtime:
            return True

        self._logger.info(f'World change requested: {world_name}')

        # regeneration reuses the name but rewrites world.yaml; drop the stale render
        self._map_render_memo.pop(world_name, None)

        world = world_view.load(level_filter=level_filter, derive=self.node.conf.Arena.KG_DERIVE.value or 'auto')
        level_origins = world_view.level_origins()
        origins = level_origins if level_origins is not None else {fid: (0.0, 0.0) for fid in world.level_ids}
        offsets = self._stack_offsets(world_name, world)
        self.node._realizer.reset_level_origins(origins)
        self.node._realizer.reset_placement(offsets, world.level_bounds())
        self.node._realizer.reset_elevator_levels(world.elevator_levels())
        self.node._realizer.reset_stair_levels(world.stair_levels())
        compacted_description = world.compact_world(origins=origins)

        multi_level_map = self._load_multi_level_map(world_view, world.level_ids)

        source = map_source(self.node)
        disk_mode = source is MapSource.DISK
        disk_level = ""
        disk_map: WorldMap | None = None
        if disk_mode:
            disk_levels = list(multi_level_map.level_ids) if multi_level_map is not None else []
            if multi_level_map is None or len(world.level_ids) != 1 or len(disk_levels) != 1:
                raise RuntimeError(f"debug.map_source:=disk requires a single-level world with one on-disk map; world {world_name!r} has {len(world.level_ids)} level(s), {len(disk_levels)} disk map(s)")
            disk_level = disk_levels[0]
            disk_map = multi_level_map.select_map(disk_level)

        if disk_map is not None:
            world_map = disk_map
        else:
            world_map = WorldMap.from_world_description(compacted_description, resolution=_DEFAULT_RESOLUTION, time=self.node.sim_time, _level_origins=level_origins)
        x_min, y_min, x_max, y_max = world_map.bounds
        for level_id, (bx0, by0, bx1, by1) in world.level_bounds().items():
            if level_id in offsets:
                dx, dy, _dz = offsets[level_id]
                x_min, y_min, x_max, y_max = min(x_min, bx0 + dx), min(y_min, by0 + dy), max(x_max, bx1 + dx), max(y_max, by1 + dy)
        extent = arena_runtime_msgs.msg.WorldExtent(x_min=float(x_min), y_min=float(y_min), x_max=float(x_max), y_max=float(y_max))

        req = arena_runtime_msgs.srv.ConfirmWorld.Request()
        req.env_id = self.node._env_id
        req.extent = extent
        confirm = await self._cli_confirm_world.call_timeout(req)
        if confirm is None:
            self._logger.error(f'confirm_world timed out for world {world_name!r}; aborting world change')
            return False
        if not confirm.success:
            self._logger.error(f'confirm_world rejected world {world_name!r}: {confirm.error_msg}')
            return False

        if confirm.reallocated:
            ref_x = float(confirm.reference[0])
            ref_y = float(confirm.reference[1])
            self.node._reference = (ref_x, ref_y)
            self.node._prespawn_offset = (
                float(confirm.prespawn[0]) - ref_x,
                float(confirm.prespawn[1]) - ref_y,
            )
            self.node._realizer.set_origin(ref_x, ref_y)
            self._publish_anchor_tf(ref_x, ref_y)

        self._logger.warn(f'Loading World {world_name}')

        point_world_assets(world_view.path, world.sources)
        static_footprints = await asyncio.gather(*(o.footprint() for o in world.all_static_entities))
        self.update_world(world_map=world_map, world_description=world, multi_level_map=multi_level_map, static_footprints=static_footprints)
        self._mechanism_blocked = frozenset()
        self._mechanism_served = frozenset()

        self._world_name = world_name
        self._world_mtime = world_view.mtime(world.sources)
        self._world_path = Path(world_view.path)
        self._world_sources = list(world.sources)
        self._static_markers = None
        self.node.rosparam[str].set('world', world_name)

        if self._map_server_present:
            await self._push_map_source(world_view, world.level_ids, compacted_description, level_origins)

        await self._environment_manager.reset(purge=ObstacleLayer.WORLD)
        detected_walls: dict[str, Sequence[Wall]] = {}
        if disk_map is not None:
            if any(level.pedestrian_walls for level in world.all_levels):
                detected_walls = {disk_level: tuple(disk_map.detect_walls())}
        await self._environment_manager.spawn_world_obstacles(self._world, detected_walls=detected_walls, world_map=world_map)
        self.node._publish_semantics_snapshot()

        return True

    def _stack_offsets(self, world_name: str, world: World) -> dict[str, tuple[float, float, float]]:
        """Per level simulator offsets under `world.levels`, empty to spread, every stair and escalator pair marked joined when stacked."""
        mode = self.node.conf.Arena.WORLD_LEVELS.value or 'auto'
        simulator = self._environment_manager.simulator_name
        stacks = self._environment_manager.stacks_levels
        if mode == 'stack' and world.layout != 'stacked':
            raise RuntimeError(f'world.levels:=stack: {world_name!r} is a spread world, every level has a floor plan of its own')
        if mode == 'stack' and not stacks:
            raise RuntimeError(f'world.levels:=stack: the {simulator} simulator lays levels out side by side only')
        if mode == 'auto' and world.layout == 'stacked' and not stacks:
            self._logger.warning(f'world.levels:=auto spreads {world_name!r}: the {simulator} simulator lays levels out side by side only')
        if len(world.levels) < 2 or not (mode == 'stack' or (mode == 'auto' and world.layout == 'stacked' and stacks)):
            return {}
        offsets = world.stack_offsets()
        stairs = [stair for level in world.levels.values() for stair in level.stairs]
        names = {stair.name for stair in stairs}
        for stair in stairs:
            stair.joined = stair.to in names or any(other.to == stair.name for other in stairs)
        self._logger.info(f'world.levels stacks {world_name!r}: {offsets}')
        return offsets

    async def _load_png_to_server(self, png_bytes: bytes, map_yaml_text: str) -> None:
        """Shift+LoadMap. Caller guarantees map_server is present."""
        tmp_map = self._shift_map(png_bytes, map_yaml_text)
        try:
            map_yaml = os.path.join(tmp_map.name, 'map.yaml')
            assert self._cli is not None
            response = await self._cli.call_timeout(nav2_msgs.srv.LoadMap.Request(map_url=f'{map_yaml}'))
        finally:
            tmp_map.cleanup()
        if response is None:
            raise RuntimeError(f'failed to load map for world {self._world_name}: service timed out')
        if response.result > 0:
            raise RuntimeError(f'failed to load map for world {self._world_name}: status code {response.result}')

    async def _load_map_to_server(self, png_bytes: bytes, map_yaml_text: str, description: Level) -> None:
        """Shift+LoadMap+door-mask. Caller guarantees map_server is present. Applies the mechanism map if enabled."""
        self._served_map = (png_bytes, map_yaml_text, description)
        blocked = self._mechanism_blocked if self._mechanism_map_enabled else frozenset()
        await self._load_png_to_server(self._paint_blocked(png_bytes, map_yaml_text, blocked), map_yaml_text)
        self._mechanism_served = blocked
        self._publish_door_mask(description, png_bytes, map_yaml_text)

    def _world_doors(self) -> list[tuple[str, Door]]:
        """(realized name, door shifted into the flattened map frame) for every loaded door."""
        realizer: Realizer = self.node._realizer
        doors: list[tuple[str, Door]] = []
        for level_id, level in self._world.levels.items():
            ox, oy = self._map.level_origins.get(level_id, (0.0, 0.0))
            for door in level.all_doors:
                shifted = attrs.evolve(door, start=Position(door.start.x + ox, door.start.y + oy), end=Position(door.end.x + ox, door.end.y + oy))
                doors.append((str(realizer.prefix(door.name, level_id)), shifted))
        return doors

    def _paint_blocked(self, png_bytes: bytes, map_yaml_text: str, blocked: frozenset[str]) -> bytes:
        if not blocked:
            return png_bytes
        segments = [((d.start.x, d.start.y), (d.end.x, d.end.y), d.width) for name, d in self._world_doors() if name in blocked]
        return mechanism_map.paint_segments(png_bytes, map_yaml_text, segments)

    def set_mechanism_map(self, enabled: bool, agents: Sequence[str]) -> None:
        """Enable/disable the mechanism-aware planner map for the robot names/sim paths in `agents`."""
        self._mechanism_map_enabled = enabled
        self._mechanism_agents = tuple(agents)
        if not enabled:
            self._mechanism_blocked = frozenset()
            self._schedule_mechanism_push()
            return
        simulator = self.node._simulator
        if simulator is None:
            self._logger.warning('mechanism map: no simulator yet')
            return
        try:
            snapshots = simulator.semantics_snapshot()
        except Exception as exc:
            self._logger.warning(f'mechanism map: no semantics snapshot yet ({exc!r})')
            return
        self._update_mechanism_map(snapshots)

    def _update_mechanism_map(self, snapshots: "Sequence[SemanticEntitySnapshot]") -> None:
        if not self._mechanism_map_enabled:
            return
        self._mechanism_blocked = mechanism_map.blocked_doors(self._world_doors(), snapshots, self._mechanism_agents)
        self._schedule_mechanism_push()

    def _schedule_mechanism_push(self) -> None:
        if self._mechanism_blocked == self._mechanism_served or not self._map_server_present or self._served_map is None:
            return
        if self._mechanism_task is not None and not self._mechanism_task.done():
            return
        try:
            self._mechanism_task = asyncio.get_running_loop().create_task(self._mechanism_push_loop())
        except RuntimeError:
            self._mechanism_task = None

    async def _mechanism_push_loop(self) -> None:
        """Reload the served map until it matches the latest blocked-door set (latest wins)."""
        try:
            while self._served_map is not None and self._mechanism_served != self._mechanism_blocked:
                blocked = self._mechanism_blocked
                png_bytes, map_yaml_text, description = self._served_map
                async with self._map_server_lock:
                    await self._load_png_to_server(self._paint_blocked(png_bytes, map_yaml_text, blocked), map_yaml_text)
                self._mechanism_served = blocked
                self._publish_door_mask(description, png_bytes, map_yaml_text)
                self._logger.info(f'mechanism map: {len(blocked)} door(s) served as occupied: {sorted(blocked)}')
        except Exception as exc:
            self._logger.error(f'mechanism map push failed: {exc!r}')

    async def _push_world_to_map_server(self, description: Level, level_origins: dict[str, tuple[float, float]] | None = None) -> None:
        """Render+load the computed map. Caller guarantees map_server is present."""
        png_bytes, map_yaml_text = await self._render_world_map(self._world_name, description, level_origins=level_origins)
        await self._load_map_to_server(png_bytes, map_yaml_text, description)

    async def _push_disk_map_to_map_server(self, map_yaml: Path, description: Level) -> None:
        """Load an on-disk preview map and the image it names. Caller guarantees map_server is present."""
        map_yaml_text = map_yaml.read_text()
        png_bytes = (map_yaml.parent / yaml.safe_load(map_yaml_text)['image']).read_bytes()
        await self._load_map_to_server(png_bytes, map_yaml_text, description)

    async def _push_map_source(self, world_view: WorldView, level_ids: Sequence[str], description: Level | None, level_origins: dict[str, tuple[float, float]] | None) -> None:
        """Serve the map `debug.map_source` selects. Caller guarantees map_server is present."""
        if description is None:
            return
        if map_source(self.node) is MapSource.DISK:
            await self._push_disk_map_to_map_server(world_view.map_yaml(level_ids[0]), description)
        else:
            await self._push_world_to_map_server(description, level_origins=level_origins)

    def _publish_door_mask(
        self,
        description: Level,
        png_bytes: bytes,
        map_yaml_text: str,
    ) -> None:
        """Publish an OccupancyGrid: 0 inside door polygons, -1 elsewhere, without the doors the mechanism map serves as blocked. Aligned to the static map."""
        img = PIL.Image.open(io.BytesIO(png_bytes))
        width, height = img.size

        map_yaml = yaml.safe_load(map_yaml_text)
        resolution = float(map_yaml['resolution'])
        origin_world = list(map_yaml.get('origin', [0, 0, 0]))

        doors = mechanism_map.mask_doors(description.all_doors, self._world_doors(), self._mechanism_served)
        polys = [p for d in doors for p in render_door_polygons(d)] + [p for e in description.all_elevators for p in render_elevator_polygons(e)]
        data = door_mask(polys, description.all_walls, width, height, resolution, (float(origin_world[0]), float(origin_world[1])), _DOOR_MASK_BUFFER_M)

        shifted_origin = self._environment_manager.realize(Position(x=float(origin_world[0]), y=float(origin_world[1])))
        assert isinstance(shifted_origin, Position)

        grid = nav_msgs.msg.OccupancyGrid()
        grid.header.frame_id = 'map'
        grid.header.stamp = self.node.sim_time.to_msg()
        grid.info.resolution = resolution
        grid.info.width = width
        grid.info.height = height
        grid.info.origin.position.x = shifted_origin.x
        grid.info.origin.position.y = shifted_origin.y
        grid.info.origin.orientation.w = 1.0
        grid.data = data.tolist()

        self._door_mask_pub.publish(grid)

    def publish_world_markers(self, snapshots: "Sequence[SemanticEntitySnapshot]") -> None:
        """Publish the latched world overlay, clearing the previous world's markers on the first publish after a change."""
        stamp = self.node.sim_time.to_msg()
        realizer: Realizer = self.node._realizer
        msg = visualization_msgs.msg.MarkerArray()
        if self._static_markers is None:
            self._static_markers = markers.static_markers(self._world, realizer, stamp)
            msg.markers.extend(markers.deleteall_markers())  # pyright: ignore[reportAttributeAccessIssue]
        msg.markers.extend(self._static_markers)  # pyright: ignore[reportAttributeAccessIssue]
        msg.markers.extend(markers.mechanism_markers(self._world, realizer, snapshots, stamp))  # pyright: ignore[reportAttributeAccessIssue]
        self._world_markers_pub.publish(msg)
        self._update_mechanism_map(snapshots)

    def refresh_world_markers(self, snapshots: "Sequence[SemanticEntitySnapshot]") -> None:
        """Republish the world overlay from the held description, rebuilding the static layers."""
        self._static_markers = None
        self.publish_world_markers(snapshots)

    async def require_map_server(self) -> None:
        """Idempotent: lazy-launch map_server and push the current world. Safe to call concurrently."""
        async with self._map_server_lock:
            if not self._map_server_present:
                await self.launch_map_server(self._env_log_spec())
                self._map_server_present = True
            if self._cli is None:
                self._cli = self.node.create_client_wrapper(
                    nav2_msgs.srv.LoadMap,
                    self.node.service_namespace('map_server', 'load_map'),
                )
            await self._cli.ensure()
            await self.ensure_map_server()
            if self._world_name:
                bare_wn, wn_filter = WorldIdentifier.split_filter(self._world_name)
                world_view = await WorldIdentifier(bare_wn).resolve()
                loaded = world_view.load(level_filter=wn_filter, derive=self.node.conf.Arena.KG_DERIVE.value or 'auto')
                origins = world_view.level_origins()
                compacted = loaded.compact_world(origins=origins if origins is not None else {fid: (0.0, 0.0) for fid in loaded.level_ids})
                await self._push_map_source(world_view, loaded.level_ids, compacted, origins)

    def __init__(self, *args: object, environment_manager: EnvironmentManager, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._environment_manager = environment_manager

        self.update_world(
            world_map=WorldMap(
                occupancy=WorldLayers(walls=WorldOccupancy(np.full((1, 1), WorldOccupancy.EMPTY, dtype=np.uint8))),
                origin=Position(x=0.0, y=0.0),
                resolution=_DEFAULT_RESOLUTION,
                time=self.node.sim_time,
            ),
            world_description=World.from_levels(Level()),
        )
        self._world_name = ''
        self._world_mtime = 0.0
        self._world_path = Path()
        self._world_sources = []
        self._cli = None
        self._map_server_present = False
        self._map_server_lock = asyncio.Lock()
        self._map_render_memo = {}
        self._static_markers = None
        self._served_map = None
        self._mechanism_map_enabled = False
        self._mechanism_agents = ()
        self._mechanism_blocked = frozenset()
        self._mechanism_served = frozenset()
        self._mechanism_task = None
        self._world_markers_pub = self.node.create_publisher(
            visualization_msgs.msg.MarkerArray,
            'world_markers',
            rclpy.qos.QoSProfile(
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self._door_mask_pub = self.node.create_publisher(
            nav_msgs.msg.OccupancyGrid,
            'door_mask',
            rclpy.qos.QoSProfile(
                durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
                history=rclpy.qos.HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

    async def start(self):
        """Start the world manager."""
        self._logger.info("starting")

        self._cli_confirm_world = self.node.create_client_wrapper(
            arena_runtime_msgs.srv.ConfirmWorld,
            "/arena/confirm_world",
        )
        await self._cli_confirm_world.ensure()

        initial_world = self.node.conf.Arena.WORLD.value
        if initial_world:
            await self.apply_world(initial_world)

    @property
    def loaded_world(self) -> str:
        """Currently loaded world, which trails the episode loop's next-episode world briefly during a reset cycle."""
        return self._world_name

    @property
    def loaded_world_path(self) -> Path:
        """Directory the loaded world resolved from, empty before the first world."""
        return self._world_path

    async def sync(self, timeout: float = -1) -> bool:
        """Wait until at least one world has been applied. Used by external lifecycle callers."""
        start_time = self.node.get_clock().now().seconds_nanoseconds()[0]
        while not self._world_name:
            await asyncio.sleep(0.01)
            if timeout >= 0:
                elapsed = self.node.get_clock().now().seconds_nanoseconds()[0] - start_time
                if elapsed >= timeout:
                    return False
        return True
