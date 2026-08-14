import asyncio
import io
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import arena_runtime_msgs.msg
import arena_runtime_msgs.srv
import arena_simulation_setup.tree.World as World
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
import rclpy.qos
import shapely
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_rclpy_mixins.Async import ClientWrapper
from arena_rclpy_mixins.shared import FrameNamespace
from arena_runtime._node import NodeInterface
from arena_simulation_setup.shared import Position
from arena_simulation_setup.tree import DynamicPaths
from arena_simulation_setup.tree.World.World import _render_door_polygons, _render_elevator_polygons

from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.realizer import Realizer
from task_generator.simulators.human.utils import ObstacleLayer
from task_generator.utils.flags import MapSource, map_source

from .utils import MultiLevelMap, WorldLayers, WorldMap, WorldOccupancy
from .world_manager import WorldManager

if TYPE_CHECKING:
    from task_generator.node import TaskGenerator

_DEFAULT_RESOLUTION = 0.05
_DOOR_MASK_BUFFER_M = 0.2


class MapServerHandler(NodeInterface):
    """Handler functions for the map server lifecycle."""

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

            await self.node.do_launch(launch.LaunchDescription([launch.actions.IncludeLaunchDescription(launch.launch_description_sources.PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('arena_bringup'), 'launch/utils/map_server.launch.py')))]))

        self._logger.info('map server launched.')


class WorldManagerROS(MapServerHandler, WorldManager):
    """Initialize the WorldManager."""

    node: "TaskGenerator"

    _environment_manager: EnvironmentManager

    _cli: ClientWrapper | None
    _cli_confirm_world: ClientWrapper
    _world_name: str
    _world_mtime: float
    _map_server_present: bool
    _map_render_memo: dict[str, tuple[bytes, str]]

    def _load_multi_level_map(self, world_root: Path) -> MultiLevelMap | None:
        maps: dict[str, WorldMap] = {}
        for level_yaml in sorted(world_root.glob('*/map.yaml')):
            level_id = level_yaml.parent.name
            if level_id in {'scenarios', 'assets', 'map'}:
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
        description: World.LevelDescription,
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
        bare_name, level_filter = World.WorldIdentifier.parse(world_name)
        world_view = await World.WorldIdentifier(bare_name).resolve()
        mtimes = [p.stat().st_mtime for p in Path(world_view.path).glob('*/world.yaml')]
        mtime = max(mtimes) if mtimes else 0.0

        if world_name == self._world_name and mtime == self._world_mtime:
            return True

        self._logger.info(f'World change requested: {world_name}')

        # regeneration reuses the name but rewrites world.yaml; drop the stale render
        self._map_render_memo.pop(world_name, None)

        world = world_view.load(level_filter=level_filter)
        world.apply_elevator_door_sides()
        level_origins = world_view.level_origins()
        origins = level_origins if level_origins is not None else {fid: (0.0, 0.0) for fid in world.level_ids}
        self.node._realizer.reset_level_origins(origins)
        compacted_description = world.compact_world(origins=origins)

        multi_level_map = self._load_multi_level_map(Path(world_view.path))

        source = map_source(self.node)
        disk_mode = source is MapSource.DISK
        disk_level = ""
        if disk_mode:
            disk_levels = list(multi_level_map.level_ids) if multi_level_map is not None else []
            if len(world.level_ids) != 1 or len(disk_levels) != 1:
                raise RuntimeError(f"debug.map_source:=disk requires a single-level world with one on-disk map; world {world_name!r} has {len(world.level_ids)} level(s), {len(disk_levels)} disk map(s)")
            disk_level = disk_levels[0]
            disk_map = multi_level_map.get_map(disk_level)

        floors = [floor for level in world.all_levels for floor in level.all_floors]
        extent = arena_runtime_msgs.msg.WorldExtent()
        if disk_mode:
            rows, cols = disk_map.shape
            extent.x_min = float(disk_map.origin.x)
            extent.y_min = float(disk_map.origin.y)
            extent.x_max = float(disk_map.origin.x + cols * disk_map.resolution)
            extent.y_max = float(disk_map.origin.y + rows * disk_map.resolution)
        elif floors:
            extent.x_min = float(min(f.pos.x - f.x_length / 2 for f in floors))
            extent.y_min = float(min(f.pos.y - f.y_length / 2 for f in floors))
            extent.x_max = float(max(f.pos.x + f.x_length / 2 for f in floors))
            extent.y_max = float(max(f.pos.y + f.y_length / 2 for f in floors))

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

        if disk_mode:
            world_map = disk_map
        else:
            if compacted_description is None:
                compacted_description = world.compact_world(origins={fid: (0.0, 0.0) for fid in world.level_ids})
            world_map = WorldMap.from_world_description(compacted_description, resolution=_DEFAULT_RESOLUTION, time=self.node.sim_time, _level_origins=level_origins)
        DynamicPaths.WORLD.path = world_view.path
        self.update_world(world_map=world_map, world_description=world, multi_level_map=multi_level_map)

        self._world_name = world_name
        self._world_mtime = mtime
        self.node.rosparam[str].set('world', world_name)

        if self._map_server_present:
            if disk_mode:
                await self._push_disk_map_to_map_server(Path(world_view.path) / disk_level, compacted_description)
            elif compacted_description is not None:
                await self._push_world_to_map_server(compacted_description, level_origins=level_origins)

        await self._environment_manager.reset(purge=ObstacleLayer.WORLD)
        detected_walls = {}
        if disk_mode:
            zones = [zone for level in world.all_levels for zone in level.zones]
            if not zones or any(zone.wall_material.name for zone in zones):
                detected_walls = {disk_level: tuple(disk_map.detect_walls())}
        await self._environment_manager.spawn_world_obstacles(self._world, detected_walls=detected_walls)
        self.node._publish_semantics_snapshot()

        return True

    async def _load_map_to_server(self, png_bytes: bytes, map_yaml_text: str, description: World.LevelDescription) -> None:
        """Shift+LoadMap+door-mask. Caller guarantees map_server is present."""
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
        self._publish_door_mask(description, png_bytes, map_yaml_text)

    async def _push_world_to_map_server(self, description: World.LevelDescription, level_origins: dict[str, tuple[float, float]] | None = None) -> None:
        """Render+load the computed map. Caller guarantees map_server is present."""
        png_bytes, map_yaml_text = await self._render_world_map(self._world_name, description, level_origins=level_origins)
        await self._load_map_to_server(png_bytes, map_yaml_text, description)

    async def _push_disk_map_to_map_server(self, level_dir: Path, description: World.LevelDescription) -> None:
        """Load the on-disk map.png/map.yaml. Caller guarantees map_server is present."""
        png_bytes = (level_dir / 'map.png').read_bytes()
        map_yaml_text = (level_dir / 'map.yaml').read_text()
        await self._load_map_to_server(png_bytes, map_yaml_text, description)

    def _publish_door_mask(
        self,
        description: World.LevelDescription,
        png_bytes: bytes,
        map_yaml_text: str,
    ) -> None:
        """Publish an OccupancyGrid: 0 inside door polygons, -1 elsewhere. Aligned to the static map."""
        img = PIL.Image.open(io.BytesIO(png_bytes))
        width, height = img.size

        map_yaml = yaml.safe_load(map_yaml_text)
        resolution = float(map_yaml['resolution'])
        origin_world = list(map_yaml.get('origin', [0, 0, 0]))

        data = np.full(width * height, -1, dtype=np.int8)
        polys = [p for d in description.all_doors for p in _render_door_polygons(d)] + [p for e in description.all_elevators for p in _render_elevator_polygons(e)]
        for poly in polys:
            if poly.is_empty:
                continue
            # Buffer carves into adjacent walls so global costmap inflation
            # doesn't fully close the opening.
            poly = poly.buffer(_DOOR_MASK_BUFFER_M)
            min_x, min_y, max_x, max_y = poly.bounds
            col_min = max(0, int((min_x - origin_world[0]) / resolution))
            col_max = min(width, int((max_x - origin_world[0]) / resolution) + 1)
            row_min = max(0, int((min_y - origin_world[1]) / resolution))
            row_max = min(height, int((max_y - origin_world[1]) / resolution) + 1)
            for row in range(row_min, row_max):
                cy = origin_world[1] + (row + 0.5) * resolution
                for col in range(col_min, col_max):
                    cx = origin_world[0] + (col + 0.5) * resolution
                    if poly.covers(shapely.Point(cx, cy)):
                        data[row * width + col] = 0

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

    async def require_map_server(self) -> None:
        """Idempotent: lazy-launch map_server and push the current world. Safe to call concurrently."""
        async with self._map_server_lock:
            if not self._map_server_present:
                await self.node.do_launch(
                    launch.LaunchDescription(
                        [
                            launch.actions.GroupAction(
                                [
                                    launch_ros.actions.PushRosNamespace(self.node.get_fully_qualified_name()),
                                    launch.actions.IncludeLaunchDescription(launch.launch_description_sources.PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('arena_bringup'), 'launch/utils/map_server.launch.py'))),
                                ]
                            ),
                        ]
                    )
                )
                self._map_server_present = True
            if self._cli is None:
                self._cli = self.node.create_client_wrapper(
                    nav2_msgs.srv.LoadMap,
                    self.node.service_namespace('map_server', 'load_map'),
                )
            await self._cli.ensure()
            await self.ensure_map_server()
            if self._world_name:
                bare_wn, wn_filter = World.WorldIdentifier.parse(self._world_name)
                world_view = await World.WorldIdentifier(bare_wn).resolve()
                loaded = world_view.load(level_filter=wn_filter)
                origins = world_view.level_origins()
                compacted = loaded.compact_world(origins=origins if origins is not None else {fid: (0.0, 0.0) for fid in loaded.level_ids})
                if compacted is not None:
                    await self._push_world_to_map_server(compacted, level_origins=origins)

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
            world_description=World.WorldDescription.from_levels(World.LevelDescription()),
        )
        self._world_name = ''
        self._world_mtime = 0.0
        self._cli = None
        self._map_server_present = False
        self._map_server_lock = asyncio.Lock()
        self._map_render_memo = {}
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
        """Currently loaded world. Read `node._episodes.current.world` for the intended next-episode world (the two diverge briefly during a reset cycle)."""
        return self._world_name

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
