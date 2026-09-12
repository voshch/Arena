"""Engine-less backend for human:=dummy, motion is authored upstream (human_steering)."""

import asyncio
import copy
import traceback
from collections.abc import Mapping, Sequence

import rclpy.qos
from arena_people_msgs.msg import Pedestrian, Pedestrians
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from task_generator.constants.rng import stable_int
from task_generator.shared import DynamicObstacle, Obstacle, Wall
from task_generator.simulators.human.noop import NoopHumanSimulator
from task_generator.simulators.human.possession import snapshot_roster

_STATIC_QOS = rclpy.qos.QoSProfile(
    reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
    durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
    history=rclpy.qos.HistoryPolicy.KEEP_LAST,
    depth=1,
)

_KEEPALIVE_HZ = 30.0


def _deleteall_marker(ns: str) -> Marker:
    marker = Marker()
    marker.header.frame_id = "map"
    marker.ns = ns
    marker.action = Marker.DELETEALL
    return marker


class DummyHumanSimulator(NoopHumanSimulator):
    """Puppet stage: peds idle at their roster pose until the possession stream drives them."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)

        self._cache: dict[str, Pedestrian] = {}
        self._walls: dict[str, Wall] = {}
        self._static_obstacles: dict[str, Obstacle] = {}

        self._static_walls_pub = self.node.create_publisher(
            MarkerArray,
            self._namespace("pedestrian_markers", "static_walls"),
            _STATIC_QOS,
        )
        self._static_objects_pub = self.node.create_publisher(
            MarkerArray,
            self._namespace("pedestrian_markers", "static_objects"),
            _STATIC_QOS,
        )

        self._keepalive_task: asyncio.Task = asyncio.create_task(self._keepalive_loop())

    # possession hooks

    def _on_stream_merged(self, names: Sequence[str]) -> None:
        del names
        self._publish_roster()

    def _on_ped_released(self, name: str, last_state: Pedestrian) -> None:
        cached = self._cache.get(name)
        if cached is None:
            return
        cached.pose = copy.deepcopy(last_state.pose)
        cached.joint_state = JointState()

    def _on_ped_moved(self, name: str, pose: Pose) -> None:
        cached = self._cache.get(name)
        if cached is None:
            return
        cached.pose = copy.deepcopy(pose)
        self._publish_roster()

    def _publish_roster(self) -> None:
        out = Pedestrians()
        out.header.frame_id = "map"
        out.pedestrians = snapshot_roster(self._cache)
        self.publish_arena_peds(out)

    # keepalive

    async def _keepalive_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(1.0 / _KEEPALIVE_HZ)
                self._publish_roster()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.error(f"Error in keepalive loop: {e}\n{traceback.format_exc()}")

    # lifecycle

    async def _spawn_dynamic_obstacles_impl(
        self,
        obstacles: Sequence[DynamicObstacle],
    ) -> Sequence[DynamicObstacle | None]:
        for obstacle in obstacles:
            ped = Pedestrian()
            ped.name = obstacle.sim_path
            ped.id = stable_int(obstacle.sim_path) & 0xFFFFFFFF
            ped.pose = obstacle.pose.to_msg()
            ped.animation_state = Pedestrian.IDLE
            self._cache[obstacle.sim_path] = ped
        return obstacles

    async def _remove_pedestrians_impl(self) -> bool:
        tracked = [obstacle for key, obstacle in self._ped_bus_index.items() if key == obstacle.sim_path]
        await self._simulator.pedestrian_delete(tracked)
        self._cache.clear()
        return True

    # static world/obstacle markers

    async def _spawn_obstacles_impl(
        self,
        obstacles: Sequence[Obstacle],
    ) -> Sequence[Obstacle | None]:
        for obstacle in obstacles:
            self._static_obstacles[obstacle.name] = obstacle
        self._publish_static_objects()
        return obstacles

    async def _remove_obstacles_impl(self, names: Sequence[str]) -> bool:
        for name in names:
            self._static_obstacles.pop(name, None)
        self._publish_static_objects()
        return True

    async def _spawn_walls_impl(self, walls: Mapping[str, Wall]) -> bool:
        self._walls = dict(walls)
        self._publish_static_walls()
        return True

    async def _remove_walls_impl(self, names: Sequence[str]) -> bool:
        for name in names:
            self._walls.pop(name, None)
        self._publish_static_walls()
        return True

    def _publish_static_walls(self) -> None:
        markers = MarkerArray()
        markers.markers.append(_deleteall_marker("dummy_walls"))
        stamp = self.node.sim_time.to_msg()
        for idx, wall in enumerate(self._walls.values()):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "dummy_walls"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05
            marker.color.r = 1.0
            marker.color.a = 1.0
            marker.points = [wall.start.to_msg(), wall.end.to_msg()]
            markers.markers.append(marker)
        self._static_walls_pub.publish(markers)

    def _publish_static_objects(self) -> None:
        markers = MarkerArray()
        markers.markers.append(_deleteall_marker("dummy_objects"))
        stamp = self.node.sim_time.to_msg()
        for idx, obstacle in enumerate(self._static_obstacles.values()):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp
            marker.ns = "dummy_objects"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose = obstacle.pose.to_msg()
            scale = obstacle.scale
            marker.scale.x = scale.x if scale is not None else 0.5
            marker.scale.y = scale.y if scale is not None else 0.5
            marker.scale.z = scale.z if scale is not None else 0.5
            marker.color.b = 1.0
            marker.color.a = 0.6
            markers.markers.append(marker)
        self._static_objects_pub.publish(markers)
