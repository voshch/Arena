from __future__ import annotations

import math
from typing import TYPE_CHECKING

import arena_people_msgs.msg
import arena_robots_msgs.msg
import geometry_msgs.msg
import nav2_msgs.msg
import rclpy
import rclpy.node
import rclpy.time
import shapely
import shapely.affinity
import std_msgs.msg
from arena_rclpy_mixins.shared import Namespace
from arena_robots.caps import PolygonSpec
from rclpy.parameter import Parameter

if TYPE_CHECKING:
    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.manager.robot_manager.robot_manager import RobotManager

_ACTION_CODES: dict[str | None, int] = {
    None: 0,
    'stop': 1,
    'slowdown': 2,
    'approach': 3,
    'limit': 4,
}


class CollisionTrackerNode(rclpy.node.Node):
    """Separate Node, same process, same executor as the task_generator.

    Reads RobotManager.pose + EnvironmentManager geometry caches and publishes
    collision topics. Tick timer uses sim time → automatically gated during
    pauses.
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        polygons: dict[str, PolygonSpec],
        *,
        rate_hz: float = 10.0,
        near_miss_margin: float = 0.0,
        default_pedestrian_radius: float = 0.3,
        footprint: list[list[float]] | None = None,
    ):
        super().__init__(
            'collision_tracker',
            namespace=str(robot_manager.namespace),
            use_global_arguments=False,
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )

        self._rm = robot_manager
        self._env: EnvironmentManager = robot_manager._environment_manager
        self._near_miss_margin = near_miss_margin
        self._default_peds_radius = default_pedestrian_radius
        self._peds_msg: arena_people_msgs.msg.Pedestrians | None = None

        self._poly_cache: dict[str, dict] = {}
        for name, spec in polygons.items():
            if spec.type == 'polygon':
                if spec.points is None or len(spec.points) < 3:
                    self.get_logger().warning(f'collision_tracker: skipping polygon {name!r}: fewer than 3 points')
                    continue
                self._poly_cache[name] = {
                    'type': 'polygon',
                    'shapely': shapely.Polygon(spec.points),
                    'action_code': _ACTION_CODES.get(spec.action_type, 0),
                }
            else:
                r_c = spec.radius if spec.radius is not None else 0.0
                self._poly_cache[name] = {
                    'type': 'circle',
                    'R_c': r_c,
                    'action_code': _ACTION_CODES.get(spec.action_type, 0),
                }

        self._footprint_base: shapely.Polygon | None = None
        if footprint is not None and len(footprint) >= 3:
            self._footprint_base = shapely.Polygon(footprint)
        # None while a reset is in flight
        self._valid_after: rclpy.time.Time | None = None

        self._pub_state = self.create_publisher(nav2_msgs.msg.CollisionMonitorState, 'collision_monitor_state', 10)
        self._pub_events = self.create_publisher(arena_robots_msgs.msg.CollisionEvents, 'collision_events', 10)
        env_ns = Namespace(robot_manager.node.get_namespace())
        self._sub_peds = self.create_subscription(
            arena_people_msgs.msg.Pedestrians,
            str(env_ns('arena_peds')),
            self._on_peds,
            10,
        )
        self._sub_reset_start = self.create_subscription(std_msgs.msg.Empty, str(env_ns('reset_start')), self._on_reset_start, 1)
        self._sub_reset_end = self.create_subscription(std_msgs.msg.Empty, str(env_ns('reset_end')), self._on_reset_end, 1)
        self._timer = self.create_timer(1.0 / rate_hz, self._tick)

    def _on_peds(self, msg: arena_people_msgs.msg.Pedestrians):
        self._peds_msg = msg

    def _on_reset_start(self, _msg: std_msgs.msg.Empty):
        self._valid_after = None

    def _on_reset_end(self, _msg: std_msgs.msg.Empty):
        self._peds_msg = None
        self._valid_after = self.get_clock().now()

    def _robot_polygon(self, entry: dict, rx: float, ry: float, rth: float) -> shapely.Polygon:
        if entry['type'] == 'polygon':
            poly = shapely.affinity.rotate(entry['shapely'], rth, origin=(0, 0), use_radians=True)
            return shapely.affinity.translate(poly, xoff=rx, yoff=ry)
        return shapely.Point(rx, ry).buffer(entry['R_c'])

    def _tick(self):
        stamped = self._rm.pose_stamped
        if stamped is None:
            return
        pose, stamp = stamped
        if self._valid_after is None or stamp <= self._valid_after:
            return

        rx, ry, rth = pose.to_2d()

        walls = self._env.walls_geometry
        statics = self._env.static_polygons

        events: list[arena_robots_msgs.msg.CollisionEvent] = []
        polygons_hit: dict[str, int] = {}

        for name, entry in self._poly_cache.items():
            robot_poly = self._robot_polygon(entry, rx, ry, rth)

            if not walls.is_empty and robot_poly.intersects(walls):
                ev = arena_robots_msgs.msg.CollisionEvent()
                ev.obstacle_id = '<wall>'
                ev.polygon_name = name
                ev.distance = 0.0
                ev.obstacle_position = geometry_msgs.msg.Point(x=rx, y=ry, z=0.0)
                events.append(ev)
                polygons_hit[name] = entry['action_code']

            for obs_name, poly in statics.items():
                if not robot_poly.intersects(poly):
                    continue
                centroid = poly.centroid
                ev = arena_robots_msgs.msg.CollisionEvent()
                ev.obstacle_id = obs_name
                ev.polygon_name = name
                ev.distance = 0.0
                ev.obstacle_position = geometry_msgs.msg.Point(x=float(centroid.x), y=float(centroid.y), z=0.0)
                events.append(ev)
                polygons_hit[name] = entry['action_code']

            if self._peds_msg is not None:
                for p in self._peds_msg.pedestrians:
                    px, py = p.pose.position.x, p.pose.position.y
                    if math.isnan(px) or math.isnan(py):
                        continue
                    ped_disc = shapely.Point(px, py).buffer(self._default_peds_radius)
                    if not robot_poly.intersects(ped_disc):
                        continue
                    ev = arena_robots_msgs.msg.CollisionEvent()
                    ev.obstacle_id = p.name
                    ev.polygon_name = name
                    ev.distance = 0.0
                    ev.obstacle_position = geometry_msgs.msg.Point(x=px, y=py, z=0.0)
                    events.append(ev)
                    polygons_hit[name] = entry['action_code']

        state = nav2_msgs.msg.CollisionMonitorState()
        if polygons_hit:
            name, code = min(polygons_hit.items(), key=lambda kv: (1 if kv[1] == 0 else 0, kv[1]))
            state.polygon_name = name
            state.action_type = code
        else:
            state.polygon_name = ''
            state.action_type = 0
        self._pub_state.publish(state)

        evmsg = arena_robots_msgs.msg.CollisionEvents()
        evmsg.header.stamp = self.get_clock().now().to_msg()
        evmsg.header.frame_id = 'map'
        evmsg.events = events
        self._pub_events.publish(evmsg)

        if self._footprint_base is not None and self._rm.node.rosparam[bool].get_unsafe('fail_on_collision'):
            footprint = shapely.affinity.translate(
                shapely.affinity.rotate(self._footprint_base, rth, origin=(0, 0), use_radians=True),
                xoff=rx,
                yoff=ry,
            )
            hit = (not walls.is_empty and footprint.intersects(walls)) or any(footprint.intersects(p) for p in statics.values())
            if not hit and self._peds_msg is not None:
                for p in self._peds_msg.pedestrians:
                    px, py = p.pose.position.x, p.pose.position.y
                    if math.isnan(px) or math.isnan(py):
                        continue
                    if footprint.intersects(shapely.Point(px, py).buffer(self._default_peds_radius)):
                        hit = True
                        break
            if hit:
                self._rm.node.fail_episode('collision')

    def shutdown(self):
        self.destroy_node()
