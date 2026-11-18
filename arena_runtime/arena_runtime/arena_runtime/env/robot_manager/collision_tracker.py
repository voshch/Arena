from __future__ import annotations

from typing import TYPE_CHECKING

import arena_people_msgs.msg
import arena_robots_msgs.msg
import geometry_msgs.msg
import nav2_msgs.msg
import shapely
import shapely.affinity
import std_msgs.msg
import visualization_msgs.msg as viz
from arena_rclpy_mixins.lazy import LazyPublisher
from arena_rclpy_mixins.shared import Namespace
from arena_rclpy_mixins.Time import Time, TimeNode
from arena_robots.caps import PolygonSpec
from builtin_interfaces.msg import Duration
from rclpy.parameter import Parameter

from arena_runtime.env.collision_grid import Cell
from arena_runtime.humans.devices import shapes as pedestrian_shapes

if TYPE_CHECKING:
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.robot_manager.robot_manager import RobotManager

_ACTION_CODES: dict[str | None, int] = {
    None: 0,
    'stop': 1,
    'slowdown': 2,
    'approach': 3,
    'limit': 4,
}

_RED = std_msgs.msg.ColorRGBA(r=0.9, g=0.1, b=0.1, a=1.0)
_GREY = std_msgs.msg.ColorRGBA(r=0.55, g=0.6, b=0.65, a=1.0)
_DEVICE = std_msgs.msg.ColorRGBA(r=0.1, g=0.45, b=0.95, a=1.0)
_CONTACT = std_msgs.msg.ColorRGBA(r=0.95, g=0.1, b=0.1, a=0.6)

_CELL_KINDS: dict[Cell, str] = {
    Cell.MAP: arena_robots_msgs.msg.CollisionEvent.KIND_STATIC,
    Cell.WALL: arena_robots_msgs.msg.CollisionEvent.KIND_WALL,
    Cell.STATIC: arena_robots_msgs.msg.CollisionEvent.KIND_STATIC,
}


class CollisionTrackerNode(TimeNode):
    """Separate Node, same process, same executor as the env node.

    Reads RobotManager.pose + EnvironmentManager.collision_grid. The footprint
    defines a collision: it feeds `collision_events` and the robot manager's
    collision callback. Pedestrians are their body disc and held device on
    the ground. The cap polygons only feed `collision_monitor_state`. Tick
    timer uses sim time, so it is gated during pauses.
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        polygons: dict[str, PolygonSpec],
        *,
        rate_hz: float = 10.0,
        default_pedestrian_radius: float = 0.3,
        footprint: list[list[float]] | None = None,
    ):
        for name, spec in polygons.items():
            if spec.type == 'circle' and spec.radius is None:
                raise ValueError(f'collision_tracker: polygon {name!r}: circle without a radius')
        super().__init__(
            'collision_tracker',
            namespace=str(robot_manager.namespace),
            use_global_arguments=False,
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)],
        )

        self._rm = robot_manager
        self._env: EnvironmentManager = robot_manager._environment_manager
        self._default_peds_radius = default_pedestrian_radius
        self._peds_msg: arena_people_msgs.msg.Pedestrians | None = None
        self._shapes: list[tuple[arena_people_msgs.msg.Pedestrian, shapely.Geometry | None, shapely.Polygon | None]] | None = None

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
                self._poly_cache[name] = {
                    'type': 'circle',
                    'R_c': spec.radius,
                    'action_code': _ACTION_CODES.get(spec.action_type, 0),
                }

        self._footprint_base: shapely.Polygon | None = None
        if footprint is not None and len(footprint) >= 3:
            self._footprint_base = shapely.Polygon(footprint)
        # None while a reset is in flight
        self._valid_after: Time | None = None

        self._pub_state: LazyPublisher[nav2_msgs.msg.CollisionMonitorState] = LazyPublisher(self.create_publisher(nav2_msgs.msg.CollisionMonitorState, 'collision_monitor_state', 10))
        self._pub_events = None if self._footprint_base is None else self.create_publisher(arena_robots_msgs.msg.CollisionEvents, 'collision_events', 10)
        self._pub_markers: LazyPublisher[viz.MarkerArray] = LazyPublisher(self.create_publisher(viz.MarkerArray, 'collision_markers', 10))
        self._marker_ids: dict[tuple[str, str], int] = {}
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
        self._shapes = None

    def _on_reset_start(self, _msg: std_msgs.msg.Empty):
        self._valid_after = None

    def _on_reset_end(self, _msg: std_msgs.msg.Empty):
        self._peds_msg = None
        self._marker_ids.clear()
        self._shapes = None
        self._valid_after = self.sim_time

    def _robot_polygon(self, entry: dict, rx: float, ry: float, rth: float) -> shapely.Polygon:
        if entry['type'] == 'polygon':
            poly = shapely.affinity.rotate(entry['shapely'], rth, origin=(0, 0), use_radians=True)
            return shapely.affinity.translate(poly, xoff=rx, yoff=ry)
        return shapely.Point(rx, ry).buffer(entry['R_c'])

    def _monitor_state(self, rx: float, ry: float, rth: float) -> nav2_msgs.msg.CollisionMonitorState:
        grid = self._env.collision_grid
        polygons_hit: dict[str, int] = {}
        for name, entry in self._poly_cache.items():
            robot_poly = self._robot_polygon(entry, rx, ry, rth)
            if (grid is not None and grid.hit(robot_poly) is not None) or self._hits_pedestrian(robot_poly):
                polygons_hit[name] = entry['action_code']

        state = nav2_msgs.msg.CollisionMonitorState()
        if polygons_hit:
            name, code = min(polygons_hit.items(), key=lambda kv: (1 if kv[1] == 0 else 0, kv[1]))
            state.polygon_name = name
            state.action_type = code
        else:
            state.polygon_name = ''
            state.action_type = 0
        return state

    def _tick(self):
        stamped = self._rm.pose_stamped
        if stamped is None:
            return
        pose, stamp = stamped
        if self._valid_after is None or Time.from_rclpy(stamp) <= self._valid_after:
            return

        rx, ry, rth = pose.to_2d()

        grid = self._env.collision_grid
        statics = self._env.static_polygons

        self._pub_state.publish(lambda: self._monitor_state(rx, ry, rth))

        if self._footprint_base is None or self._pub_events is None:
            return

        footprint = shapely.affinity.translate(
            shapely.affinity.rotate(self._footprint_base, rth, origin=(0, 0), use_radians=True),
            xoff=rx,
            yoff=ry,
        )
        events: list[arena_robots_msgs.msg.CollisionEvent] = []

        hit = grid.hit(footprint) if grid is not None else None
        if hit is not None:
            cell, obstacle_id = hit
            static = statics.get(obstacle_id)
            ev = arena_robots_msgs.msg.CollisionEvent()
            ev.kind = _CELL_KINDS[cell]
            ev.obstacle_id = obstacle_id
            ev.distance = 0.0
            ev.obstacle_position = geometry_msgs.msg.Point(x=rx, y=ry, z=0.0) if static is None else geometry_msgs.msg.Point(x=float(static.centroid.x), y=float(static.centroid.y), z=0.0)
            events.append(ev)

        for p, body, device in self._pedestrian_shapes():
            if body is not None and footprint.intersects(body):
                events.append(self._event(arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN, p, p.pose.position.x, p.pose.position.y))
            if device is not None and footprint.intersects(device):
                touch = footprint.intersection(device).centroid
                events.append(self._event(arena_robots_msgs.msg.CollisionEvent.KIND_DEVICE, p, touch.x, touch.y, p.device.kind))

        self._pub_markers.publish(lambda: self._markers(footprint, events))

        evmsg = arena_robots_msgs.msg.CollisionEvents()
        evmsg.header.stamp = self.sim_time.to_msg()
        evmsg.header.frame_id = 'map'
        evmsg.events = events
        self._pub_events.publish(evmsg)

        if events:
            self._rm.report_collision()

    def _markers(self, footprint: shapely.Polygon, events: list[arena_robots_msgs.msg.CollisionEvent]) -> viz.MarkerArray:
        """The robot footprint and the held devices on the ground, red while touched, and a red column at each contact."""
        stamp = self.sim_time.to_msg()
        touched = {ev.obstacle_id for ev in events if ev.kind == arena_robots_msgs.msg.CollisionEvent.KIND_DEVICE}

        def marker(ns: str, mid: int, kind: int, scale: tuple[float, float, float]) -> viz.Marker:
            m = viz.Marker(ns=ns, id=mid, type=kind, action=viz.Marker.ADD)
            m.header.frame_id, m.header.stamp = 'map', stamp
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = scale
            return m

        def ring(geom: shapely.Polygon) -> list[geometry_msgs.msg.Point]:
            coords = list(geom.exterior.coords)
            return [geometry_msgs.msg.Point(x=float(a[0]), y=float(a[1]), z=0.03) for pair in zip(coords, coords[1:], strict=False) for a in pair]

        robot = marker('footprint', 0, viz.Marker.LINE_LIST, (0.03, 0.0, 0.0))
        robot.points = ring(footprint)
        robot.color = _RED if events else _GREY
        devices = marker('devices', 0, viz.Marker.LINE_LIST, (0.025, 0.0, 0.0))
        points: list[geometry_msgs.msg.Point] = []
        colors: list[std_msgs.msg.ColorRGBA] = []
        for p, _, device in self._pedestrian_shapes():
            if device is not None:
                outline = ring(device)
                points.extend(outline)
                colors.extend([_RED if p.name in touched else _DEVICE] * len(outline))
        devices.points, devices.colors, devices.color = points, colors, _DEVICE
        if not devices.points:
            devices.action = viz.Marker.DELETE
        out = [robot, devices]
        for ev in events:
            if ev.kind not in (arena_robots_msgs.msg.CollisionEvent.KIND_DEVICE, arena_robots_msgs.msg.CollisionEvent.KIND_PEDESTRIAN):
                continue
            mid = self._marker_ids.setdefault((ev.kind, ev.obstacle_id), len(self._marker_ids))
            column = marker('contacts', mid, viz.Marker.CYLINDER, (0.16, 0.16, 1.2))
            column.pose.position = geometry_msgs.msg.Point(x=ev.obstacle_position.x, y=ev.obstacle_position.y, z=0.6)
            column.color = _CONTACT
            column.lifetime = Duration(sec=1)
            out.append(column)
        return viz.MarkerArray(markers=out)

    @staticmethod
    def _event(kind: str, ped: arena_people_msgs.msg.Pedestrian, x: float, y: float, device: str = "") -> arena_robots_msgs.msg.CollisionEvent:
        return arena_robots_msgs.msg.CollisionEvent(kind=kind, obstacle_id=ped.name, distance=0.0, obstacle_position=geometry_msgs.msg.Point(x=float(x), y=float(y), z=0.0), device=device)

    def _pedestrian_shapes(self) -> list[tuple[arena_people_msgs.msg.Pedestrian, shapely.Geometry | None, shapely.Polygon | None]]:
        """Body disc and held device of every pedestrian of the latest message, built once per message."""
        if self._shapes is None:
            peds = self._peds_msg.pedestrians if self._peds_msg is not None else []
            self._shapes = [(p, *pedestrian_shapes(p, self._default_peds_radius)) for p in peds]
        return self._shapes

    def _hits_pedestrian(self, poly: shapely.Polygon) -> bool:
        return any((body is not None and poly.intersects(body)) or (device is not None and poly.intersects(device)) for _, body, device in self._pedestrian_shapes())

    def shutdown(self):
        self.destroy_node()
