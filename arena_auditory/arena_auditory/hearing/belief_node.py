"""Directional-belief consumer of the auditory sound-event bus.

Subscribes to one listener's heard sound events, maintains a decaying
pedestrian-likelihood grid in the map frame (``belief_grid.BeliefGrid``), and
publishes

  1. the grid as a 0..100 ``OccupancyGrid`` (``hearing/belief_grid``), consumed by
     ``hearing_policy``, which turns it into the Nav2 ``SpeedFilter`` mask, and by RViz,
  2. optionally, a ``MarkerArray`` drawing the wedge of each event, fading
     with the grid's own decay so the fan shows what still carries mass.

Only (sound_type, bearing, received level, stamp) are read off the event.  The
simulator's ``HeardSoundEvent`` also carries ``source_position``, ``distance``
and ``source_agent_name``. Those are ground truth and this node never touches
them, so the layer is a fair stand-in for one fed by a real front-end.

Bearing convention
------------------
``bearing_frame`` says how to read ``bearing_rad``, and the consumer is the
side that rotates.  The simulator's ``HeardSoundEvent.bearing_rad`` is
map-frame by construction (``sound_propagation_node`` fills it with
``atan2(sy - ly, sx - lx)``, no yaw subtraction), so the live node defaults to
``map`` and paints the bearing as given.  A front-end output is robot-frame,
CCW-positive from +x, the convention of the dataset column
``bearing_robot_rad`` and of the DCASE azimuth, and is consumed with
``bearing_frame: robot``, which adds the robot's yaw.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import rclpy
import tf2_ros
from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Point, Quaternion
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from std_msgs.msg import Bool, ColorRGBA
from task_generator_msgs.msg import HeardSoundEvent, RobotFleet
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.hearing.belief_grid import BeliefGrid, BeliefParams
from arena_auditory.hearing.fleet import RobotBinding, bind_robot


def transient_event_qos(depth: int = 50) -> QoSProfile:
    """Matches task_generator.auditory.qos_profiles.transient_event_qos."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def latched_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def yaw_from_quat(q: Quaternion) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class BeliefNode(Node):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("hearing_belief", **kwargs)

        self.declare_parameter("heard_sound_topic", "")  # explicit; empty = <tg_node>/<robot>/<heard_sound_suffix> from the fleet
        self.declare_parameter("heard_sound_suffix", "heard_sound")
        self.declare_parameter("robot_fleet_topic", "")
        self.declare_parameter("robot", "")  # fleet entry to bind; empty = the first robot
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("occupied_threshold", 50)  # OccupancyGrid cells >= this are occupied, mass never lands there
        self.declare_parameter("reset_topic", "")  # Bool, True clears the grid (episode reset)
        self.declare_parameter("belief_topic", "hearing/belief_grid")
        self.declare_parameter("marker_topic", "hearing/belief_wedges")
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("marker_range_m", 4.0)  # drawn wedge length; the grid keeps max_range_m
        self.declare_parameter("marker_max", 12)  # newest wedges kept on screen
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "")  # explicit; empty = <robot frame prefix>/<model base_frame> from the fleet
        self.declare_parameter("bearing_frame", "map")  # 'robot' | 'map'
        self.declare_parameter("use_listener_position_fallback", True)
        self.declare_parameter("publish_rate_hz", 5.0)

        # grid geometry: 0.0 resolution -> inherit the map's
        self.declare_parameter("resolution", 0.0)
        self.declare_parameter("standalone_grid", False)
        self.declare_parameter("standalone_origin", [-10.0, -10.0])
        self.declare_parameter("standalone_size", [20.0, 20.0])

        # belief model
        self.declare_parameter("wedge_deg", 10.0)
        self.declare_parameter("min_half_width_m", 0.3)
        self.declare_parameter("tau_sec", 2.0)
        self.declare_parameter("max_range_m", 15.0)
        self.declare_parameter("min_range_m", 0.5)
        self.declare_parameter("reference_distance_m", 1.0)
        self.declare_parameter("use_level_range", False)
        self.declare_parameter("range_sigma_frac", 0.35)
        self.declare_parameter("range_floor_weight", 0.15)
        self.declare_parameter("event_mass", 1.0)
        # Belief 1.0 = the steady state of one source emitting at this rate.
        # 2 Hz is a walking pedestrian's footstep cadence on the Arena bus. A
        # per-frame front-end (SELDnet, one event per 100 ms) needs 10.0.
        self.declare_parameter("nominal_event_rate_hz", 2.0)
        self.declare_parameter("mass_full_scale", 0.0)  # >0 overrides the derived scale
        self.declare_parameter("emission_db_types", ["footstep", "greeting", "speech"])
        self.declare_parameter("emission_db_values", [45.0, 60.0, 60.0])
        self.declare_parameter("sound_types", ["footstep", "greeting", "speech"])
        self.declare_parameter("min_received_db", -1e9)

        self._params = self._read_belief_params()
        self._grid: BeliefGrid | None = None
        self._map_info = None
        self._last_update = self.get_clock().now()
        self._wedges: deque[tuple[int, int, float, float, float, float, str]] = deque()  # id, t_ns, x, y, theta, range, type
        self._next_wedge_id = 0
        self._retired: list[int] = []
        self._accepted = 0
        self._dropped_no_pose = 0
        self._dropped_no_bearing = 0
        self._binding: RobotBinding | None = None

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._pub_belief = self.create_publisher(OccupancyGrid, str(self.get_parameter("belief_topic").value), 1)
        self._pub_markers = None
        if bool(self.get_parameter("publish_markers").value):
            self._pub_markers = self.create_publisher(MarkerArray, str(self.get_parameter("marker_topic").value), 1)

        self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter("map_topic").value),
            self._cb_map,
            latched_qos(),
        )

        reset_topic = str(self.get_parameter("reset_topic").value)
        if reset_topic:
            self.create_subscription(Bool, reset_topic, self._cb_reset, latched_qos())

        self._base_frame = str(self.get_parameter("base_frame").value)
        heard_topic = str(self.get_parameter("heard_sound_topic").value)
        if heard_topic:
            self._bind_events(heard_topic)
        else:
            self.create_subscription(RobotFleet, str(self.get_parameter("robot_fleet_topic").value), self._cb_fleet, latched_qos())

        if bool(self.get_parameter("standalone_grid").value):
            self._build_standalone_grid()

        rate = max(float(self.get_parameter("publish_rate_hz").value), 0.1)
        self.create_timer(1.0 / rate, self._on_timer)
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.get_logger().info(f"hearing_belief up: bearing_frame={self.get_parameter('bearing_frame').value}")

    def _bind_events(self, topic: str) -> None:
        self.create_subscription(HeardSoundEvent, topic, self._cb_event, transient_event_qos())
        self.get_logger().info(f"hearing_belief: events on {topic!r}, robot frame {self._base_frame!r}")

    def _cb_fleet(self, msg: RobotFleet) -> None:
        if self._binding is not None:
            return
        binding = bind_robot(msg, str(self.get_parameter("robot").value), str(self.get_parameter("robot_fleet_topic").value))
        if binding is None:
            return
        self._binding = binding
        if not self._base_frame:
            self._base_frame = binding.base_frame
        self._bind_events(f"{binding.tg_node}/{binding.name}/{self.get_parameter('heard_sound_suffix').value}")

    def _read_belief_params(self) -> BeliefParams:
        g = self.get_parameter
        types = [str(t).strip().lower() for t in g("emission_db_types").value]
        vals = [float(v) for v in g("emission_db_values").value]
        emission = dict(zip(types, vals, strict=True))
        return BeliefParams(
            wedge_deg=float(g("wedge_deg").value),
            min_half_width_m=float(g("min_half_width_m").value),
            tau_sec=float(g("tau_sec").value),
            max_range_m=float(g("max_range_m").value),
            min_range_m=float(g("min_range_m").value),
            reference_distance_m=float(g("reference_distance_m").value),
            use_level_range=bool(g("use_level_range").value),
            range_sigma_frac=float(g("range_sigma_frac").value),
            range_floor_weight=float(g("range_floor_weight").value),
            event_mass=float(g("event_mass").value),
            nominal_event_rate_hz=float(g("nominal_event_rate_hz").value),
            mass_full_scale=float(g("mass_full_scale").value),
            emission_db=emission,
        )

    def _on_set_parameters(self, _params: list[Parameter]) -> SetParametersResult:
        # applied on the next tick, the grid keeps its geometry
        try:
            self._params = self._read_belief_params()
            if self._grid is not None:
                self._grid.params = self._params
        except Exception as exc:  # pragma: no cover
            return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def _build_standalone_grid(self) -> None:
        res = float(self.get_parameter("resolution").value) or 0.1
        ox, oy = [float(v) for v in self.get_parameter("standalone_origin").value]
        sx, sy = [float(v) for v in self.get_parameter("standalone_size").value]
        self._grid = BeliefGrid(ox, oy, res, int(sx / res), int(sy / res), self._params)
        self.get_logger().info(f"standalone belief grid {self._grid.width}x{self._grid.height} @ {res} m")

    def _cb_reset(self, msg: Bool) -> None:
        if msg.data and self._grid is not None:
            self._grid.clear()
            self._retired.extend(w[0] for w in self._wedges)
            self._wedges.clear()

    def _cb_map(self, msg: OccupancyGrid) -> None:
        if bool(self.get_parameter("standalone_grid").value):
            return
        info = msg.info
        if self._map_info is not None and (
            info.width == self._map_info.width and info.height == self._map_info.height and abs(info.resolution - self._map_info.resolution) < 1e-9 and abs(info.origin.position.x - self._map_info.origin.position.x) < 1e-9 and abs(info.origin.position.y - self._map_info.origin.position.y) < 1e-9
        ):
            return
        self._map_info = info
        threshold = int(self.get_parameter("occupied_threshold").value)
        data = np.asarray(msg.data, dtype=np.int16).reshape(info.height, info.width)
        native_free = data < threshold
        res = float(self.get_parameter("resolution").value)
        if res <= 0.0:
            self._grid = BeliefGrid.from_occupancy_info(info, self._params, free=native_free)
        else:
            w = max(int(round(info.width * info.resolution / res)), 1)
            h = max(int(round(info.height * info.resolution / res)), 1)
            scale = res / info.resolution
            rows = np.minimum((np.arange(h) * scale).astype(int), info.height - 1)
            cols = np.minimum((np.arange(w) * scale).astype(int), info.width - 1)
            free = native_free[rows][:, cols]
            self._grid = BeliefGrid(info.origin.position.x, info.origin.position.y, res, w, h, self._params, free=free)
        self.get_logger().info(f"belief grid {self._grid.width}x{self._grid.height} @ {self._grid.resolution} m origin ({self._grid.origin_x:.2f}, {self._grid.origin_y:.2f}) frame {msg.header.frame_id!r}")

    def _robot_pose(self, stamp: TimeMsg) -> tuple[float, float, float] | None:
        map_frame = str(self.get_parameter("map_frame").value)
        base_frame = self._base_frame
        for when in (Time.from_msg(stamp), Time()):
            try:
                tf = self._tf_buffer.lookup_transform(map_frame, base_frame, when)
            except Exception:
                continue
            t = tf.transform.translation
            return float(t.x), float(t.y), yaw_from_quat(tf.transform.rotation)
        return None

    def _cb_event(self, msg: HeardSoundEvent) -> None:
        if self._grid is None:
            return
        wanted = {str(s).strip().lower() for s in self.get_parameter("sound_types").value}
        stype = str(msg.sound_type).strip().lower()
        if wanted and stype not in wanted:
            return
        if not bool(msg.audible):
            return
        received = float(msg.received_volume_db)
        bearing = float(msg.bearing_rad)
        if not math.isfinite(bearing):
            self._dropped_no_bearing += 1
            return
        if received < float(self.get_parameter("min_received_db").value):
            return

        pose = self._robot_pose(msg.header.stamp)
        if pose is None and bool(self.get_parameter("use_listener_position_fallback").value):
            # the rig's own reported location, not ground truth about a source;
            # carries no yaw, so only sound with bearing_frame='map'
            pose = (float(msg.listener_position.x), float(msg.listener_position.y), 0.0)
        if pose is None:
            self._dropped_no_pose += 1
            if self._dropped_no_pose % 50 == 1:
                self.get_logger().warning(f"no {self.get_parameter('map_frame').value} -> {self._base_frame} transform, dropped {self._dropped_no_pose} events")
            return

        x, y, yaw = pose
        info = self._grid.add_event(
            robot_x=x,
            robot_y=y,
            robot_yaw=yaw,
            bearing_rad=bearing,
            sound_type=stype,
            received_db=received,
            bearing_frame=str(self.get_parameter("bearing_frame").value),
        )
        self._accepted += 1
        self._wedges.append((self._next_wedge_id, self.get_clock().now().nanoseconds, x, y, info["theta"], info["range_m"], stype))
        self._next_wedge_id += 1

    def _on_timer(self) -> None:
        if self._grid is None:
            return
        now = self.get_clock().now()
        dt = (now - self._last_update).nanoseconds * 1e-9
        self._last_update = now
        self._grid.decay(dt)

        stamp = now.to_msg()
        frame = str(self.get_parameter("map_frame").value)
        self._pub_belief.publish(self._as_grid(self._grid.belief_int8(), stamp, frame))
        if self._pub_markers is not None:
            self._pub_markers.publish(self._wedge_markers(now.nanoseconds, stamp, frame))

    def _as_grid(self, data: np.ndarray, stamp: TimeMsg, frame: str) -> OccupancyGrid:
        g = OccupancyGrid()
        g.header.stamp = stamp
        g.header.frame_id = frame
        g.info.resolution = self._grid.resolution
        g.info.width = self._grid.width
        g.info.height = self._grid.height
        g.info.origin.position.x = self._grid.origin_x
        g.info.origin.position.y = self._grid.origin_y
        g.info.origin.orientation.w = 1.0
        g.data = data.reshape(-1).tolist()
        return g

    _WEDGE_EXPIRE = 1.5  # in units of tau

    def _wedge_markers(self, now_ns: int, stamp: TimeMsg, frame: str) -> MarkerArray:
        """Every live wedge under a stable id, faded by exp(-age / tau), DELETE for the ones that expired."""
        arr = MarkerArray()
        tau = max(float(self._params.tau_sec), 1e-6)
        half = math.radians(float(self._params.wedge_deg)) * 0.5
        keep = max(int(self.get_parameter("marker_max").value), 1)
        while self._wedges and ((now_ns - self._wedges[0][1]) * 1e-9 > self._WEDGE_EXPIRE * tau or len(self._wedges) > keep):
            self._retired.append(self._wedges.popleft()[0])
        draw_range = float(self.get_parameter("marker_range_m").value)
        for wid in self._retired:
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = "hearing_wedge"
            m.id = wid
            m.action = Marker.DELETE
            arr.markers.append(m)
        self._retired.clear()
        for wid, t_ns, x, y, theta, full_range, stype in self._wedges:
            rng = min(full_range, draw_range)
            fade = math.exp(-max(now_ns - t_ns, 0) * 1e-9 / tau)
            m = Marker()
            m.header.frame_id = frame
            m.header.stamp = stamp
            m.ns = "hearing_wedge"
            m.id = wid
            m.type = Marker.TRIANGLE_LIST
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color = ColorRGBA(
                r=1.0 if stype == "footstep" else 0.2,
                g=0.8 if stype == "footstep" else 0.75,
                b=0.2 if stype == "footstep" else 1.0,
                a=float(0.35 * fade),
            )
            steps = 8
            for k in range(steps):
                a0 = theta - half + 2.0 * half * k / steps
                a1 = theta - half + 2.0 * half * (k + 1) / steps
                m.points.append(Point(x=x, y=y, z=0.05))
                m.points.append(Point(x=x + rng * math.cos(a0), y=y + rng * math.sin(a0), z=0.05))
                m.points.append(Point(x=x + rng * math.cos(a1), y=y + rng * math.sin(a1), z=0.05))
            arr.markers.append(m)
        return arr

def main() -> None:
    rclpy.init()
    node = BeliefNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
