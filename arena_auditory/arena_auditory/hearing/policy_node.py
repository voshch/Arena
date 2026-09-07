"""Listen-then-yield: the single writer of the Nav2 SpeedFilter mask for one robot.

Composes three layers by lowest nonzero percentage:

  1. belief: the decaying pedestrian belief from ``hearing_belief``, dilated and thresholded
     (``belief_grid.speed_mask_from_belief``), so the robot slows near likely mass;
  2. listen: ``listen_mps`` on the approach to a blind bend along the global plan, so the
     robot's own drivetrain noise drops enough to hear around the corner before it reaches it;
  3. hold: ``hold_mps`` on a band before the bend while yielding, a creep rather than a stop
     since Nav2 reads a 0 % mask as "no limit".

Yield engages when the fraction of belief mass inside a disc around the bend exceeds
``yield_fraction`` while the robot is on the approach, and releases once the mass has moved
behind the robot (after ``min_yield_s`` of holding), once it has faded or the received level
has been falling for ``recede_s`` (each only after ``min_yield_s`` of silence, a pedestrian who
stopped is still there), or on ``yield_timeout_s``. After a release the robot goes
through the bend at listen speed, never full speed, so a pedestrian who stopped to yield in
turn is not driven into. Bends are map-frame points with hysteresis, so a replanned path does
not re-arm the same corner. State goes out on ``hearing/policy_state`` as JSON.
"""

from __future__ import annotations

import json
import math

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Bool, ColorRGBA, String
from task_generator_msgs.msg import HeardSoundEvent, RobotFleet
from visualization_msgs.msg import Marker, MarkerArray

from arena_auditory.hearing.belief_grid import BeliefParams, speed_mask_from_belief
from arena_auditory.hearing.belief_node import latched_qos, transient_event_qos, yaw_from_quat
from arena_auditory.hearing.corners import BlindBend, find_blind_bend
from arena_auditory.hearing.fleet import RobotBinding, bind_robot
from arena_auditory.hearing.policy import State, YieldMachine, YieldParams, compose_masks, mass_split, paint_lane
from arena_auditory.lockstep import register_hard_channel


class PolicyNode(Node):
    def __init__(self, **kwargs: object) -> None:
        super().__init__("hearing_policy", **kwargs)
        d = self.declare_parameter
        d("robot_fleet_topic", "")
        d("robot", "")
        d("base_frame", "")
        d("max_linear_mps", 0.0)  # explicit; 0 = from the robot's mobile cap via the fleet
        d("heard_sound_suffix", "heard_sound")
        d("plan_suffix", "plan")
        d("belief_topic", "hearing/belief_grid")
        d("map_topic", "/map")
        d("reset_topic", "")
        d("speed_mask_topic", "hearing/speed_filter_mask")
        d("state_topic", "hearing/policy_state")
        d("marker_topic", "hearing/policy_markers")
        d("map_frame", "map")
        d("publish_rate_hz", 10.0)
        d("occupied_threshold", 50)
        d("listen_enabled", True)
        d("yield_enabled", True)
        d("listen_mps", 0.2)
        d("hold_mps", 0.03)
        d("lookahead_m", 4.0)
        d("approach_m", 4.0)
        d("hold_len_m", 1.5)
        d("hold_offset_m", 1.0)
        d("lane_radius_m", 0.6)
        d("corner_radius_m", 2.0)
        d("bend_hysteresis_m", 1.0)
        d("rearm_after_m", 3.0)
        d("yield_fraction", 0.5)
        d("release_fraction", 0.25)
        d("min_yield_s", 3.0)
        d("yield_timeout_s", 15.0)
        d("recede_s", 2.0)
        d("level_trend_db_per_s", -1.0)  # received level falling faster than this counts as receding
        d("belief_threshold", 0.6)
        d("speed_min_pct", 40)
        d("speed_free_pct", 100)
        d("reaction_radius_m", 2.0)

        g = self.get_parameter
        self._belief_params = BeliefParams(
            belief_threshold=float(g("belief_threshold").value),
            speed_min_pct=int(g("speed_min_pct").value),
            speed_free_pct=int(g("speed_free_pct").value),
            reaction_radius_m=float(g("reaction_radius_m").value),
        )
        self._machine = YieldMachine(
            YieldParams(
                yield_fraction=float(g("yield_fraction").value),
                release_fraction=float(g("release_fraction").value),
                min_yield_s=float(g("min_yield_s").value),
                yield_timeout_s=float(g("yield_timeout_s").value),
                recede_s=float(g("recede_s").value),
            )
        )
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._binding: RobotBinding | None = None
        self._base_frame = str(g("base_frame").value)
        self._max_mps = float(g("max_linear_mps").value)
        self._map: OccupancyGrid | None = None
        self._occupied: np.ndarray | None = None
        self._belief: OccupancyGrid | None = None
        self._plan_xy: np.ndarray | None = None
        self._plan_dirty = False
        self._bend: BlindBend | None = None
        self._consumed_xy: tuple[float, float] | None = None
        self._consumed_at: tuple[float, float] | None = None
        self._last_event_t = -1e9
        self._level_ema: float | None = None
        self._level_slope = 0.0
        self._level_t = 0.0
        self._receding_since: float | None = None
        self._time_yielding = 0.0
        self._last_tick: float | None = None

        self._pub_mask = self.create_publisher(OccupancyGrid, str(g("speed_mask_topic").value), latched_qos())
        self._pub_state = self.create_publisher(String, str(g("state_topic").value), latched_qos())
        self._pub_markers = self.create_publisher(MarkerArray, str(g("marker_topic").value), 1)
        self.create_subscription(OccupancyGrid, str(g("map_topic").value), self._cb_map, latched_qos())
        self.create_subscription(OccupancyGrid, str(g("belief_topic").value), self._cb_belief, 1)
        reset_topic = str(g("reset_topic").value)
        if reset_topic:
            self.create_subscription(Bool, reset_topic, self._cb_reset, latched_qos())
        self.create_subscription(RobotFleet, str(g("robot_fleet_topic").value), self._cb_fleet, latched_qos())
        self._rate = max(float(g("publish_rate_hz").value), 0.1)
        self.create_timer(1.0 / self._rate, self._on_timer)
        self.get_logger().info(f"hearing_policy up: listen={bool(g('listen_enabled').value)} yield={bool(g('yield_enabled').value)}")

    def _cb_fleet(self, msg: RobotFleet) -> None:
        if self._binding is not None:
            return
        binding = bind_robot(msg, str(self.get_parameter("robot").value), str(self.get_parameter("robot_fleet_topic").value))
        if binding is None:
            return
        self._binding = binding
        if not self._base_frame:
            self._base_frame = binding.base_frame
        if self._max_mps <= 0.0:
            self._max_mps = binding.max_linear_mps
        prefix = f"{binding.tg_node}/{binding.name}"
        self.create_subscription(Path, f"{prefix}/{self.get_parameter('plan_suffix').value}", self._cb_plan, 1)
        self.create_subscription(HeardSoundEvent, f"{prefix}/{self.get_parameter('heard_sound_suffix').value}", self._cb_event, transient_event_qos())
        self.get_logger().info(f"hearing_policy bound to {binding.name}: base {self._base_frame}, max {self._max_mps:.2f} m/s")
        if bool(self.get_parameter("use_sim_time").value):
            register_hard_channel(
                self,
                name=f"policy/{binding.name}",
                topic=self._pub_mask.topic_name,
                msg_type="nav_msgs/msg/OccupancyGrid",
                period_s=1.1 / self._rate,
                env=self.resolve_topic_name(binding.tg_node),
            )

    def _cb_map(self, msg: OccupancyGrid) -> None:
        self._map = msg
        data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self._occupied = data >= int(self.get_parameter("occupied_threshold").value)

    def _cb_belief(self, msg: OccupancyGrid) -> None:
        self._belief = msg

    def _cb_plan(self, msg: Path) -> None:
        self._plan_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float64) if msg.poses else None
        self._plan_dirty = True

    def _cb_reset(self, msg: Bool) -> None:
        if msg.data:
            self._bend = None
            self._consumed_xy = None
            self._machine.new_bend()
            self._machine.state = State.CRUISE
            self._level_ema = None
            self._receding_since = None
            self._time_yielding = 0.0

    def _cb_event(self, msg: HeardSoundEvent) -> None:
        if not bool(msg.audible):
            return
        t = self._now()
        level = float(msg.received_volume_db)
        if not math.isfinite(level):
            return
        if self._level_ema is None:
            self._level_ema, self._level_slope = level, 0.0
        else:
            dt = max(t - self._level_t, 1e-3)
            ema = 0.7 * self._level_ema + 0.3 * level
            self._level_slope = 0.7 * self._level_slope + 0.3 * (ema - self._level_ema) / dt
            self._level_ema = ema
        self._level_t = t
        self._last_event_t = t

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _robot_pose(self) -> tuple[float, float, float] | None:
        try:
            tf = self._tf_buffer.lookup_transform(str(self.get_parameter("map_frame").value), self._base_frame, Time())
        except Exception:
            return None
        t = tf.transform.translation
        return float(t.x), float(t.y), yaw_from_quat(tf.transform.rotation)

    def _pct(self, mps: float) -> int:
        if self._max_mps <= 0.0:
            return 0
        return int(max(1, min(100, round(100.0 * mps / self._max_mps))))

    def _update_bend(self, robot_xy: tuple[float, float]) -> None:
        """Recompute on a new plan only. Bends are map-frame points: a replanned bend within the
        hysteresis is the same bend, a consumed one stays consumed until the robot moved on."""
        g = self.get_parameter
        if self._plan_xy is None or self._occupied is None or self._map is None:
            self._bend = None
            return
        if not self._plan_dirty:
            return
        self._plan_dirty = False
        info = self._map.info
        found = find_blind_bend(
            self._plan_xy,
            self._occupied,
            (info.origin.position.x, info.origin.position.y),
            float(info.resolution),
            robot_xy,
            lookahead_m=float(g("lookahead_m").value),
            approach_m=float(g("approach_m").value),
            hold_len_m=float(g("hold_len_m").value),
            hold_offset_m=float(g("hold_offset_m").value),
        )
        if found is None:
            self._bend = None
            return
        hyst = float(g("bend_hysteresis_m").value)
        if self._consumed_xy is not None and math.dist(found.bend_xy, self._consumed_xy) <= hyst:
            if self._consumed_at is not None and math.dist(robot_xy, self._consumed_at) >= float(g("rearm_after_m").value):
                self._consumed_xy = None
            else:
                self._bend = None
                return
        if self._bend is None or math.dist(found.bend_xy, self._bend.bend_xy) > hyst:
            self._machine.new_bend()
            self._level_ema = None
            self._receding_since = None
        self._bend = found

    def _on_timer(self) -> None:
        if self._belief is None or self._map is None or self._binding is None and not self._base_frame:
            return
        pose = self._robot_pose()
        if pose is None:
            return
        now = self._now()
        dt = 0.0 if self._last_tick is None else max(now - self._last_tick, 0.0)
        self._last_tick = now
        g = self.get_parameter
        robot_xy = (pose[0], pose[1])
        self._update_bend(robot_xy)

        info = self._belief.info
        origin = (info.origin.position.x, info.origin.position.y)
        res = float(info.resolution)
        belief = np.asarray(self._belief.data, dtype=np.float32).reshape(info.height, info.width) / 100.0
        layers = [speed_mask_from_belief(belief, res, self._belief_params)]
        shape = belief.shape

        state = State.CRUISE
        frac = ahead = behind = total = 0.0
        dist = float("nan")
        bend = self._bend
        if bend is not None:
            dist = bend.dist_from(self._plan_xy, robot_xy) if self._plan_xy is not None else float("nan")
            in_approach = dist <= float(g("approach_m").value)
            past = dist <= 0.3
            ahead, behind, total = mass_split(belief, origin, res, bend.bend_xy, float(g("corner_radius_m").value), robot_xy, bend.direction)
            frac = ahead / total if total > 0.0 else 0.0
            if self._level_slope < float(g("level_trend_db_per_s").value):
                if self._receding_since is None:
                    self._receding_since = now
            else:
                self._receding_since = None
            receding_s = now - self._receding_since if self._receding_since is not None else 0.0
            state = self._machine.step(now, in_approach=in_approach, past_bend=past, frac_ahead=frac, ahead=ahead, behind=behind, event_age_s=now - self._last_event_t, receding_s=receding_s)
            if not bool(g("yield_enabled").value) and state is State.YIELD:
                state = self._machine.state = State.LISTEN
            if past:
                self._consumed_xy, self._consumed_at = bend.bend_xy, robot_xy
                self._bend = None
            elif bool(g("listen_enabled").value) and state in (State.LISTEN, State.YIELD, State.PASS):
                layers.append(paint_lane(shape, origin, res, bend.approach_xy, float(g("lane_radius_m").value), self._pct(float(g("listen_mps").value))))
            if state is State.YIELD:
                layers.append(paint_lane(shape, origin, res, bend.hold_xy, float(g("lane_radius_m").value), self._pct(float(g("hold_mps").value))))
                self._time_yielding += dt

        mask = compose_masks(*layers)
        out = OccupancyGrid()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = str(g("map_frame").value)
        out.info = info
        out.data = mask.reshape(-1).tolist()
        self._pub_mask.publish(out)
        binding = np.zeros(shape, dtype=np.int8)
        if len(layers) > 1:
            binding = np.argmin(np.where(np.stack(layers) > 0, np.stack(layers), 127), axis=0).astype(np.int8)
        r, c = int((robot_xy[1] - origin[1]) / res), int((robot_xy[0] - origin[0]) / res)
        at_robot = int(mask[r, c]) if 0 <= r < shape[0] and 0 <= c < shape[1] else 0
        layer_at_robot = ("belief", "listen", "hold")[int(binding[r, c])] if at_robot and 0 <= r < shape[0] and 0 <= c < shape[1] else "none"
        self._pub_state.publish(String(data=json.dumps({
            "state": state.value,
            "dist_to_bend_m": None if math.isnan(dist) else round(dist, 2),
            "frac_ahead": round(frac, 3),
            "mass_ahead": round(ahead, 3),
            "mass_behind": round(behind, 3),
            "mass_total": round(total, 3),
            "level_slope_db_s": round(self._level_slope, 2),
            "limit_pct": at_robot,
            "binding_layer": layer_at_robot,
            "yield_count": self._machine.yield_count,
            "time_yielding_s": round(self._time_yielding, 2),
        })))
        self._publish_markers(bend, state)

    def _publish_markers(self, bend: BlindBend | None, state: State) -> None:
        arr = MarkerArray()
        frame = str(self.get_parameter("map_frame").value)
        stamp = self.get_clock().now().to_msg()
        for mid, (pts, color) in enumerate(((None if bend is None else bend.approach_xy, ColorRGBA(r=0.2, g=0.6, b=1.0, a=0.6)), (None if bend is None else bend.hold_xy, ColorRGBA(r=1.0, g=0.3, b=0.2, a=0.8)))):
            m = Marker()
            m.header.frame_id, m.header.stamp, m.ns, m.id = frame, stamp, "hearing_policy", mid
            if pts is None or len(pts) == 0 or (mid == 1 and state is not State.YIELD):
                m.action = Marker.DELETE
            else:
                m.type, m.action = Marker.LINE_STRIP, Marker.ADD
                m.scale.x = 0.08
                m.color = color
                m.pose.orientation.w = 1.0
                m.points = [Point(x=float(x), y=float(y), z=0.08) for x, y in pts]
            arr.markers.append(m)
        self._pub_markers.publish(arr)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
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
