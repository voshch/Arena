#!/usr/bin/env python3
"""ROS4HRI producer node: projects Arena pedestrians into the REP-155 /humans/... representation.

Publishes hri_msgs tracks, per-person metadata, per-body joint_states, and TF
for each body frame.  Drives a pool of robot_state_publisher subprocesses
(rviz_utils.hri.body_pool) so hri_rviz can render animated skeletons.
"""

from __future__ import annotations

import math

import rclpy
from arena_people_msgs.msg import FaceView, Pedestrian, Pedestrians
from arena_rclpy_mixins.lazy import LazyPublisher, LazySubscription
from arena_rclpy_mixins.spin import create_executor, spin_node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from hri_msgs.msg import EngagementLevel, IdsList
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Bool, Float32, String
from task_generator.simulators.human.gait import GaitGenerator
from tf2_msgs.msg import TFMessage

from rviz_utils.hri import BodyPool
from rviz_utils.hri.face import GAZE_IN_FACE, FaceChain, matrix_from_pose, pinhole_k, quat_from_matrix, root_drop
from rviz_utils.hri.rig import semantic_to_rig

_DEFAULT_HEIGHT = 1.65
_VIEW_SIZE = 512

_PEDS_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)

_LATCHED_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

_LIVE_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


def _engagement_level(animation_state: int) -> EngagementLevel:
    """Map Pedestrian.animation_state to hri_msgs/EngagementLevel."""
    msg = EngagementLevel()
    if animation_state in (Pedestrian.WALKING, Pedestrian.RUNNING):
        msg.level = EngagementLevel.ENGAGED
    elif animation_state == Pedestrian.IDLE:
        msg.level = EngagementLevel.DISENGAGED
    else:
        msg.level = EngagementLevel.ENGAGING
    return msg


class HriProducer(Node):
    """Projects arena_people_msgs/Pedestrians into the REP-155 /humans/... API."""

    def __init__(self) -> None:
        super().__init__("hri_producer")

        self._ns = self.get_namespace().rstrip("/")

        env_tag = self._ns.rsplit("/", 1)[-1] or "env"
        self._body_prefix = f"{env_tag}_agent_"

        self.declare_parameter("max_bodies", 32)
        max_bodies: int = self.get_parameter("max_bodies").value
        self.declare_parameter("view.fov_deg", 60.0)
        self.declare_parameter("view.clip_near", 0.15)
        self._view_k = pinhole_k(math.radians(self.get_parameter("view.fov_deg").value), _VIEW_SIZE)
        self._view_clip_near: float = self.get_parameter("view.clip_near").value
        self._humans_ns = f"{self._ns}/humans"

        self._tf_pub: LazyPublisher[TFMessage] = LazyPublisher(self.create_publisher(TFMessage, f"{self._humans_ns}/tf", 100))
        self._env_tf_remapped = self.resolve_topic_name("/tf") != "/tf"
        self._world_node = Node(f"{self.get_name()}_world_tf", namespace=self.get_namespace(), use_global_arguments=False, start_parameter_services=False)
        self._world_tf_pub = self._world_node.create_publisher(TFMessage, "/tf", 100)
        self._faces_tracked_pub: LazyPublisher[IdsList] = LazyPublisher(self.create_publisher(IdsList, f"{self._humans_ns}/faces/tracked", _LIVE_QOS))
        self._face_view_pub: dict[str, LazyPublisher[FaceView]] = {}
        self._face_chain: dict[str, FaceChain] = {}
        self._root_drop: dict[str, float] = {}
        self._pool = BodyPool(self, self._humans_ns, max_bodies=max_bodies)
        self._gait = GaitGenerator()
        self._prev_stamp_sec: dict[str, float] = {}

        self._bodies_tracked_pub = self.create_publisher(
            IdsList, f"{self._humans_ns}/bodies/tracked", _LATCHED_QOS
        )
        self._persons_tracked_pub = self.create_publisher(
            IdsList, f"{self._humans_ns}/persons/tracked", _LATCHED_QOS
        )

        self._body_js_pub: dict[str, rclpy.publisher.Publisher] = {}
        self._body_urdf_pub: dict[str, rclpy.publisher.Publisher] = {}
        self._person_body_id_pub: dict[str, rclpy.publisher.Publisher] = {}
        self._person_anon_pub: dict[str, rclpy.publisher.Publisher] = {}
        self._person_conf_pub: dict[str, rclpy.publisher.Publisher] = {}
        self._person_eng_pub: dict[str, rclpy.publisher.Publisher] = {}

        self._peds_sub = LazySubscription(
            self,
            (self._tf_pub, self._faces_tracked_pub),
            Pedestrians,
            f"{self._ns}/arena_peds",
            self._on_peds,
            _PEDS_QOS,
        )

        self.get_logger().info(f"hri_producer ready, env ns={self._ns!r}")

    def _body_id(self, ped_id: int) -> str:
        return f"{self._body_prefix}{ped_id}"

    def _ped_id(self, body_id: str) -> int:
        return int(body_id.removeprefix(self._body_prefix))

    def _ensure_body_publishers(self, body_id: str) -> None:
        if body_id in self._body_js_pub:
            return
        self._body_js_pub[body_id] = self.create_publisher(
            JointState,
            f"{self._humans_ns}/bodies/{body_id}/joint_states",
            _LIVE_QOS,
        )
        self._face_view_pub[body_id] = LazyPublisher(
            self.create_publisher(FaceView, f"{self._humans_ns}/faces/{body_id}/view", _LIVE_QOS)
        )
        # libhri reads the body URDF from this latched topic, not the param.
        urdf_pub = self.create_publisher(
            String, f"{self._humans_ns}/bodies/{body_id}/urdf", _LATCHED_QOS
        )
        self._body_urdf_pub[body_id] = urdf_pub
        urdf = self._pool.urdf_for(body_id)
        if urdf is not None:
            urdf_pub.publish(String(data=urdf))

    def _destroy_body_publishers(self, body_id: str) -> None:
        for registry in (
            self._body_js_pub,
            self._body_urdf_pub,
            self._person_body_id_pub,
            self._person_anon_pub,
            self._person_conf_pub,
            self._person_eng_pub,
        ):
            pub = registry.pop(body_id, None)
            if pub is not None:
                self.destroy_publisher(pub)
        view = self._face_view_pub.pop(body_id, None)
        if view is not None:
            self.destroy_publisher(view.publisher)
        self._face_chain.pop(body_id, None)

    def _ensure_person_publishers(self, person_id: str) -> None:
        if person_id in self._person_body_id_pub:
            return
        base = f"{self._humans_ns}/persons/{person_id}"
        self._person_body_id_pub[person_id] = self.create_publisher(
            String, f"{base}/body_id", _LATCHED_QOS
        )
        self._person_anon_pub[person_id] = self.create_publisher(
            Bool, f"{base}/anonymous", _LATCHED_QOS
        )
        self._person_conf_pub[person_id] = self.create_publisher(
            Float32, f"{base}/location_confidence", _LIVE_QOS
        )
        self._person_eng_pub[person_id] = self.create_publisher(
            EngagementLevel, f"{base}/engagement_status", _LIVE_QOS
        )
        self._person_body_id_pub[person_id].publish(String(data=person_id))
        self._person_anon_pub[person_id].publish(Bool(data=False))

    def _on_peds(self, msg: Pedestrians) -> None:
        stamp = self.get_clock().now().to_msg()
        parent_frame = msg.header.frame_id

        current_ids: set[str] = {self._body_id(ped.id) for ped in msg.pedestrians}
        heights: dict[str, float] = {self._body_id(ped.id): _DEFAULT_HEIGHT for ped in msg.pedestrians}

        for bid in self._pool.active_ids() - current_ids:
            self._gait.forget(self._ped_id(bid))
            self._prev_stamp_sec.pop(bid, None)
            self._destroy_body_publishers(bid)

        self._pool.sync(current_ids, heights)

        ids_list = IdsList(ids=sorted(current_ids))
        self._bodies_tracked_pub.publish(ids_list)
        self._persons_tracked_pub.publish(ids_list)
        self._faces_tracked_pub.publish(lambda: ids_list)

        transforms: list[TransformStamped] = []
        faces: list[TransformStamped] = []

        for ped in msg.pedestrians:
            bid = self._body_id(ped.id)

            self._ensure_body_publishers(bid)
            self._ensure_person_publishers(bid)

            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = parent_frame
            tf.child_frame_id = f"body_{bid}"
            tf.transform.translation.x = ped.pose.position.x
            tf.transform.translation.y = ped.pose.position.y
            if ped.model_uri not in self._root_drop:
                self._root_drop[ped.model_uri] = root_drop(ped.model_uri)
            tf.transform.translation.z = ped.pose.position.z + self._pool.foot_offset(bid) + self._root_drop[ped.model_uri]
            tf.transform.rotation = ped.pose.orientation
            transforms.append(tf)

            if ped.name:
                alias = TransformStamped()
                alias.header.stamp = stamp
                alias.header.frame_id = f"body_{bid}"
                alias.child_frame_id = ped.name
                alias.transform.rotation.w = 1.0
                transforms.append(alias)

            if ped.joint_state.name:
                js = JointState()
                js.header.stamp = stamp
                js.header.frame_id = ""
                js.name = [f"{n}_{bid}" for n in ped.joint_state.name]
                js.position = semantic_to_rig(ped.joint_state.name, ped.joint_state.position)
                js.velocity = list(ped.joint_state.velocity)
                js.effort = list(ped.joint_state.effort)
                self._body_js_pub[bid].publish(js)
            else:
                now_sec = stamp.sec + stamp.nanosec * 1e-9
                prev_sec = self._prev_stamp_sec.get(bid)
                dt = (now_sec - prev_sec) if prev_sec is not None and now_sec > prev_sec else 0.1
                self._prev_stamp_sec[bid] = now_sec
                speed = math.hypot(ped.twist.linear.x, ped.twist.linear.y)
                angles = self._gait.compute(ped.id, ped.animation_state, speed, dt)
                bare_js = self._gait.joint_state(angles, stamp=stamp)
                js = JointState()
                js.header = bare_js.header
                js.name = [f"{n}_{bid}" for n in bare_js.name]
                js.position = semantic_to_rig(bare_js.name, bare_js.position)
                self._body_js_pub[bid].publish(js)

            view = self._face_view_pub[bid]
            if view.wanted:
                face = self._face_tf(bid, tf, dict(zip(js.name, js.position, strict=True)))
                if face is not None:
                    faces.extend(face)
                    view.publish(lambda bid=bid: FaceView(info=self._camera_info(bid, stamp), clip_near=self._view_clip_near))

            self._person_conf_pub[bid].publish(Float32(data=1.0))
            self._person_eng_pub[bid].publish(_engagement_level(ped.animation_state))

        self._tf_pub.publish(lambda: TFMessage(transforms=transforms + faces if self._env_tf_remapped else transforms))
        if faces:
            self._world_tf_pub.publish(TFMessage(transforms=faces))

    def _face_tf(self, bid: str, body: TransformStamped, positions: dict[str, float]) -> list[TransformStamped] | None:
        """face_<id> in the body's parent frame and gaze_<id> under it, None while the body has no URDF."""
        chain = self._face_chain.get(bid)
        if chain is None:
            urdf = self._pool.urdf_for(bid)
            if urdf is None:
                return None
            chain = self._face_chain[bid] = FaceChain.from_urdf(urdf, bid)
        t, q = body.transform.translation, body.transform.rotation
        world = matrix_from_pose((t.x, t.y, t.z), (q.w, q.x, q.y, q.z)) @ chain.pose(positions)
        face = TransformStamped()
        face.header = body.header
        face.child_frame_id = f"face_{bid}"
        face.transform.translation.x, face.transform.translation.y, face.transform.translation.z = (float(v) for v in world[:3, 3])
        r = face.transform.rotation
        r.w, r.x, r.y, r.z = quat_from_matrix(world)
        gaze = TransformStamped()
        gaze.header.stamp = body.header.stamp
        gaze.header.frame_id = face.child_frame_id
        gaze.child_frame_id = f"gaze_{bid}"
        g = gaze.transform.rotation
        g.w, g.x, g.y, g.z = GAZE_IN_FACE
        return [face, gaze]

    def _camera_info(self, bid: str, stamp: Time) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = f"gaze_{bid}"
        info.width = info.height = _VIEW_SIZE
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = self._view_k
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        f, c = self._view_k[0], self._view_k[2]
        info.p = [f, 0.0, c, 0.0, 0.0, f, c, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def destroy_node(self) -> None:
        self._pool.teardown()
        self._world_node.destroy_node()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    spin_node(HriProducer(), executor=create_executor())


if __name__ == "__main__":
    main()
