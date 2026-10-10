"""Per-robot odometry and TF publisher for MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geometry_msgs.msg
import nav_msgs.msg
import tf2_ros
from builtin_interfaces.msg import Time

from arena_mujoco import hooks
from arena_mujoco.context import get_scene_store

if TYPE_CHECKING:
    import mujoco

    from arena_mujoco.services.SpawnUrdf import RobotEntry


@dataclass
class _RobotOdom:
    env_id: int
    prim_name: str
    tf_prefix: str
    base_frame: str
    odom_frame_id: str
    localization: bool
    odom_pub: object | None
    last_pub_time: float = field(default=-1.0)


_entries: list[_RobotOdom] = []
_broadcaster: tf2_ros.TransformBroadcaster | None = None


def _make_stamp(sim_time: float) -> Time:
    t = Time()
    t.sec = int(sim_time)
    t.nanosec = int(round((sim_time - int(sim_time)) * 1e9))
    return t


def _post_spawn(
    env_id: int,
    name: str,
    robot_entry: RobotEntry,
    joint_names: list[str],
) -> None:
    node = hooks.get_node()

    odom_topic = robot_entry.odom_topic
    odom_frame_id = f'{robot_entry.tf_prefix}odom'

    odom_pub = node.create_publisher(nav_msgs.msg.Odometry, odom_topic, 10) if odom_topic else None
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = tf2_ros.TransformBroadcaster(node)

    _entries.append(
        _RobotOdom(
            env_id=env_id,
            prim_name=robot_entry.prim_name,
            tf_prefix=robot_entry.tf_prefix,
            base_frame=robot_entry.base_frame,
            odom_frame_id=odom_frame_id,
            localization=robot_entry.localization,
            odom_pub=odom_pub,
        )
    )


def _despawn(env_id: int, robot_entry: RobotEntry) -> None:
    node = hooks.get_node()
    for entry in [e for e in _entries if e.env_id == env_id and e.prim_name == robot_entry.prim_name]:
        if entry.odom_pub is not None:
            node.destroy_publisher(entry.odom_pub)
        _entries.remove(entry)


def _freejoint_twist(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (linear, angular) velocities from the freejoint dof, or zeros."""
    import mujoco as mj

    jnt_adr = int(model.body_jntadr[body_id])
    jnt_num = int(model.body_jntnum[body_id])
    if jnt_num != 1 or int(model.jnt_type[jnt_adr]) != mj.mjtJoint.mjJNT_FREE:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    dof_adr = int(model.jnt_dofadr[jnt_adr])
    vx, vy, vz = float(data.qvel[dof_adr]), float(data.qvel[dof_adr + 1]), float(data.qvel[dof_adr + 2])
    wx, wy, wz = float(data.qvel[dof_adr + 3]), float(data.qvel[dof_adr + 4]), float(data.qvel[dof_adr + 5])
    return (vx, vy, vz), (wx, wy, wz)


def _pump() -> None:
    import mujoco as mj

    store = get_scene_store()
    transforms: list[geometry_msgs.msg.TransformStamped] = []

    for entry in _entries:
        sim_time = store.get_clock_time(entry.env_id)

        if sim_time <= entry.last_pub_time:
            continue
        entry.last_pub_time = sim_time

        pose = store.get_body_pose(entry.env_id, entry.prim_name)
        if pose is None:
            continue
        (px, py, pz), (qw, qx, qy, qz) = pose

        stamp = _make_stamp(sim_time)
        child_frame = f'{entry.tf_prefix}{entry.base_frame}'

        tf_msg = geometry_msgs.msg.TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = entry.odom_frame_id
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = px
        tf_msg.transform.translation.y = py
        tf_msg.transform.translation.z = pz
        tf_msg.transform.rotation.w = qw
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        transforms.append(tf_msg)

        map_tf = geometry_msgs.msg.TransformStamped()
        map_tf.header.stamp = stamp
        map_tf.header.frame_id = 'map'
        map_tf.child_frame_id = entry.odom_frame_id
        map_tf.transform.rotation.w = 1.0
        transforms.append(map_tf)

        if entry.odom_pub is None:
            continue
        odom = nav_msgs.msg.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = entry.odom_frame_id
        odom.child_frame_id = child_frame
        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = pz
        odom.pose.pose.orientation.w = qw
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz

        if entry.localization:
            model = store.get_model(entry.env_id)
            data = store.get_data(entry.env_id)
            if model is not None and data is not None:
                body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, entry.prim_name)
                if body_id >= 0:
                    (vx, vy, vz), (wx, wy, wz) = _freejoint_twist(model, data, body_id)
                    odom.twist.twist.linear.x = vx
                    odom.twist.twist.linear.y = vy
                    odom.twist.twist.linear.z = vz
                    odom.twist.twist.angular.x = wx
                    odom.twist.twist.angular.y = wy
                    odom.twist.twist.angular.z = wz

        entry.odom_pub.publish(odom)

    if transforms:
        _broadcaster.sendTransform(transforms)


hooks.register_post_spawn(_post_spawn)
hooks.register_despawn(_despawn)
hooks.register_pump(_pump)


def setup() -> None:
    """No-op marker; import side-effects handle registration."""


__all__ = ['setup']
