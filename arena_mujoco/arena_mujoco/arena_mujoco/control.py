"""ros2_control bridge: per-joint position+velocity actuators wired to JointState topics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import builtin_interfaces.msg
import mujoco
import sensor_msgs.msg

from arena_mujoco import hooks
from arena_mujoco.context import get_scene_store
from arena_mujoco.scene import sanitize_id

if TYPE_CHECKING:
    import rclpy.publisher
    import rclpy.subscription

    from arena_mujoco.services.SpawnUrdf import RobotEntry

_POSITION_KP = 1.0e3
_POSITION_KV = 1.0e2
_VELOCITY_KV = 1.0e2


class _Robot:
    """Live wiring for one spawned robot: actuator maps, command staging, ROS endpoints."""

    def __init__(
        self,
        env_id: int,
        joint_names: tuple[str, ...],
        actuators: dict[str, tuple[str, str]],
        prim_name: str,
    ) -> None:
        self.env_id = env_id
        self.prim_name = prim_name
        self.joint_names = joint_names
        self.actuators = actuators
        self.mjcf_joints: dict[str, str] = {}
        self._pos_ids: dict[str, int] = {}
        self._vel_ids: dict[str, int] = {}
        self._joint_ids: dict[str, int] = {}
        self._model: mujoco.MjModel | None = None
        self._staged: dict[int, float] = {}
        self.pub: rclpy.publisher.Publisher | None = None
        self.sub_vel: rclpy.subscription.Subscription | None = None
        self.sub_pos: rclpy.subscription.Subscription | None = None

    def stage_velocity(self, msg: sensor_msgs.msg.JointState) -> None:
        """Map each commanded velocity onto its velocity actuator's ctrl."""
        for name, value in zip(msg.name, msg.velocity, strict=False):
            entry = self.actuators.get(name)
            if entry is None or not math.isfinite(value):
                continue
            actuator_id = self._vel_ids.get(entry[1])
            if actuator_id is not None:
                self._staged[actuator_id] = float(value)

    def stage_position(self, msg: sensor_msgs.msg.JointState) -> None:
        """Map each commanded position onto its position actuator's ctrl."""
        for name, value in zip(msg.name, msg.position, strict=False):
            entry = self.actuators.get(name)
            if entry is None or not math.isfinite(value):
                continue
            actuator_id = self._pos_ids.get(entry[0])
            if actuator_id is not None:
                self._staged[actuator_id] = float(value)

    def resolve(self, model: mujoco.MjModel) -> None:
        """Resolve actuator and joint names to ids against the live model, rebinding on recompile."""
        self._pos_ids.clear()
        self._vel_ids.clear()
        self._joint_ids.clear()
        self._staged.clear()
        for pos_name, vel_name in self.actuators.values():
            pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, pos_name)
            vel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, vel_name)
            if pos_id >= 0:
                self._pos_ids[pos_name] = pos_id
            if vel_id >= 0:
                self._vel_ids[vel_name] = vel_id
        for joint, mjcf in self.mjcf_joints.items():
            jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mjcf)
            if jnt_id >= 0:
                self._joint_ids[joint] = jnt_id
        self._model = model

    def apply_and_publish(self, sim_time: float) -> None:
        """Apply staged ctrl onto data, then publish joint states (rate-limited)."""
        store = get_scene_store()
        model = store.get_model(self.env_id)
        data = store.get_data(self.env_id)
        if model is None or data is None:
            return
        if self._model is not model:
            self.resolve(model)
        for actuator_id, value in self._staged.items():
            data.ctrl[actuator_id] = value
        if self.pub is None:
            return
        self.pub.publish(self._joint_state_msg(model, data, sim_time))

    def _joint_state_msg(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        sim_time: float,
    ) -> sensor_msgs.msg.JointState:
        msg = sensor_msgs.msg.JointState()
        msg.header.stamp = _make_stamp(sim_time)
        for joint in self.joint_names:
            jnt_id = self._joint_ids.get(joint)
            if jnt_id is None:
                continue
            qadr = int(model.jnt_qposadr[jnt_id])
            vadr = int(model.jnt_dofadr[jnt_id])
            msg.name.append(joint)
            msg.position.append(float(data.qpos[qadr]))
            msg.velocity.append(float(data.qvel[vadr]))
            msg.effort.append(float(data.qfrc_actuator[vadr]))
        return msg


_robots: dict[tuple[int, str], _Robot] = {}

_pending: dict[str, dict[str, tuple[str, str]]] = {}
_pending_mjcf: dict[str, dict[str, str]] = {}


def _joints_under(root: mujoco.MjsBody) -> list[mujoco.MjsJoint]:
    """Return every actuatable joint in the subtree rooted at root."""
    joints: list[mujoco.MjsJoint] = []
    stack = [root]
    while stack:
        body = stack.pop()
        joints.extend(j for j in body.joints if j.type != mujoco.mjtJoint.mjJNT_FREE)
        stack.extend(body.bodies)
    return joints


def _actuator_names(joint_mjcf: str) -> tuple[str, str]:
    """Name the position and velocity actuators off the compiled MJCF joint name."""
    return f'{joint_mjcf}_pos', f'{joint_mjcf}_vel'


def _prm(*vals: float) -> list[float]:
    """Pad leading gain/bias coefficients to MuJoCo's fixed length-10 actuator prm vector."""
    return [*vals, *([0.0] * (10 - len(vals)))]


def _add_position_actuator(spec: mujoco.MjSpec, target_joint: str, name: str) -> None:
    """Add a position-servo actuator (kp position feedback, kv damping) targeting target_joint."""
    act = spec.add_actuator()
    act.name = name
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.target = target_joint
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm = _prm(_POSITION_KP)
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm = _prm(0.0, -_POSITION_KP, -_POSITION_KV)


def _add_velocity_actuator(spec: mujoco.MjSpec, target_joint: str, name: str) -> None:
    """Add a velocity-servo actuator (kv velocity feedback) targeting target_joint."""
    act = spec.add_actuator()
    act.name = name
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.target = target_joint
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm = _prm(_VELOCITY_KV)
    act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
    act.biasprm = _prm(0.0, 0.0, -_VELOCITY_KV)


def _pre_compile(
    env_id: int,
    spec: mujoco.MjSpec,
    robot_root_body: mujoco.MjsBody,
    robot_entry: RobotEntry,
    joint_names: list[str],
) -> None:
    """Add position+velocity actuators per actuated joint, record the actuator map."""
    attached = sorted(j.name for j in _joints_under(robot_root_body) if j.name)
    actuator_map: dict[str, tuple[str, str]] = {}
    mjcf_map: dict[str, str] = {}
    for urdf_joint in joint_names:
        joint_mjcf = _match_joint(attached, urdf_joint)
        if joint_mjcf is None:
            continue
        ifaces = robot_entry.command_interfaces.get(urdf_joint) or ('velocity',)
        add_pos = 'position' in ifaces
        add_vel = 'velocity' in ifaces or not add_pos
        pos_name, vel_name = _actuator_names(joint_mjcf)
        if add_pos:
            _add_position_actuator(spec, joint_mjcf, pos_name)
        if add_vel:
            _add_velocity_actuator(spec, joint_mjcf, vel_name)
        actuator_map[urdf_joint] = (pos_name if add_pos else '', vel_name if add_vel else '')
        mjcf_map[urdf_joint] = joint_mjcf
    _pending[robot_entry.prim_name] = actuator_map
    _pending_mjcf[robot_entry.prim_name] = mjcf_map


def _match_joint(attached: list[str], urdf_joint: str) -> str | None:
    """Find the compiled MJCF joint for urdf_joint among attached names."""
    suffix = sanitize_id(urdf_joint)
    delimited = f'_{suffix}'
    candidates = [c for c in attached if c == suffix or c.endswith(delimited)]
    return min(candidates, key=len) if candidates else None


def _post_spawn(
    env_id: int,
    name: str,
    robot_entry: RobotEntry,
    joint_names: list[str],
) -> None:
    """Create the command subscribers and the joint_states publisher for this robot."""
    actuator_map = _pending.pop(robot_entry.prim_name, {})
    robot = _Robot(env_id, tuple(joint_names), actuator_map, robot_entry.prim_name)
    robot.mjcf_joints = _pending_mjcf.pop(robot_entry.prim_name, {})
    ns = _namespace(robot_entry)
    node = hooks.get_node()

    robot.pub = node.create_publisher(sensor_msgs.msg.JointState, f'{ns}/mujoco/joint_states', 10)
    robot.sub_vel = node.create_subscription(
        sensor_msgs.msg.JointState,
        f'{ns}/mujoco/joint_commands_velocity',
        robot.stage_velocity,
        10,
    )
    robot.sub_pos = node.create_subscription(
        sensor_msgs.msg.JointState,
        f'{ns}/mujoco/joint_commands_position',
        robot.stage_position,
        10,
    )
    _robots[(env_id, name)] = robot


def _despawn(env_id: int, robot_entry: RobotEntry) -> None:
    node = hooks.get_node()
    for key, robot in list(_robots.items()):
        if robot.env_id != env_id or robot.prim_name != robot_entry.prim_name:
            continue
        node.destroy_publisher(robot.pub)
        node.destroy_subscription(robot.sub_vel)
        node.destroy_subscription(robot.sub_pos)
        del _robots[key]


def _pump() -> None:
    """Apply staged commands and publish joint states for every active robot."""
    store = get_scene_store()
    for robot in _robots.values():
        robot.apply_and_publish(store.get_clock_time(robot.env_id))


def _make_stamp(sim_time: float) -> builtin_interfaces.msg.Time:
    """Build a ROS time stamp from the per-env sim clock, matching the odom subsystem."""
    stamp = builtin_interfaces.msg.Time()
    stamp.sec = int(sim_time)
    stamp.nanosec = int(round((sim_time - int(sim_time)) * 1e9))
    return stamp


def _namespace(robot_entry: RobotEntry) -> str:
    """Derive the bridge ROS namespace from the adapter-supplied joint_states_topic."""
    return robot_entry.joint_states_topic.removesuffix('/joint_states')


hooks.register_pre_compile(_pre_compile)
hooks.register_post_spawn(_post_spawn)
hooks.register_despawn(_despawn)
hooks.register_pump(_pump)


def get_robot(env_id: int, name: str) -> _Robot | None:
    """Return the live control wiring for (env_id, name), or None if not spawned."""
    return _robots.get((env_id, name))


__all__ = ['get_robot']
