"""IMU sensor publisher: framequat + gyro + accelerometer -> sensor_msgs/Imu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from arena_robots.Sensor import SensorType
from sensor_msgs.msg import Imu
from std_msgs.msg import Header

from arena_mujoco.context import get_scene_store
from arena_mujoco.sensors.core import SensorPublisher, make_stamp, register_sensor, sensor_pose

if TYPE_CHECKING:
    import rclpy.node
    from arena_robots.Sensor import SensorSpec

    from arena_mujoco.services.SpawnUrdf import RobotEntry, UrdfSensor

_DEFAULT_RATE_HZ = 100.0

_SENSORS = {
    'framequat': mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    'gyro': mujoco.mjtSensor.mjSENS_GYRO,
    'accelerometer': mujoco.mjtSensor.mjSENS_ACCELEROMETER,
}


@register_sensor(SensorType.IMU)
class ImuPublisher(SensorPublisher):
    """Publishes sensor_msgs/Imu fused from MuJoCo framequat, gyro, accelerometer."""

    def __init__(
        self,
        node: rclpy.node.Node,
        env_id: int,
        spec: SensorSpec,
        frame_id: str,
        robot: RobotEntry,
        element_prefix: str,
    ) -> None:
        super().__init__(node, env_id, spec, frame_id, robot, element_prefix)
        self._pub = node.create_publisher(Imu, spec.topic, 10)
        self._adr: dict[str, int] | None = None
        rate = None if self.urdf_sensor is None else self.urdf_sensor.update_rate
        self._rate = rate if rate else _DEFAULT_RATE_HZ

    def rebind(self) -> None:
        self._adr = None

    @classmethod
    def add_model_elements(
        cls,
        spec_builder: mujoco.MjSpec,
        mount_body: mujoco.MjsBody,
        sensor_spec: SensorSpec,
        element_prefix: str,
        urdf_sensor: UrdfSensor | None,
    ) -> None:
        """Add an imu site plus framequat, gyro, and accelerometer sensors."""
        prefix = f'{element_prefix}{sensor_spec.name}_'
        pos, quat = sensor_pose(urdf_sensor)
        mount_body.add_site(name=prefix + 'site', pos=pos, quat=quat.tolist())
        for name, sensor_type in _SENSORS.items():
            sensor = spec_builder.add_sensor()
            sensor.name = prefix + name
            sensor.type = sensor_type
            sensor.objtype = mujoco.mjtObj.mjOBJ_SITE
            sensor.objname = prefix + 'site'

    def _addresses(self, model: mujoco.MjModel) -> dict[str, int]:
        """sensordata offsets of this imu's sensors, absent ones left out."""
        prefix = f'{self.element_prefix}{self.spec.name}_'
        adr = {}
        for name in _SENSORS:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + name)
            if sensor_id >= 0:
                adr[name] = int(model.sensor_adr[sensor_id])
        return adr

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._adr is None:
            self._adr = self._addresses(model)
        adr = self._adr

        msg = Imu()
        msg.header = Header(stamp=make_stamp(get_scene_store().time), frame_id=self.frame_id)

        if 'framequat' in adr:
            w, x, y, z = (float(v) for v in data.sensordata[adr['framequat'] : adr['framequat'] + 4])
            msg.orientation.x = x
            msg.orientation.y = y
            msg.orientation.z = z
            msg.orientation.w = w
        msg.orientation_covariance[0] = -1.0

        if 'gyro' in adr:
            gyro = data.sensordata[adr['gyro'] : adr['gyro'] + 3]
            msg.angular_velocity.x = float(gyro[0])
            msg.angular_velocity.y = float(gyro[1])
            msg.angular_velocity.z = float(gyro[2])
        msg.angular_velocity_covariance[0] = -1.0

        if 'accelerometer' in adr:
            accel = data.sensordata[adr['accelerometer'] : adr['accelerometer'] + 3]
            msg.linear_acceleration.x = float(accel[0])
            msg.linear_acceleration.y = float(accel[1])
            msg.linear_acceleration.z = float(accel[2])
        msg.linear_acceleration_covariance[0] = -1.0

        self._pub.publish(msg)

    @property
    def rate_hz(self) -> float:
        return self._rate
