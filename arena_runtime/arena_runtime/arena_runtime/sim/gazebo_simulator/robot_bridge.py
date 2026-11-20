import enum
import xml.etree.ElementTree

import attrs
import yaml
from arena_robots.sensors import output_topics


class MappingDirection(enum.StrEnum):
    BIDIRECTIONAL = '@'
    GZ_TO_ROS = '['
    ROS_TO_GZ = ']'


_OUTPUT_TYPES: dict[str, tuple[str, str]] = {
    'laserscan': ('sensor_msgs/msg/LaserScan', 'gz.msgs.LaserScan'),
    'pointcloud': ('sensor_msgs/msg/PointCloud2', 'gz.msgs.PointCloudPacked'),
    'image': ('sensor_msgs/msg/Image', 'gz.msgs.Image'),
    'depth': ('sensor_msgs/msg/Image', 'gz.msgs.Image'),
    'camera_info': ('sensor_msgs/msg/CameraInfo', 'gz.msgs.CameraInfo'),
    'imu': ('sensor_msgs/msg/Imu', 'gz.msgs.IMU'),
}


@attrs.define
class _TopicMapping(dict[str, str]):
    gz_topic: str
    ros_topic: str
    ros_type: str
    gz_type: str
    direction: MappingDirection = attrs.field(converter=MappingDirection)

    def as_arg(self) -> str:
        return f"{self.gz_topic}@{self.ros_type}{self.direction.value}{self.gz_type}"

    def as_remapping(self) -> tuple[str, str]:
        return (self.gz_topic, self.ros_topic)

    def as_yaml_dict(self) -> dict[str, str]:
        return attrs.asdict(self, value_serializer=lambda _, __, v: v.name if isinstance(v, MappingDirection) else v)


class BridgeConfiguration(list[_TopicMapping]):
    @classmethod
    def from_urdf_sensors(cls, urdf_xml: str, sim_path: str) -> "BridgeConfiguration":
        prefix = sim_path + "/"
        root = xml.etree.ElementTree.fromstring(urdf_xml)
        mappings: list[_TopicMapping] = []

        for sensor in root.findall(".//gazebo//sensor"):
            topic = sensor.findtext("./topic") or sensor.findtext(".//topic")
            topic = topic.strip() if topic is not None else None
            if not topic or not topic.startswith(prefix):
                continue
            info = sensor.findtext("./camera/camera_info_topic") or sensor.findtext(".//camera_info_topic")
            info = info.strip() if info is not None else None
            if info is not None and not info.startswith(prefix):
                info = None

            outputs = output_topics(
                sensor.get("type") or '',
                topic.removeprefix(prefix),
                info.removeprefix(prefix) if info is not None else None,
            )
            if outputs is None:
                continue
            for kind, ros_topic in outputs.items():
                ros_type, gz_type = _OUTPUT_TYPES[kind]
                mappings.append(_TopicMapping(gz_topic=f"{prefix}{ros_topic}", ros_topic=ros_topic, ros_type=ros_type, gz_type=gz_type, direction=MappingDirection.GZ_TO_ROS))

        return cls(mappings)

    def as_args(self) -> list[str]:
        return list(map(_TopicMapping.as_arg, self))

    def as_remappings(self) -> list[tuple[str, str]]:
        return list(map(_TopicMapping.as_remapping, self))

    def as_yaml(self) -> str:
        result = yaml.safe_dump(list(map(_TopicMapping.as_yaml_dict, self)))
        assert result is not None
        if isinstance(result, bytes):
            return result.decode('utf-8')
        return str(result)
