"""Contact sensor publisher: contacts on the sensor frame's link -> ros_gz_interfaces/Contacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
from arena_robots.Sensor import SensorType
from geometry_msgs.msg import Vector3
from ros_gz_interfaces.msg import Contact, Contacts, Entity
from std_msgs.msg import Header

from arena_mujoco.context import get_scene_store
from arena_mujoco.sensors.core import SensorPublisher, make_stamp, register_sensor

if TYPE_CHECKING:
    import rclpy.node
    from arena_robots.Sensor import SensorSpec

    from arena_mujoco.services.SpawnUrdf import RobotEntry, UrdfSensor

_DEFAULT_RATE_HZ = 50.0


def outside_contacts(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    """Rows (contact index, geom of body_id, other geom) for every penetrating contact with a geom outside body_id's kinematic tree."""
    if data.ncon == 0:
        return np.empty((0, 3), dtype=int)
    geoms = data.contact.geom
    bodies = model.geom_bodyid[geoms]
    own = bodies == body_id
    outside = model.body_rootid[bodies] != model.body_rootid[body_id]
    touching = data.contact.dist <= 0.0
    first = np.flatnonzero(own[:, 0] & outside[:, 1] & touching)
    second = np.flatnonzero(own[:, 1] & outside[:, 0] & touching)
    return np.concatenate([np.stack([first, geoms[first, 0], geoms[first, 1]], axis=1), np.stack([second, geoms[second, 1], geoms[second, 0]], axis=1)])


def _entity(model: mujoco.MjModel, geom: int) -> Entity:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom])
    return Entity(id=int(geom), name=name or '', type=Entity.COLLISION)


def contacts_message(model: mujoco.MjModel, data: mujoco.MjData, body_id: int, header: Header) -> Contacts:
    """One Contact per touching geom pair: world positions, normals pointing into body_id's geom, penetration depths."""
    pairs: dict[tuple[int, int], Contact] = {}
    for index, own, other in outside_contacts(model, data, body_id).tolist():
        contact = pairs.get((own, other))
        if contact is None:
            contact = pairs[(own, other)] = Contact(collision1=_entity(model, own), collision2=_entity(model, other))
        x, y, z = data.contact.pos[index].tolist()
        normal = data.contact.frame[index, :3] * (-1.0 if data.contact.geom[index, 0] == own else 1.0)
        contact.positions.append(Vector3(x=x, y=y, z=z))
        contact.normals.append(Vector3(x=float(normal[0]), y=float(normal[1]), z=float(normal[2])))
        contact.depths.append(float(-data.contact.dist[index]))
    return Contacts(header=header, contacts=list(pairs.values()))


@register_sensor(SensorType.CONTACT)
class ContactPublisher(SensorPublisher):
    """Publishes the contacts between the sensor frame's link and anything but its own robot, an empty list while it touches nothing."""

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
        self._pub = node.create_publisher(Contacts, spec.topic, 10)
        self._site_name = f'{element_prefix}{spec.name}_site'
        self._body_id = -1
        rate = None if self.urdf_sensor is None else self.urdf_sensor.update_rate
        self._rate = rate if rate else _DEFAULT_RATE_HZ

    def rebind(self) -> None:
        self._body_id = -1

    @classmethod
    def add_model_elements(
        cls,
        spec_builder: mujoco.MjSpec,
        mount_body: mujoco.MjsBody,
        sensor_spec: SensorSpec,
        element_prefix: str,
        urdf_sensor: UrdfSensor | None,
    ) -> None:
        """Add a site marking the body whose contacts are reported."""
        del spec_builder, urdf_sensor
        mount_body.add_site().name = f'{element_prefix}{sensor_spec.name}_site'

    def publish(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        if self._body_id < 0:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self._site_name)
            if site_id < 0:
                return
            self._body_id = int(model.site_bodyid[site_id])
        header = Header(stamp=make_stamp(get_scene_store().time), frame_id=self.frame_id)
        self._pub.publish(contacts_message(model, data, self._body_id, header))

    @property
    def rate_hz(self) -> float:
        return self._rate
