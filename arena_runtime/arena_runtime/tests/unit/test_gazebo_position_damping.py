import xml.etree.ElementTree as ET

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _free_position_joints

_URDF = """<robot name="arm">
  <link name="base"/><link name="upper"/><link name="wheel"/>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/><limit effort="43" velocity="1.95" lower="-1" upper="1"/><dynamics damping="40" friction="1"/></joint>
  <joint name="wheel_joint" type="continuous"><parent link="base"/><child link="wheel"/><dynamics damping="0.5" friction="0.1"/></joint>
  <ros2_control name="arm_system" type="system">
    <joint name="shoulder"><command_interface name="position"/><state_interface name="position"/></joint>
    <joint name="wheel_joint"><command_interface name="velocity"/><state_interface name="velocity"/></joint>
  </ros2_control>
</robot>"""


def _dynamics(urdf: str, joint: str) -> dict[str, str]:
    element = ET.fromstring(urdf).find(f"joint[@name='{joint}']/dynamics")
    assert element is not None
    return dict(element.attrib)


def test_a_position_commanded_joint_keeps_its_friction_and_loses_its_damping():
    assert _dynamics(_free_position_joints(_URDF), 'shoulder') == {'damping': '0', 'friction': '1'}


def test_a_velocity_commanded_joint_keeps_its_damping():
    assert _dynamics(_free_position_joints(_URDF), 'wheel_joint') == {'damping': '0.5', 'friction': '0.1'}


def test_everything_but_the_zeroed_damping_reaches_gazebo_verbatim():
    assert _free_position_joints(_URDF) == _URDF.replace(' damping="40"', ' damping="0"', 1)
