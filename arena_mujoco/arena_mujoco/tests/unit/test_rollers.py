import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from arena_mujoco.control import _add_velocity_actuator
from arena_mujoco.rollers import add_rollers, roller_wheels
from arena_mujoco.scene import SceneStore
from arena_mujoco.services.SpawnUrdf import _attach_robot, _mujoco_compat

_RADIUS = 0.127
_HALF_BASE = 0.2
_HALF_TRACK = 0.25
_REACH = _HALF_BASE + _HALF_TRACK
_WHEELS = {
    'front_left': (_HALF_BASE, _HALF_TRACK, '0.70710678 -0.70710678 0'),
    'front_right': (_HALF_BASE, -_HALF_TRACK, '0.70710678 0.70710678 0'),
    'back_left': (-_HALF_BASE, _HALF_TRACK, '0.70710678 0.70710678 0'),
    'back_right': (-_HALF_BASE, -_HALF_TRACK, '0.70710678 -0.70710678 0'),
}


def _urdf(tmp_path: Path, mu: float = 1.0, mu2: float = 0.0, wheel_yaw: float = 0.0, frame: str = 'gz:expressed_in="base_footprint"') -> Path:
    parts = [
        '<robot name="r" xmlns:gz="http://gazebosim.org/schema">',
        '<link name="base_footprint"/>',
        '<link name="base_link"><inertial><mass value="80"/><inertia ixx="3" iyy="5" izz="6" ixy="0" ixz="0" iyz="0"/></inertial>',
        '<collision><origin xyz="0 0 0.1"/><geometry><box size="0.7 0.6 0.2"/></geometry></collision></link>',
        f'<joint name="base_joint" type="fixed"><parent link="base_footprint"/><child link="base_link"/><origin xyz="0 0 {_RADIUS}"/></joint>',
    ]
    for name, (x, y, fdir1) in _WHEELS.items():
        parts += [
            f'<link name="{name}_wheel_link"><inertial><mass value="6"/><inertia ixx="0.03" iyy="0.05" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial>',
            f'<collision><geometry><sphere radius="{_RADIUS}"/></geometry></collision></link>',
            f'<joint name="{name}_wheel_joint" type="continuous"><parent link="base_link"/><child link="{name}_wheel_link"/>',
            f'<origin xyz="{x} {y} 0" rpy="0 0 {wheel_yaw}"/><axis xyz="{np.sin(wheel_yaw)} {np.cos(wheel_yaw)} 0"/></joint>',
            f'<gazebo reference="{name}_wheel_link"><collision><surface><friction><ode><mu>{mu}</mu><mu2>{mu2}</mu2>',
            f'<fdir1 {frame}>{fdir1}</fdir1></ode></friction></surface></collision></gazebo>',
        ]
    parts.append('</robot>')
    path = tmp_path / 'r.urdf'
    path.write_text(''.join(parts))
    return path


def _spawn(store: SceneStore, urdf: Path) -> None:
    compat, _, _ = _mujoco_compat(str(urdf))
    robot = mujoco.MjSpec.from_file(compat)
    add_rollers(robot, roller_wheels(ET.parse(urdf).getroot()))
    cache = store.asset_cache(0)

    def build(spec: mujoco.MjSpec) -> mujoco.MjsBody:
        root = _attach_robot(spec, cache, robot, 'r_', 'base_footprint', (0.0, 0.0, 0.002), (1.0, 0.0, 0.0, 0.0), True)
        for name in _WHEELS:
            _add_velocity_actuator(spec, f'r_{name}_wheel_joint', f'{name}_vel')
        return root

    store.add_body(0, build)
    store.recompile(0)


def _drive(store: SceneStore, vx: float, vy: float, wz: float) -> np.ndarray:
    """Body twist (vx, vy, wz) averaged over two seconds of holding the wheel speeds of the commanded twist."""
    model = store.get_model(0)
    data = store.get_data(0)
    speeds = {
        'front_left': (vx - vy - _REACH * wz) / _RADIUS,
        'front_right': (vx + vy + _REACH * wz) / _RADIUS,
        'back_left': (vx + vy - _REACH * wz) / _RADIUS,
        'back_right': (vx - vy + _REACH * wz) / _RADIUS,
    }
    store.step(250)
    for name, speed in speeds.items():
        data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'{name}_vel')] = speed
    store.step(500)
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'r_base_footprint_free')
    qadr = int(model.jnt_qposadr[free])
    vadr = int(model.jnt_dofadr[free])
    twists = []
    for _ in range(1000):
        store.step(1)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[qadr + 3 : qadr + 7])
        linear = rotation.reshape(3, 3).T @ data.qvel[vadr : vadr + 3]
        twists.append((linear[0], linear[1], data.qvel[vadr + 5]))
    return np.mean(twists, axis=0)


@pytest.fixture
def store() -> SceneStore:
    store = SceneStore()
    store.ensure_env(0)
    return store


@pytest.mark.parametrize('twist', [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 1.0), (0.4, -0.3, 0.0)])
def test_wheels_gripping_along_one_diagonal_move_the_base_at_the_commanded_twist(store: SceneStore, tmp_path: Path, twist: tuple[float, float, float]):
    _spawn(store, _urdf(tmp_path))
    assert _drive(store, *twist) == pytest.approx(twist, abs=0.02)


def test_wheels_gripping_equally_both_ways_cannot_strafe(store: SceneStore, tmp_path: Path):
    urdf = _urdf(tmp_path, mu=1.0, mu2=1.0)
    assert roller_wheels(ET.parse(urdf).getroot()) == []
    _spawn(store, urdf)
    assert abs(_drive(store, 0.0, 0.5, 0.0)[1]) < 0.1


def test_grip_direction_is_read_in_the_frame_it_is_expressed_in(tmp_path: Path):
    turned = roller_wheels(ET.parse(_urdf(tmp_path, wheel_yaw=np.pi / 2)).getroot())
    assert turned[0].link == 'front_left_wheel_link'
    assert turned[0].grip == pytest.approx((-0.70710678, -0.70710678, 0.0), abs=1e-6)
    own = roller_wheels(ET.parse(_urdf(tmp_path, wheel_yaw=np.pi / 2, frame='')).getroot())
    assert own[0].grip == pytest.approx((0.70710678, -0.70710678, 0.0), abs=1e-6)


def test_grip_across_fdir1_turns_the_roller_axis_a_quarter(tmp_path: Path):
    along = roller_wheels(ET.parse(_urdf(tmp_path)).getroot())
    across = roller_wheels(ET.parse(_urdf(tmp_path, mu=0.0, mu2=1.0)).getroot())
    for a, b in zip(along, across, strict=True):
        assert np.dot(a.grip, b.grip) == pytest.approx(0.0, abs=1e-6)
        assert abs(b.grip[2]) < 1e-6


def test_wheels_turned_on_their_mount_still_strafe(store: SceneStore, tmp_path: Path):
    _spawn(store, _urdf(tmp_path, wheel_yaw=np.pi))
    assert _drive(store, 0.0, -0.5, 0.0) == pytest.approx((0.0, -0.5, 0.0), abs=0.02)


def test_rollers_leave_the_robot_mass_unchanged(store: SceneStore, tmp_path: Path):
    plain = mujoco.MjSpec.from_file(_mujoco_compat(str(_urdf(tmp_path, mu2=1.0)))[0]).compile()
    _spawn(store, _urdf(tmp_path))
    model = store.get_model(0)
    assert model.body_mass.sum() == pytest.approx(plain.body_mass.sum())
    assert model.njnt == plain.njnt + 1 + 12 * len(_WHEELS)
