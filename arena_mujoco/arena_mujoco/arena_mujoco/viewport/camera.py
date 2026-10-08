"""Free viewport camera: poses in body axes (forward +X, up +Z) rendered offscreen and mirrored onto the viewer's camera."""

from __future__ import annotations

import math

import attrs
import mujoco
import numpy as np

from .controller import Pose, q_from_euler, q_rotate

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
DEFAULT_HFOV = 1.047
ORTHOGRAPHIC_FOCUS_M = 10.0


@attrs.define
class Lens:
    hfov: float = DEFAULT_HFOV
    orthographic: bool = False


def orbit_of(pose: Pose, distance: float) -> tuple[tuple[float, float, float], float, float]:
    """Lookat point, azimuth and elevation in degrees of the MuJoCo free camera at pose, its roll dropped."""
    forward = q_rotate(pose.orientation, (1.0, 0.0, 0.0))
    lookat = (
        pose.position[0] + forward[0] * distance,
        pose.position[1] + forward[1] * distance,
        pose.position[2] + forward[2] * distance,
    )
    azimuth = math.degrees(math.atan2(forward[1], forward[0]))
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, forward[2]))))
    return lookat, azimuth, elevation


def pose_of(lookat: tuple[float, float, float], distance: float, azimuth: float, elevation: float) -> Pose:
    """Pose of a MuJoCo free camera orbiting lookat at distance, angles in degrees."""
    az, el = math.radians(azimuth), math.radians(elevation)
    forward = (math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))
    eye = (
        lookat[0] - forward[0] * distance,
        lookat[1] - forward[1] * distance,
        lookat[2] - forward[2] * distance,
    )
    return Pose(eye, q_from_euler(0.0, -el, az))


def _focus(pose: Pose) -> float:
    """Distance along the view to the ground plane, ORTHOGRAPHIC_FOCUS_M when the view never meets it."""
    forward = q_rotate(pose.orientation, (1.0, 0.0, 0.0))
    if forward[2] > -1e-6 or pose.position[2] <= 0.0:
        return ORTHOGRAPHIC_FOCUS_M
    return -pose.position[2] / forward[2]


def aim(scene: mujoco.MjvScene, model: mujoco.MjModel, pose: Pose, lens: Lens, aspect: float) -> None:
    """Put both eyes of scene at pose with the lens' horizontal field of view over a width / height aspect."""
    forward = q_rotate(pose.orientation, (1.0, 0.0, 0.0))
    up = q_rotate(pose.orientation, (0.0, 0.0, 1.0))
    near = model.vis.map.znear * model.stat.extent
    half_width = math.tan(lens.hfov / 2.0) * (_focus(pose) if lens.orthographic else near)
    for eye in scene.camera:
        eye.pos[:] = pose.position
        eye.forward[:] = forward
        eye.up[:] = up
        eye.orthographic = int(lens.orthographic)
        eye.frustum_center = 0.0
        eye.frustum_width = half_width
        eye.frustum_top = half_width / aspect
        eye.frustum_bottom = -half_width / aspect
        eye.frustum_near = near
        eye.frustum_far = model.vis.map.zfar * model.stat.extent


def render(renderer: mujoco.Renderer, data: mujoco.MjData, pose: Pose, lens: Lens, option: mujoco.MjvOption) -> np.ndarray:
    """rgb8 frame (height, width, 3) of data seen from pose."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.0
    lookat, camera.azimuth, camera.elevation = orbit_of(pose, camera.distance)
    camera.lookat[:] = lookat
    renderer.update_scene(data, camera=camera, scene_option=option)
    aim(renderer.scene, renderer.model, pose, lens, renderer.width / renderer.height)
    return renderer.render()
