"""DisplayKind.MAP renderer: nav_msgs/OccupancyGrid → a textured quad at the grid's metric size, origin and frame.

Free cells are light, occupied ones dark, unknown ones grey; costs in between shade toward dark, so costmaps read too.
Maps lie on the floor, other grids (costmaps) slightly above so both show.
"""

from __future__ import annotations

import math

import numpy as np
import rerun as rr
from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind, StyleSpec
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rerun_utils.entity_paths import display_path, in_frame
from rerun_utils.renderers._registry import RendererCtx, register

FREE, OCCUPIED, UNKNOWN = np.array([235, 235, 230]), np.array([45, 45, 50]), np.array([150, 150, 155])
OVERLAY_Z = 0.02


def grid_texture(msg: OccupancyGrid) -> np.ndarray:
    """RGB texture of the grid, row 0 at the grid origin as OccupancyGrid stores it."""
    data = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
    cost = np.clip(data, 0, 100)[..., None] / 100.0
    rgb = FREE + (OCCUPIED - FREE) * cost
    rgb[data < 0] = UNKNOWN
    return rgb.astype(np.uint8)


def grid_quad(msg: OccupancyGrid, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Corners of the grid in its frame (origin pose, metric size) and their texture coordinates."""
    w, h = msg.info.width * msg.info.resolution, msg.info.height * msg.info.resolution
    o = msg.info.origin
    q = o.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    xy = local @ np.array([[c, s], [-s, c]]) + [o.position.x, o.position.y]
    corners = np.column_stack([xy, np.full(4, o.position.z + z)])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return corners, uv


@register(DisplayKind.MAP)
def render_map(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    style = StyleSpec.from_json(d.style_json)
    if not style.enabled:
        return
    entity = display_path(ctx.env_id, robot, d.name)
    z = 0.0 if robot is None and d.name.lower() == "map" else OVERLAY_Z

    def cb(msg: OccupancyGrid) -> None:
        if msg.info.width == 0 or msg.info.height == 0:
            return
        corners, uv = grid_quad(msg, z)
        rr.log(
            entity,
            rr.Mesh3D(vertex_positions=corners, triangle_indices=[[0, 1, 2], [0, 2, 3]], vertex_texcoords=uv, albedo_texture=grid_texture(msg)),
            *in_frame(msg.header.frame_id),
        )

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ctx.node.create_subscription(OccupancyGrid, d.topic, cb, qos)
