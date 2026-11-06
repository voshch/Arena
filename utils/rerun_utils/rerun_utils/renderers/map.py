"""DisplayKind.MAP renderer: nav_msgs/OccupancyGrid → a textured quad at the grid's metric size, origin and frame.

Free cells are light, occupied ones dark, unknown ones grey; costs in between shade toward dark.
Grids the manifest styles as costmaps lie slightly above the map as one point per cell with a cost, so the map shows
through their free and unknown cells.
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
MAX_TEXTURE_SIDE = 4096


def is_overlay(style: StyleSpec) -> bool:
    """Whether the manifest styles the grid as a costmap, which lies above the static map."""
    rviz = style.extra.get("rviz")
    return isinstance(rviz, dict) and rviz.get("Color Scheme") == "costmap"


def grid_cells(msg: OccupancyGrid) -> np.ndarray:
    """The grid's cells as (height, width)."""
    return np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)


def downsample(cells: np.ndarray, cap: int = MAX_TEXTURE_SIDE) -> np.ndarray:
    """The cells with each block of k by k reduced to its maximum, k the smallest factor that fits a side within `cap`."""
    k = -(-max(cells.shape) // cap)
    if k <= 1:
        return cells
    h, w = (-(-n // k) * k for n in cells.shape)
    padded = np.full((h, w), -1, dtype=cells.dtype)
    padded[: cells.shape[0], : cells.shape[1]] = cells
    return padded.reshape(h // k, k, w // k, k).max(axis=(1, 3))


def _shade(data: np.ndarray) -> np.ndarray:
    cost = np.clip(data, 0, 100)[..., None] / 100.0
    rgb = FREE + (OCCUPIED - FREE) * cost
    rgb[data < 0] = UNKNOWN
    return rgb.astype(np.uint8)


def grid_texture(data: np.ndarray) -> np.ndarray:
    """RGB texture of the cells, row 0 at the grid origin as OccupancyGrid stores it."""
    return _shade(data)


def occupied_cells(msg: OccupancyGrid, cells: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Centers in the grid's frame and colors of the cells with a cost, the free and unknown ones left out."""
    rows, cols = np.nonzero(cells > 0)
    res = msg.info.resolution
    o = msg.info.origin
    q = o.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.column_stack([(cols + 0.5) * res, (rows + 0.5) * res])
    xy = local @ np.array([[c, s], [-s, c]]) + [o.position.x, o.position.y]
    centers = np.column_stack([xy, np.full(len(rows), o.position.z + z)])
    return centers, _shade(cells[rows, cols])


def grid_key(msg: OccupancyGrid) -> tuple:
    """Everything of the grid that decides what is logged."""
    i, o = msg.info, msg.info.origin
    pose = (o.position.x, o.position.y, o.position.z, o.orientation.x, o.orientation.y, o.orientation.z, o.orientation.w)
    return (msg.header.frame_id, i.width, i.height, i.resolution, pose, np.asarray(msg.data, dtype=np.int16).tobytes())


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
    overlay = is_overlay(style)
    z = OVERLAY_Z if overlay else 0.0
    last: tuple | None = None

    def cb(msg: OccupancyGrid) -> None:
        nonlocal last
        if msg.info.width == 0 or msg.info.height == 0:
            return
        cells = grid_cells(msg)
        key = grid_key(msg)
        if key == last:
            return
        last = key
        if overlay:
            centers, colors = occupied_cells(msg, cells, z)
            rr.log(entity, rr.Points3D(centers, colors=colors, radii=msg.info.resolution / 2), *in_frame(msg.header.frame_id))
            return
        corners, uv = grid_quad(msg, z)
        rr.log(
            entity,
            rr.Mesh3D(vertex_positions=corners, triangle_indices=[[0, 1, 2], [0, 2, 3]], vertex_texcoords=uv, albedo_texture=grid_texture(downsample(cells))),
            *in_frame(msg.header.frame_id),
        )

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    ctx.node.create_subscription(OccupancyGrid, d.topic, cb, qos)
