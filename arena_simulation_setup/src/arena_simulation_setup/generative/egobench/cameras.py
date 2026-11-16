"""The three-view camera rig of one encounter: the pedestrian's eyes, the robot's camera and an overhead hold framed on the designed geometry."""

from __future__ import annotations

import math

from arena_simulation_setup.shared.pathing import Point

RIG = "three_view"
ROBOT_CAMERA_AT = (0.2, 0.0, 0.6)  # m in the robot's base frame, x forward, z up
FOV = 1.047  # rad, horizontal, of every view
ASPECT = 16 / 9
MARGIN_M = 1.5
MIN_HEIGHT_M = 6.0
TILT = 0.15
LOOK_HEIGHT_M = 0.5
TAIL_S = 4.0


def floor_tenth(seconds: float) -> float:
    return math.floor(seconds * 10 + 1e-9) / 10


def duration(meet: float, episode: float) -> float:
    """Seconds the rig films: the encounter plus `TAIL_S`, never past the episode end."""
    return floor_tenth(min(meet + TAIL_S, episode))


def overhead(points: list[Point], heading: float) -> tuple[list[float], list[float]]:
    """Eye and target of a hold that frames `points`, the robot's heading running left to right across the frame."""
    ux, uy = math.cos(heading), math.sin(heading)
    along = [x * ux + y * uy for x, y in points]
    across = [-x * uy + y * ux for x, y in points]
    c_along, c_across = (min(along) + max(along)) / 2, (min(across) + max(across)) / 2
    cx, cy = c_along * ux - c_across * uy, c_along * uy + c_across * ux
    half_w = (max(along) - min(along)) / 2 + MARGIN_M
    half_h = (max(across) - min(across)) / 2 + MARGIN_M
    tan_h = math.tan(FOV / 2)
    height = max(MIN_HEIGHT_M, half_w / tan_h, half_h / (tan_h / ASPECT)) + LOOK_HEIGHT_M
    back = TILT * height
    eye = [round(cx + uy * back, 2), round(cy - ux * back, 2), round(height, 2)]
    return eye, [round(cx, 2), round(cy, 2), LOOK_HEIGHT_M]


def rig(ped: str, points: list[Point], heading: float, duration: float) -> dict[str, object]:
    """The rig of one encounter: every view films through the sim camera at the same instants for `duration` s."""
    eye, target = overhead(points, heading)
    return {
        "params": {"duration": duration},
        "cameras": {
            "pedestrian": {"source": "sim", "timeline": [{"pov": {"face": f"${{face.{ped}}}", "duration": "${duration}", "fov": FOV}}]},
            "overhead": {"source": "sim", "timeline": [{"look": {"eye": eye, "target": target, "fov": FOV}}, {"hold": {"duration": "${duration}"}}]},
            "robot": {"mount": {"frame": "${robot}", "at": list(ROBOT_CAMERA_AT), "pitch": 0.0, "fov": FOV}},
        },
    }
