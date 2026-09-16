"""TM_Patrol's perception and service geometry, without a node."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("arena_people_msgs")

from task_generator.manager.world_manager.utils import GridFrame, WorldOccupancy
from task_generator.shared import Position
from task_generator.tasks.robots.patrol.impl import TM_Patrol

RESOLUTION = 0.05
SHAPE = (200, 200)  # 10 m x 10 m


def _param(value: object) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _mode(*, walls: np.ndarray | None = None, sensor_range: float = 12.0, require_los: bool = True, agent: str = "caller", clip: str = "beckon") -> TM_Patrol:
    """A TM_Patrol with only the fields its pure helpers read."""
    mode = object.__new__(TM_Patrol)
    grid = np.full(SHAPE, WorldOccupancy.EMPTY, dtype=np.uint8) if walls is None else walls
    frame = GridFrame(shape=SHAPE, origin=Position(0.0, 0.0, 0.0), resolution=RESOLUTION)
    world_map = SimpleNamespace(
        frame=frame,
        occupancy=SimpleNamespace(walls=grid),
        resolution=RESOLUTION,
        tf_pos2grid=frame.pos2grid,
    )
    mode._ctx = SimpleNamespace(world_manager=SimpleNamespace(map=world_map))
    mode._sensor_range = _param(sensor_range)
    mode._require_los = _param(require_los)
    mode._service_standoff = _param(1.0)
    mode._service_agent = _param(agent)
    mode._clip = _param(clip)
    mode._peds = None
    mode._walls_cache = None
    mode._last_look = -1e9
    return mode


def _wall_at(x: float) -> np.ndarray:
    """A full-height wall one column wide at world x."""
    grid = np.full(SHAPE, WorldOccupancy.EMPTY, dtype=np.uint8)
    col = int(x / RESOLUTION)
    grid[:, col - 1 : col + 2] = WorldOccupancy.FULL
    return grid


def _pose(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(position=Position(x, y, 0.0))


def _peds(*entries: tuple[str, float, float, tuple[str, ...]]) -> SimpleNamespace:
    return SimpleNamespace(
        pedestrians=[
            SimpleNamespace(
                name=name,
                pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y, z=0.0)),
                gestures=[SimpleNamespace(clip=c) for c in clips],
            )
            for name, x, y, clips in entries
        ],
    )


def test_open_space_is_visible_and_a_wall_blocks_it() -> None:
    across = _mode()
    assert across._line_of_sight(Position(1.0, 5.0, 0.0), Position(9.0, 5.0, 0.0))

    blocked = _mode(walls=_wall_at(5.0))
    assert not blocked._line_of_sight(Position(1.0, 5.0, 0.0), Position(9.0, 5.0, 0.0))
    assert blocked._line_of_sight(Position(1.0, 5.0, 0.0), Position(4.0, 5.0, 0.0))  # same side of the wall


def test_perception_needs_range_and_sight() -> None:
    mode = _mode(walls=_wall_at(5.0), sensor_range=4.0)
    near_ped = Position(3.0, 5.0, 0.0)
    assert mode._perceives(_pose(1.0, 5.0), near_ped)
    assert not mode._perceives(_pose(1.0, 5.0), Position(8.0, 5.0, 0.0))  # out of range and behind the wall

    far = _mode(walls=_wall_at(5.0), sensor_range=12.0)
    assert not far._perceives(_pose(1.0, 5.0), Position(8.0, 5.0, 0.0))  # in range, but the wall blocks it
    blind = _mode(walls=_wall_at(5.0), sensor_range=12.0, require_los=False)
    assert blind._perceives(_pose(1.0, 5.0), Position(8.0, 5.0, 0.0))  # the control that ignores what it could see


def test_only_the_named_agent_playing_the_call_clip_counts() -> None:
    mode = _mode()
    mode._peds = _peds(("bystander", 2.0, 2.0, ("beckon",)), ("caller", 4.0, 4.0, ("wave",)))
    assert mode._calling() is None  # the caller is waving, the beckoner is someone else

    mode._peds = _peds(("bystander", 2.0, 2.0, ()), ("caller", 4.0, 4.0, ("beckon",)))
    calling = mode._calling()
    assert calling is not None and calling[0] == "caller"
    assert (calling[1].x, calling[1].y) == (4.0, 4.0)


def test_service_pose_stops_short_of_the_pedestrian_and_faces_them() -> None:
    mode = _mode()
    goal = mode._service_pose(_pose(0.0, 0.0), Position(5.0, 0.0, 0.0))
    assert goal.position.x == pytest.approx(4.0) and goal.position.y == pytest.approx(0.0)
    assert goal.orientation.to_yaw() == pytest.approx(0.0)

    diagonal = mode._service_pose(_pose(0.0, 0.0), Position(3.0, 4.0, 0.0))
    assert math.dist((diagonal.position.x, diagonal.position.y), (3.0, 4.0)) == pytest.approx(1.0)
    assert math.dist((diagonal.position.x, diagonal.position.y), (0.0, 0.0)) == pytest.approx(4.0)
