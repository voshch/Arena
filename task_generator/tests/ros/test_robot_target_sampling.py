from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import shapely
    from arena_rclpy_mixins.Time import Time
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.tree.World.World import Level, LevelDescription, WorldDescription
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    from task_generator.constants.rng import EpisodeRng
    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.manager.world_manager.utils import WorldLayers, WorldMap, WorldOccupancy
    from task_generator.manager.world_manager.world_manager import WorldManager
except ImportError:
    pytestmark = pytest.mark.skip(reason="ROS2 not available")

RES = 0.05
ROOM = [Position(1.0, 1.0), Position(3.0, 1.0), Position(3.0, 3.0), Position(1.0, 3.0), Position(1.0, 1.0)]
SAFE = 0.3


def _manager(blocked: tuple[float, float, float, float] | None, seed: int = 0) -> SimpleNamespace:
    grid = np.full((100, 100), WorldOccupancy.EMPTY, dtype=np.uint8)
    world_manager = WorldManager.__new__(WorldManager)
    world_manager._map = WorldMap(occupancy=WorldLayers(walls=WorldOccupancy(grid)), origin=Position(0.0, 0.0), resolution=RES, time=Time())
    if blocked is not None:
        grid[world_manager._map.tf_poly2mask(shapely.box(*blocked))] = WorldOccupancy.FULL
    world_manager._map.level_origins = {}
    world_manager._multi_map = None
    world_manager._zone_masks = {}
    world_manager._world = WorldDescription(levels={"0": Level(zones=[LevelDescription.Zone(name="room", corners=ROOM)])})
    rng = EpisodeRng()
    rng.reseed(seed)
    node = SimpleNamespace(_world_manager=world_manager, conf=SimpleNamespace(General=SimpleNamespace(RNG=rng)))
    world_manager._NodeInterface__node = node
    return SimpleNamespace(node=node, name="jackal_0", safe_distance=SAFE)


def _blocked_distance(position: Position, blocked: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = blocked
    return math.hypot(max(x0 - position.x, 0.0, position.x - x1), max(y0 - position.y, 0.0, position.y - y1))


def test_zone_target_lands_on_a_free_cell_with_clearance() -> None:
    blocked = (1.0, 1.0, 3.0, 2.6)
    for seed in range(20):
        phase = RobotManager._resolve_target(_manager(blocked, seed), GoToPhase(target="room"))
        assert phase.target == "room"
        assert 1.0 <= phase.pose.position.x <= 3.0 and 2.6 <= phase.pose.position.y <= 3.0
        assert _blocked_distance(phase.pose.position, blocked) >= SAFE - RES


def test_zone_target_without_a_free_cell_names_the_zone() -> None:
    with pytest.raises(ValueError, match="goto target 'room' has no free cell with 0.30 m clearance"):
        RobotManager._resolve_target(_manager((0.5, 0.5, 3.5, 3.5)), GoToPhase(target="room"))


def test_unknown_target_is_left_for_the_ped_follower() -> None:
    phase = GoToPhase(target="pedestrian_3")
    assert RobotManager._resolve_target(_manager(None), phase) is phase


def test_authored_dispatch_pose_of_a_zone_target_is_kept() -> None:
    phase = GoToPhase(target="room", pose=Pose(position=Position(2.5, 1.5), orientation=Orientation.identity()))
    assert RobotManager._resolve_target(_manager(None), phase) is phase
