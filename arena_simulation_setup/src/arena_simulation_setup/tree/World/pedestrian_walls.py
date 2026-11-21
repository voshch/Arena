"""Pedestrian wall segments of a world in the scenario frame, as the runtime builds them."""

from __future__ import annotations

from arena_simulation_setup.tree.World.model import World

WallSegment = tuple[tuple[float, float], tuple[float, float]]


def wall_segments_of_world(world: World) -> list[WallSegment]:
    """Wall segments the humansim receives: every level's closed walls, positions only."""
    segments: list[WallSegment] = []
    for level in world.all_levels:
        walls = [*level.all_walls, *(wall for stair in level.stairs for wall in stair.open_walls())]
        segments.extend(((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)) for wall in walls)
    return segments
