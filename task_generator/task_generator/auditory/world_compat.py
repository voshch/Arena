from __future__ import annotations

from arena_simulation_setup.tree.World import LevelDescription, WorldDescription


def world_zones(world: WorldDescription) -> tuple[LevelDescription.Zone, ...]:
    """Every zone of the world, ordered by level id."""
    return tuple(
        zone
        for _, zones in world_zone_groups(world)
        for zone in zones
    )


def world_zone_groups(
    world: WorldDescription,
) -> tuple[tuple[str, tuple[LevelDescription.Zone, ...]], ...]:
    """Level-scoped zones, so overlapping building floors stay uncoupled."""
    return tuple(
        (level_id, tuple(level.zones))
        for level_id, level in sorted(world.levels.items())
    )
