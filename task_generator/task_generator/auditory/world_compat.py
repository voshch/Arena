from __future__ import annotations

from collections.abc import Iterable


def world_zones(world: object) -> tuple[object, ...]:
    """Return zones from legacy single-level and current multi-level worlds."""
    legacy_zones = getattr(world, "zones", None)
    if legacy_zones is not None:
        return tuple(legacy_zones)

    levels = getattr(world, "levels", None)
    if levels is None:
        raise TypeError(
            f"{type(world).__name__} exposes neither 'zones' nor 'levels'"
        )

    if isinstance(levels, dict):
        ordered_levels: Iterable[object] = (
            levels[level_id] for level_id in sorted(levels)
        )
    else:
        ordered_levels = levels

    zones = tuple(
        zone
        for level in ordered_levels
        for zone in getattr(level, "zones", ())
    )
    return zones


def world_zone_groups(world: object) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """Return level-scoped zones without coupling overlapping building floors."""
    legacy_zones = getattr(world, "zones", None)
    if legacy_zones is not None:
        return (("0", tuple(legacy_zones)),)
    levels = getattr(world, "levels", None)
    if levels is None:
        raise TypeError(
            f"{type(world).__name__} exposes neither 'zones' nor 'levels'"
        )
    items = (
        sorted(levels.items())
        if isinstance(levels, dict)
        else [(str(index), level) for index, level in enumerate(levels)]
    )
    return tuple(
        (str(level_id), tuple(getattr(level, "zones", ())))
        for level_id, level in items
    )
