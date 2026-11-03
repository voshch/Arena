"""The env slot extent covers every level of a multi-level world in the compacted frame."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_simulation_setup")

from arena_simulation_setup.tree.World import WorldIdentifier  # noqa: E402


def _extent(floors):
    floors = list(floors)
    return (
        min(f.pos.x - f.x_length / 2 for f in floors),
        min(f.pos.y - f.y_length / 2 for f in floors),
        max(f.pos.x + f.x_length / 2 for f in floors),
        max(f.pos.y + f.y_length / 2 for f in floors),
    )


def test_compacted_floors_cover_all_levels():
    try:
        view = WorldIdentifier("mechanisms_elevator").resolve_sync()
    except Exception as e:
        pytest.skip(f"mechanisms_elevator not resolvable: {e!r}")
    world = view.load(validate=False)
    origins = view.level_origins()
    assert origins and origins["1"] != (0.0, 0.0)
    level_local = _extent(f for level in world.all_levels for f in level.all_floors)
    compacted = _extent(world.compact_world(origins).all_floors)
    assert compacted[1] <= origins["1"][1] < level_local[1]
