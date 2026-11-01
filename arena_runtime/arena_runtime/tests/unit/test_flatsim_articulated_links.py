"""Flatsim outlines of articulated link obstacles: annotated height, carried along by the joints."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

PREFIX = "env_0/"


def _chest(root: Path):
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier, ObjectView
    from task_generator.shared import Obstacle, Pose, Position

    bundle = root / "chest"
    for link, box in (("base", [[-0.3, 0.3], [-0.2, 0.2], [0.0, 0.4]]), ("lid", [[-0.3, 0.3], [-0.2, 0.2], [0.4, 0.42]])):
        (bundle / "links" / link).mkdir(parents=True)
        (bundle / "links" / link / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": box}))
    articulation = {
        "links": {"base": ["base"], "lid": ["lid"]},
        "joints": [{"name": "lid", "type": "revolute", "parent": "base", "child": "lid", "origin": [0.0, 0.2, 0.42], "axis": [-1, 0, 0], "limit": [0.0, 1.5]}],
        "states": {"closed": {"lid": 0.0}, "open": {"lid": 1.5}},
    }
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.3, 0.3], [-0.2, 0.2], [0.0, 0.42]], "articulation": articulation}))
    return Obstacle(name="chest", model=ObjectIdentifier.inline(ObjectView(bundle), name="chest"), pose=Pose(Position(1.0, 1.0)))


def test_link_outlines_sit_at_their_annotated_height_and_swing_with_their_joint(on_flatsim, tmp_path: Path):
    import attrs

    async def body(sim, world):
        chest = _chest(tmp_path)
        assert list(await sim.spawn_objects([chest])) == [True]
        assert {name for name in world.names if name.startswith(PREFIX)} == {f"{PREFIX}chest__base", f"{PREFIX}chest__lid"}
        assert world._extent(f"{PREFIX}chest__base") == pytest.approx((0.0, 0.4))
        assert world._extent(f"{PREFIX}chest__lid") == pytest.approx((0.4, 0.42))
        assert list(await sim.move_objects([attrs.evolve(chest, joints="open")])) == [True]
        assert world._extent(f"{PREFIX}chest__lid") == pytest.approx((0.42 - 0.02 * math.cos(1.5), 0.42 + 0.4 * math.sin(1.5)))
        assert list(await sim.delete_objects([chest])) == [True]
        assert not any(name.startswith(PREFIX) for name in world.names)

    on_flatsim(body)
