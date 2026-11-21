"""Offline pedestrian walls of a concourse world, against the walls the runtime builds."""

from __future__ import annotations

import asyncio
import math
import random

from arena_simulation_setup.generative.world.concourse import WorldGeneratorConcourse
from arena_simulation_setup.tree.World.pedestrian_walls import wall_segments_of_world


def test_concourse_pedestrian_walls_surround_a_pillar() -> None:
    config = {"length": 24.0, "width": 4.0, "pillars": [{"at_s": 12.0, "radius": 0.35}]}
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    segments = wall_segments_of_world(world)
    assert len(segments) > 4
    near_pillar = [point for segment in segments for point in segment if math.dist(point, (12.0, 0.0)) < 0.6]
    assert len(near_pillar) >= 3


def test_pedestrian_walls_match_the_closed_walls_the_runtime_receives() -> None:
    config = {"length": 24.0, "width": 4.0, "pillars": [{"at_s": 12.0, "radius": 0.35}]}
    world = WorldGeneratorConcourse(config, random.Random(42)).compute()
    level = world.all_levels[0]
    closed = asyncio.run(level.closed_walls())
    runtime_pairs = sorted(tuple(sorted([(round(w.start.x, 4), round(w.start.y, 4)), (round(w.end.x, 4), round(w.end.y, 4))])) for w in closed)
    offline_pairs = sorted(tuple(sorted([(round(a[0], 4), round(a[1], 4)), (round(b[0], 4), round(b[1], 4))])) for a, b in wall_segments_of_world(world))
    assert offline_pairs == runtime_pairs
