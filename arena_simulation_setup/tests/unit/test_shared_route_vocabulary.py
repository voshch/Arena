from __future__ import annotations

import math
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from arena_simulation_setup.shared.route import ORDINALS, WORDINGS, instruct, level_nouns, portals, zone_shapes
from arena_simulation_setup.shared.task import GoToPhase
from arena_simulation_setup.tree.World.World import LevelDescription
from arena_simulation_setup.utils.cattrs import converter
from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

WORLDS = Path(__file__).parents[2] / "worlds"
ZONED = sorted(path.parents[1].name for path in WORLDS.glob("*/0/world.yaml") if len(yaml.safe_load(path.read_text()).get("zones") or []) > 1)

TEMPLATE_WORDS = frozenset("go to walk through then take the door on your left right into ahead turn around and continue straight about meter meters stop next just inside here".split()) | frozenset(ORDINALS)

ROOM = r"the [a-z0-9() -]+?"
SIDE = r"(left|right)"
CROSS = rf"(take the (({'|'.join(ORDINALS)}) )?door on your {SIDE} into {ROOM}|go through the door ahead into {ROOM}|turn {SIDE} into {ROOM}|continue straight into {ROOM}|turn around and (go through the door|continue) into {ROOM})"
TURN = rf"(turn ({SIDE}|around) and )?"
SENTENCES = [
    re.compile(rf"go to {ROOM}\."),
    re.compile(rf"{TURN}walk through {ROOM}, then {CROSS}\."),
    re.compile(rf"{CROSS}\."),
    re.compile(rf"{TURN}walk about [0-9]+ meters?( into {ROOM})? and stop( next to {ROOM})?\."),
    re.compile(rf"stop( next to {ROOM})? (just inside {ROOM}|here)\."),
]


def _level(world: str) -> LevelDescription:
    return converter.structure(yaml.safe_load((WORLDS / world / "0" / "world.yaml").read_text()), LevelDescription)


def _goto(target: str, x: float, y: float) -> GoToPhase:
    return GoToPhase(target=target, pose=Pose(position=Position(x, y), orientation=Orientation.identity()))


def instructions(level: LevelDescription) -> Iterator[str]:
    """Every instruction between two zone interiors from four headings in both wordings, and the directions to every static entity."""
    spots = {name: shape.representative_point().coords[0] for name, shape in zone_shapes(level).items()}
    found = portals(level)
    for start in spots.values():
        for target, goal in spots.items():
            for quarter in range(4):
                for wording in WORDINGS:
                    yield instruct(_goto(target, *goal), level, (*start, quarter * math.pi / 2.0), wording, found=found)["text"]
    first = next(iter(spots.values()))
    for zone in level.zones:
        for entity in zone.entities.static:
            yield instruct(_goto(zone.name, entity.pose.position.x, entity.pose.position.y), level, (*first, 0.0), "route", found=found)["text"]


def test_some_shipped_world_has_zones():
    assert {"hospital_1", "office_1"} <= set(ZONED)


@pytest.mark.parametrize("world", ZONED)
def test_every_sentence_follows_a_template(world):
    for text in instructions(_level(world)):
        for sentence in re.split(r"(?<=\.) (?=[A-Z])", text):
            assert sentence[0].isupper(), text
            assert any(pattern.fullmatch(sentence.lower()) for pattern in SENTENCES), sentence


@pytest.mark.parametrize("world", ZONED)
def test_words_are_template_words_nouns_or_numbers(world):
    level = _level(world)
    nouns = {word for spoken in level_nouns(level).values() for word in spoken.split()}
    for text in instructions(level):
        words = set(re.sub(r"[.,]", "", text.lower()).split())
        assert {word for word in words - TEMPLATE_WORDS - nouns if not word.isdigit()} == set(), text


def test_shipped_worlds_exercise_the_whole_template_vocabulary():
    seen = {word for world in ZONED for text in instructions(_level(world)) for word in re.sub(r"[.,]", "", text.lower()).split()}
    assert TEMPLATE_WORDS - seen <= {"fourth", "fifth"}
