from __future__ import annotations

import pytest
import yaml

from task_generator.tasks.obstacles.edge_case.effects import Intercept, Rally
from task_generator.tasks.obstacles.edge_case.scenario_block import Block, BlockError, parse, read


def write(tmp_path, block):
    doc = {"robots": [{"start": [0, 0, 0]}], "dynamic": []}
    if block is not None:
        doc["edge_case"] = block
    p = tmp_path / "scenario.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_reads_a_block(tmp_path) -> None:
    block = read(write(tmp_path, {"id": "c", "effects": [{"type": "intercept"}]}))
    assert block is not None and block.id == "c" and isinstance(block.effects[0], Intercept)


def test_no_block_is_silence_not_an_error(tmp_path) -> None:
    assert read(write(tmp_path, None)) is None


def test_unknown_key_is_refused(tmp_path) -> None:
    with pytest.raises(BlockError, match="unknown key"):
        read(write(tmp_path, {"id": "c", "knob": "vision_fov"}))


def test_malformed_effect_is_refused_at_read_time(tmp_path) -> None:
    with pytest.raises(BlockError, match="effects\\[0\\]"):
        read(write(tmp_path, {"id": "c", "effects": [{"type": "rally"}]}))


def test_labels_and_provenance_round_trip() -> None:
    block = parse({
        "id": "h_004", "prompt_id": "h_004", "steps": ["A:blackout", "C:approach"], "base": "normal_a",
        "provenance": {"model": "gemini"}, "decisions": {"A": {"case": "blackout"}},
        "effects": [{"type": "rally", "target": "exit", "when": {"at": 8}}],
    })
    assert block.steps == ("A:blackout", "C:approach") and block.provenance == {"model": "gemini"}
    assert isinstance(block.effects[0], Rally) and block.effects[0].when.at == 8.0
    assert "rally" in block.describe()


def test_empty_block_is_the_base_arm() -> None:
    assert Block().empty and parse({"id": "base"}).empty
    assert not parse({"objects": "./objects.yaml"}).empty


def test_owned_agents_are_carried() -> None:
    block = parse({"id": "c", "owned": {"D": ["d_1", "d_2"], "A": []}})
    assert block.owned == {"D": ("d_1", "d_2"), "A": ()}
    with pytest.raises(BlockError, match="must be mappings"):
        parse({"id": "c", "owned": ["d_1"]})
