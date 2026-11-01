from __future__ import annotations

import pytest

from arena_cam import context

ROBOTS = {"jackal": "env_2/jackal/base_link", "go1": "env_2/go1/base"}
HUMANS = {"env_2/level_1/edge_0": 4, "guard": 7}


def _value(key: str) -> str:
    return context.value(key, 2, ROBOTS, HUMANS)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("env", "env_2"),
        ("robot", "env_2/jackal/base_link"),
        ("robot.go1", "env_2/go1/base"),
        ("human.edge_0", "body_env_2_agent_4"),
        ("face.guard", "env_2_agent_7"),
    ],
)
def test_each_token_reads_its_env(key: str, expected: str) -> None:
    assert _value(key) == expected


def test_a_token_is_filled_inside_text() -> None:
    assert context.fill("${env}/camera_link", _value) == "env_2/camera_link"


def test_tokens_lists_the_keys_a_text_names() -> None:
    assert context.tokens("${env}/${robot.go1}") == ["env", "robot.go1"]
    assert context.tokens("env_0/jackal/base_link") == []


@pytest.mark.parametrize(
    ("key", "known"),
    [
        ("robot.ghost", "jackal"),
        ("human.ghost", "guard"),
        ("face.ghost", "guard"),
    ],
)
def test_an_unknown_name_says_what_the_env_has(key: str, known: str) -> None:
    with pytest.raises(LookupError, match=known):
        _value(key)


def test_a_robotless_env_says_so() -> None:
    with pytest.raises(LookupError, match="no robots"):
        context.value("robot", 2, {}, {})


def test_a_roster_name_matches_on_a_whole_path_part() -> None:
    assert context.human_id({"env_0/edge_01": 1}, "edge_0") is None
