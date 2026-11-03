"""Tests for how arena viz and arena human resolve an env target against a node namespace."""

import pytest

from arena_cli import human, viz

NODE = "/arena/env_3/arena_env_node"


@pytest.mark.parametrize("matches", [viz._matches, human._matches])
@pytest.mark.parametrize("target", ["3", "env_3", "/arena/env_3", "arena/env_3", NODE])
def test_target_names_the_env(matches, target: str) -> None:
    assert matches(NODE, target)


@pytest.mark.parametrize("matches", [viz._matches, human._matches])
@pytest.mark.parametrize("target", ["30", "env_30", "arena_env_node", "arena"])
def test_target_of_another_env_misses(matches, target: str) -> None:
    assert not matches(NODE, target)
