from __future__ import annotations

import asyncio
import typing

import pytest

from arena_runtime.env.catalogs import CATALOGS, LISTERS


def test_every_catalog_service_has_a_lister_and_the_tool_names_the_same_kinds() -> None:
    pytest.importorskip("arena_mcp")
    from arena_runtime.env.episode_mcp import episode_catalog

    kinds = typing.get_args(typing.get_type_hints(episode_catalog)["kind"])

    assert set(LISTERS) == set(CATALOGS) == set(kinds)


def test_worlds_scenarios_and_robots_list_without_an_env() -> None:
    worlds = asyncio.run(LISTERS["worlds"](""))
    scenarios = asyncio.run(LISTERS["scenarios"]("map_empty"))
    robots = asyncio.run(LISTERS["robots"](""))

    assert "map_empty" in worlds
    assert "preflight" in scenarios
    assert robots and all(isinstance(name, str) for name in robots)


def test_the_local_config_trees_list_without_an_env() -> None:
    environments = asyncio.run(LISTERS["environments"](""))
    parametrizeds = asyncio.run(LISTERS["parametrizeds"](""))

    assert isinstance(environments, list)
    assert isinstance(parametrizeds, list)


def test_scenarios_of_a_missing_world_raise_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(LISTERS["scenarios"]("no_such_world_at_all"))
