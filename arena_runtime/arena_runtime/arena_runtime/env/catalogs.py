"""What each env catalog service lists, callable without an env."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import arena_robots.Robot
import arena_simulation_setup.tree.assets.Human
import arena_simulation_setup.tree.assets.Object
import arena_simulation_setup.tree.configs.environment
import arena_simulation_setup.tree.configs.parametrized
import arena_simulation_setup.tree.World as World
from arena_simulation_setup.tree import Identifier

CATALOGS = {
    "worlds": ("QueryWorlds", "query/worlds"),
    "scenarios": ("QueryScenarios", "query/scenarios"),
    "robots": ("QueryRobots", "query/robots"),
    "static_obstacles": ("QueryStaticObstacles", "query/static_obstacles"),
    "dynamic_obstacles": ("QueryDynamicObstacles", "query/dynamic_obstacles"),
    "environments": ("QueryEnvironments", "query/environments"),
    "parametrizeds": ("QueryParametrizeds", "query/parametrizeds"),
}


def _shortnames(identifier: type[Identifier]) -> list[str]:
    return [item.shortname for item in identifier.listall()]


async def _shortnames_async(identifier: type[Identifier], **kwargs: bool) -> list[str]:
    return [item.shortname for item in await identifier.listall_async(**kwargs)]


async def list_worlds(world: str) -> list[str]:
    return _shortnames(World.WorldIdentifier)


async def list_scenarios(world: str) -> list[str]:
    """The scenarios of `world`, FileNotFoundError when it does not exist."""
    return _shortnames(World.WorldIdentifier(world).resolve_sync().scenario)


async def list_robots(world: str) -> list[str]:
    return _shortnames(arena_robots.Robot.RobotIdentifier)


async def list_static_obstacles(world: str) -> list[str]:
    return await _shortnames_async(arena_simulation_setup.tree.assets.Object.ObjectIdentifier, network=True)


async def list_dynamic_obstacles(world: str) -> list[str]:
    return await _shortnames_async(arena_simulation_setup.tree.assets.Human.HumanIdentifier, network=True)


async def list_environments(world: str) -> list[str]:
    return _shortnames(arena_simulation_setup.tree.configs.environment.EnvironmentIdentifier)


async def list_parametrizeds(world: str) -> list[str]:
    return _shortnames(arena_simulation_setup.tree.configs.parametrized.ParametrizedIdentifier)


LISTERS: dict[str, Callable[[str], Awaitable[list[str]]]] = {
    "worlds": list_worlds,
    "scenarios": list_scenarios,
    "robots": list_robots,
    "static_obstacles": list_static_obstacles,
    "dynamic_obstacles": list_dynamic_obstacles,
    "environments": list_environments,
    "parametrizeds": list_parametrizeds,
}
