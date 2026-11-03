from __future__ import annotations

import asyncio
import importlib
import math

import pytest
from arena_simulation_setup.tree.World import WorldIdentifier
from task_generator.tasks.registry import (
    _REGISTRY_NAMESPACE,
    TaskModeMeta,
    TaskModeRegistry,
    default_scenario,
    identifier_to_available,
    identifier_to_available_async,
)


class _ModeA:
    pass


class _ModeB:
    pass


def _scenario_dirs(world: str) -> set[str]:
    view = WorldIdentifier(world).resolve_sync()
    return {entry.name for entry in (view.path / "scenarios").iterdir() if entry.is_dir()}


def test_register_returns_loader_and_records_meta() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    ns = _REGISTRY_NAMESPACE("mode_a")

    def _schema(node: object, namespace: object) -> None:
        del node, namespace

    def _load() -> type:
        return _ModeA

    assert registry.register("a", namespace=ns, schema=_schema)(_load) is _load
    assert registry.meta("a") == TaskModeMeta(namespace=ns, schema=_schema)


def test_meta_schema_defaults_to_none() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    registry.register("a", namespace=_REGISTRY_NAMESPACE("mode_a"))(lambda: _ModeA)
    assert registry.meta("a").schema is None


def test_get_invokes_loader_once_and_caches() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    calls: list[str] = []

    @registry.register("a", namespace=_REGISTRY_NAMESPACE("mode_a"))
    def _load() -> type:
        calls.append("a")
        return _ModeA

    assert calls == []
    assert registry.get("a") is _ModeA
    assert registry.get("a") is _ModeA
    assert calls == ["a"]


def test_meta_does_not_invoke_loader() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    calls: list[str] = []

    @registry.register("a", namespace=_REGISTRY_NAMESPACE("mode_a"))
    def _load() -> type:
        calls.append("a")
        return _ModeA

    registry.meta("a")
    list(registry.keys())
    assert "a" in registry
    assert calls == []


def test_keys_and_membership_track_registrations() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    assert list(registry.keys()) == []
    assert "a" not in registry
    registry.register("a", namespace=_REGISTRY_NAMESPACE("mode_a"))(lambda: _ModeA)
    registry.register("b", namespace=_REGISTRY_NAMESPACE("mode_b"))(lambda: _ModeB)
    assert list(registry.keys()) == ["a", "b"]
    assert "a" in registry
    assert "b" in registry
    assert "c" not in registry
    assert registry.get("b") is _ModeB


def test_duplicate_key_raises_and_keeps_first_meta() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    first = _REGISTRY_NAMESPACE("first")
    registry.register("a", namespace=first)(lambda: _ModeA)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("a", namespace=_REGISTRY_NAMESPACE("second"))(lambda: _ModeB)
    assert registry.meta("a").namespace == first
    assert registry.get("a") is _ModeA


def test_registries_are_independent() -> None:
    first: TaskModeRegistry[str, type] = TaskModeRegistry()
    second: TaskModeRegistry[str, type] = TaskModeRegistry()
    first.register("a", namespace=_REGISTRY_NAMESPACE("mode_a"))(lambda: _ModeA)
    assert "a" not in second
    second.register("a", namespace=_REGISTRY_NAMESPACE("mode_a"))(lambda: _ModeB)
    assert first.get("a") is _ModeA
    assert second.get("a") is _ModeB


def test_episode_namespace_is_refused_before_registration() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    with pytest.raises(ValueError, match="task.episode"):
        registry.register("episode", namespace=_REGISTRY_NAMESPACE("episode"))
    assert "episode" not in registry


def test_episode_subnamespace_is_allowed() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    registry.register("a", namespace=_REGISTRY_NAMESPACE("episode", "a"))(lambda: _ModeA)
    assert "a" in registry


def test_get_unknown_key_raises_key_error() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    registry.register("a", namespace=_REGISTRY_NAMESPACE("mode_a"))(lambda: _ModeA)
    with pytest.raises(KeyError, match="'missing'"):
        registry.get("missing")


def test_meta_unknown_key_raises_key_error() -> None:
    registry: TaskModeRegistry[str, type] = TaskModeRegistry()
    with pytest.raises(KeyError):
        registry.meta("missing")


def test_identifier_to_available_lists_world_scenarios() -> None:
    scenario = WorldIdentifier("map_empty").resolve_sync().scenario
    names = list(identifier_to_available(scenario))
    assert len(names) == len(set(names))
    assert set(names) == _scenario_dirs("map_empty")


def test_identifier_to_available_async_matches_sync() -> None:
    scenario = WorldIdentifier("reception").resolve_sync().scenario
    names = asyncio.run(identifier_to_available_async(scenario))
    assert isinstance(names, list)
    assert sorted(names) == sorted(identifier_to_available(scenario))
    assert set(names) == _scenario_dirs("reception")


def test_default_scenario_prefers_default() -> None:
    assert len(_scenario_dirs("map_empty")) > 1
    assert default_scenario("map_empty") == "default"


def test_default_scenario_falls_back_to_first_scenario() -> None:
    scenarios = _scenario_dirs("three_storied_residential")
    assert "default" not in scenarios
    assert default_scenario("three_storied_residential") in scenarios


def test_declare_mode_schemas_declares_what_every_registered_schema_declares() -> None:
    pytest.importorskip("rclpy")
    for package in ("modules", "obstacles", "robots"):
        importlib.import_module(f"task_generator.tasks.{package}")
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer
    from arena_runtime.env.task_layer import declare_mode_schemas

    from task_generator.modes import Modes
    from task_generator.tasks import layer
    from task_generator.tasks.registry import MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES

    def declared(node: ROSParamServer) -> dict[str, str]:
        return {name: repr(node.get_parameter(name).value) for name in node.get_parameters_by_prefix("")}

    registries = (MODULE_MODES, OBSTACLES_MODES, ROBOTS_MODES)
    reference = ROSParamServer("schemas_reference")
    node = ROSParamServer("schemas_probe")
    try:
        builtin = set(node.get_parameters_by_prefix(""))
        for registry in registries:
            for key in registry.keys():
                meta = registry.meta(key)
                if meta.schema is not None:
                    meta.schema(reference, meta.namespace)
        declare_mode_schemas(layer, node)
        expected = {name: value for name, value in declared(reference).items() if name not in builtin}
        actual = {name: value for name, value in declared(node).items() if name not in builtin}
        assert expected
        assert actual == expected
        assert OBSTACLES_MODES.meta(Modes.TM_Obstacles.SCENARIO).namespace == ROBOTS_MODES.meta(Modes.TM_Robots.SCENARIO).namespace
        stationary = ROBOTS_MODES.meta(Modes.TM_Robots.STATIONARY).namespace
        assert math.isnan(node.get_parameter(stationary("pos_x")).value)
        assert node.get_parameter(stationary("pos_theta")).value == 0.0
    finally:
        reference.destroy_node()
        node.destroy_node()
