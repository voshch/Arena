from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(("source", "cls"), [("eth_ucy", "ETHUCY"), ("glow", "GLOW"), ("juelich", "Juelich"), ("madras", "MADRAS"), ("socnavbench", "SocNavBench")])
def test_registry_target_is_the_class(source: str, cls: str) -> None:
    importers = pytest.importorskip("arena_assets.importers")
    pytest.importorskip("arena_simulation_setup.shared.track")
    target = getattr(importlib.import_module(f"task_generator.adapters.{source}"), cls)
    assert importers.IMPORTERS.load("scenario", source) is target
    assert target.meta.kind == "scenario"
