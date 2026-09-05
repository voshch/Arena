"""World reference closure, offline: no wall kinds, so nothing reaches a network resolver."""

from __future__ import annotations

from arena_simulation_setup.tree.World.World import MultiLevelWorldView

_WORLD = """
zones:
  - description: "test zone"
    name: "z"
    walls:
      - start: {x: 0.0, y: 0.0}
        end: {x: 1.0, y: 0.0}
    corners:
      - [0.0, 0.0]
      - [1.0, 0.0]
      - [1.0, 1.0]
      - [0.0, 1.0]
    mat: ""
    entities:
      static: []
"""


def _world(tmp_path):
    (tmp_path / 'world.yaml').write_text(_WORLD)
    return MultiLevelWorldView(tmp_path)


def test_closure_skips_unset_material(tmp_path):
    """`mat: ""` is an unset field, not a reference to an asset called ''."""
    identifiers = list(_world(tmp_path).identifiers(scenarios=False))
    assert [i for i in identifiers if not i.name] == []


def test_closure_yields_the_materials_that_are_set(tmp_path):
    """The ceiling and wall materials fall back to named defaults, and those are references."""
    identifiers = list(_world(tmp_path).identifiers(scenarios=False))
    assert {i.shortname for i in identifiers} == {'Common/Material/Concrete_Smooth', 'Common/Material/Marble'}
