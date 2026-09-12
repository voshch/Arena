"""Unit tests for Agent E: per-level directory shape, selective loading, WorldIdentifier.parse."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from arena_simulation_setup.tree.World.World import (
    ElevatorDescriptor,
    Level,
    LevelDescription,
    MultiLevelWorldView,
    WorldDescription,
    WorldIdentifier,
)
from arena_simulation_setup.utils.geometry import Position

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_zone(name: str = "room") -> LevelDescription.Zone:
    return LevelDescription.Zone(
        name=name,
        corners=[
            Position(0.0, 0.0),
            Position(5.0, 0.0),
            Position(5.0, 5.0),
            Position(0.0, 5.0),
        ],
    )


def _write_level_yaml(path: Path, zones: list[LevelDescription.Zone] | None = None) -> None:
    """Write a minimal LevelDescription YAML to *path*."""
    data: dict = {"zones": []}
    if zones:
        from arena_simulation_setup.utils.cattrs import converter

        data["zones"] = [converter.unstructure(z) for z in zones]
    path.write_text(yaml.safe_dump(data))


# ---------------------------------------------------------------------------
# WorldIdentifier.parse
# ---------------------------------------------------------------------------


def test_parse_no_filter():
    name, filt = WorldIdentifier.parse("myworld")
    assert name == "myworld"
    assert filt is None


def test_parse_single_level():
    name, filt = WorldIdentifier.parse("myworld[1]")
    assert name == "myworld"
    assert filt == {"1"}


def test_parse_multi_level():
    name, filt = WorldIdentifier.parse("myworld[1,3]")
    assert name == "myworld"
    assert filt == {"1", "3"}


def test_parse_with_spaces():
    name, filt = WorldIdentifier.parse("myworld[ 1 , 3 ]")
    assert name == "myworld"
    assert filt == {"1", "3"}


def test_parse_empty_brackets_returns_none_filter():
    name, filt = WorldIdentifier.parse("myworld[]")
    assert name == "myworld"
    assert filt is None


# ---------------------------------------------------------------------------
# WorldDescription.validate: level-qualified destinations
# ---------------------------------------------------------------------------


def _make_world_two_floors_linked() -> WorldDescription:
    """Two levels (0, 1) each with one elevator, linked 0->1 and 1->0."""
    from arena_simulation_setup.shared.world import Elevator

    elev0 = Elevator(name="elev_a", position=Position(1.0, 1.0))
    elev1 = Elevator(name="elev_b", position=Position(1.0, 1.0))

    zone0 = LevelDescription.Zone(
        name="z0",
        corners=[Position(0, 0), Position(10, 0), Position(10, 10), Position(0, 10)],
        elevators=[elev0],
    )
    zone1 = LevelDescription.Zone(
        name="z1",
        corners=[Position(0, 0), Position(10, 0), Position(10, 10), Position(0, 10)],
        elevators=[elev1],
    )

    level0 = Level(zones=[zone0])
    fe0 = ElevatorDescriptor(name="elev_a")
    fe0.add_destination("elev_b", "1")
    level0.levelElevators = [fe0]

    level1 = Level(zones=[zone1])
    fe1 = ElevatorDescriptor(name="elev_b")
    fe1.add_destination("elev_a", "0")
    level1.levelElevators = [fe1]

    return WorldDescription(levels={"0": level0, "1": level1})


def test_validate_linked_two_levels_passes():
    wd = _make_world_two_floors_linked()
    warnings = wd.validate()
    assert warnings == []


def test_validate_dangling_dest_hard_error_when_level_loaded():
    from arena_simulation_setup.shared.world import Elevator

    elev0 = Elevator(name="elev_a", position=Position(1.0, 1.0))
    zone0 = LevelDescription.Zone(
        name="z0",
        corners=[Position(0, 0), Position(10, 0), Position(10, 10), Position(0, 10)],
        elevators=[elev0],
    )
    level0 = Level(zones=[zone0])
    fe0 = ElevatorDescriptor(name="elev_a")
    fe0.add_destination("nonexistent_elev", "1")
    level0.levelElevators = [fe0]

    level1 = Level(zones=[LevelDescription.Zone(name="z1", corners=[Position(0, 0), Position(10, 0), Position(10, 10), Position(0, 10)])])

    wd = WorldDescription(levels={"0": level0, "1": level1})
    with pytest.raises(RuntimeError, match="nonexistent_elev"):
        wd.validate()


def test_validate_dangling_dest_warn_when_level_not_loaded(caplog: pytest.LogCaptureFixture):
    from arena_simulation_setup.shared.world import Elevator

    elev0 = Elevator(name="elev_a", position=Position(1.0, 1.0))
    zone0 = LevelDescription.Zone(
        name="z0",
        corners=[Position(0, 0), Position(10, 0), Position(10, 10), Position(0, 10)],
        elevators=[elev0],
    )
    level0 = Level(zones=[zone0])
    fe0 = ElevatorDescriptor(name="elev_a")
    fe0.add_destination("elev_b", "99")
    level0.levelElevators = [fe0]

    wd = WorldDescription(levels={"0": level0})
    with caplog.at_level(logging.WARNING):
        warns = wd.validate(loaded_level_ids={"0"})

    assert len(warns) == 1
    assert "selective load" in warns[0]
    assert "99" in warns[0]


# ---------------------------------------------------------------------------
# Auto-discovery: per-level subdirs
# ---------------------------------------------------------------------------


def test_auto_discovery_two_levels(tmp_path: Path):
    world_dir = tmp_path / "myworld"
    for fid in ("0", "1"):
        fd = world_dir / fid
        fd.mkdir(parents=True)
        _write_level_yaml(fd / "world.yaml", zones=[_minimal_zone(f"room_{fid}")])

    view = MultiLevelWorldView(world_dir)
    wd = view.load(validate=False)

    assert set(wd.levels.keys()) == {"0", "1"}


def test_auto_discovery_zone_names(tmp_path: Path):
    world_dir = tmp_path / "myworld"
    for fid in ("0", "1"):
        fd = world_dir / fid
        fd.mkdir(parents=True)
        _write_level_yaml(fd / "world.yaml", zones=[_minimal_zone(f"room_{fid}")])

    view = MultiLevelWorldView(world_dir)
    wd = view.load(validate=False)

    zones_level0 = [z.name for z in wd.levels["0"].zones]
    assert "room_0" in zones_level0


# ---------------------------------------------------------------------------
# Auto-wrap legacy: single root world.yaml -> level '0'
# ---------------------------------------------------------------------------


def test_legacy_single_file_loads_as_level_0(tmp_path: Path):
    world_dir = tmp_path / "legacyworld"
    world_dir.mkdir()
    _write_level_yaml(world_dir / "world.yaml", zones=[_minimal_zone("hall")])

    view = MultiLevelWorldView(world_dir)
    wd = view.load(validate=False)

    assert "0" in wd.levels
    assert len(wd.levels) == 1


def test_legacy_single_file_zone_preserved(tmp_path: Path):
    world_dir = tmp_path / "legacyworld"
    world_dir.mkdir()
    _write_level_yaml(world_dir / "world.yaml", zones=[_minimal_zone("my_hall")])

    view = MultiLevelWorldView(world_dir)
    wd = view.load(validate=False)

    zones = [z.name for z in wd.levels["0"].zones]
    assert "my_hall" in zones


# ---------------------------------------------------------------------------
# Floor filter: selective load
# ---------------------------------------------------------------------------


def test_level_filter_returns_only_requested(tmp_path: Path):
    world_dir = tmp_path / "multilevel"
    for fid in ("1", "2", "3"):
        fd = world_dir / fid
        fd.mkdir(parents=True)
        _write_level_yaml(fd / "world.yaml", zones=[_minimal_zone(f"room_{fid}")])

    view = MultiLevelWorldView(world_dir)
    wd = view.load(validate=False, level_filter={"1"})

    assert set(wd.levels.keys()) == {"1"}


def test_level_filter_excludes_others(tmp_path: Path):
    world_dir = tmp_path / "multilevel"
    for fid in ("1", "2", "3"):
        fd = world_dir / fid
        fd.mkdir(parents=True)
        _write_level_yaml(fd / "world.yaml", zones=[_minimal_zone(f"room_{fid}")])

    view = MultiLevelWorldView(world_dir)
    wd = view.load(validate=False, level_filter={"1", "3"})

    assert "2" not in wd.levels
    assert "1" in wd.levels
    assert "3" in wd.levels


# ---------------------------------------------------------------------------
# WorldDescription.export per-level shape
# ---------------------------------------------------------------------------


def test_export_produces_per_level_dirs(tmp_path: Path):
    wd = WorldDescription.from_levels(LevelDescription(zones=[_minimal_zone()]))
    tarball = wd.export()
    names = [m.name for m in tarball.getmembers()]
    assert any("0/world.yaml" in n for n in names)
    assert any("0/map.png" in n for n in names)
    assert any("0/map.yaml" in n for n in names)


def test_export_no_map_levels_path(tmp_path: Path):
    wd = WorldDescription.from_levels(LevelDescription(zones=[_minimal_zone()]))
    tarball = wd.export()
    names = [m.name for m in tarball.getmembers()]
    assert not any("map/levels" in n for n in names)
