"""Juelich importer: archive runs into replay scenarios, a world per distinct lab layout of a set, read by the native loaders."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

SET = "hermes_bottleneck"
CORRIDOR = "POLYGON ((0 -2, 10 -2, 10 2, 0 2, 0 -2), (6 -0.5, 7 -0.5, 7 0.5, 6 0.5, 6 -0.5))"
ROOM = "POLYGON ((0 0, 6 0, 6 6, 0 6, 0 0))"
FPS = 16.0


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")
    pytest.importorskip("arena_simulation_setup.shared.track")


def _rows(ident: int, first: int, last: int, start: tuple[float, float], end: tuple[float, float]) -> list[str]:
    rows = []
    for frame in range(first, last + 1):
        u = (frame - first) / (last - first)
        rows.append(f"{ident} {frame} {start[0] + u * (end[0] - start[0]):.3f} {start[1] + u * (end[1] - start[1]):.3f} 175.0")
    return rows


def seed(cache: Path, doi: str | None = None) -> Path:
    from task_generator.adapters.juelich import SETS

    entry = SETS[SET]
    directory = cache / SET
    directory.mkdir(parents=True)
    metadata = {
        "experiment": {
            "title": "Bottleneck",
            "archive_doi": doi,
            "persons": [{"given_name": "Armin", "family_name": "Seyfried"}, {"given_name": "Maik", "family_name": "Boltes"}],
            "run": [{"run_name": "b1", "wkt_geometry": CORRIDOR}, {"run_name": "b2", "wkt_geometry": CORRIDOR}, {"run_name": "b3", "wkt_geometry": ROOM}],
        }
    }
    (directory / Path(entry.metadata).name).write_text(json.dumps(metadata))
    with zipfile.ZipFile(directory / Path(entry.trajectories).name, "w") as z:
        z.writestr("b1.txt", "\n".join([*_rows(1, 0, 32, (100.0, 0.0), (300.0, 0.0)), *_rows(2, 16, 48, (500.0, -100.0), (500.0, 100.0))]) + "\n")
        z.writestr("b2.txt", "\n".join(_rows(4, 0, 32, (900.0, 0.0), (100.0, 0.0))) + "\n")
        z.writestr("trajectories/b3.txt", "\n".join(["# id frame x y z, positions [in m]", *_rows(7, 0, 16, (1.0, 3.0), (3.0, 3.0))]) + "\n")
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from task_generator.adapters.juelich import Juelich

    return Juelich(cache=seed(tmp_path / "cache"))


def _files(root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def _world(out: Path, name: str):
    from arena_simulation_setup.tree.World import WorldIdentifier

    return WorldIdentifier(name).load(out / name)


def test_registry_target_is_the_class() -> None:
    from arena_assets.importers import IMPORTERS

    from task_generator.adapters.juelich import Juelich

    assert IMPORTERS.load("scenario", "juelich") is Juelich
    assert Juelich.meta.kind == "scenario"


def test_items_name_a_set_and_a_run(importer) -> None:
    from task_generator.adapters.juelich import split_item

    assert split_item(f"{SET}/b1") == (SET, "b1")
    with pytest.raises(KeyError, match="items are <set>/<run> with the sets basigo_bidirectional, basigo_crossing_90"):
        split_item("hermes_corner/c1")
    with pytest.raises(KeyError, match=f"no run 'b9' in the Juelich set {SET}"):
        importer.fetch(f"{SET}/b9")


def test_runs_of_one_layout_share_a_world_and_another_layout_gets_its_own(importer, tmp_path: Path) -> None:
    out = tmp_path / "worlds"
    assert importer.import_item(f"{SET}/b1", out) == out / f"juelich_{SET}_0" / "scenarios" / "b1"
    assert importer.import_item(f"{SET}/b2", out) == out / f"juelich_{SET}_0" / "scenarios" / "b2"
    assert importer.import_item(f"{SET}/b3", out) == out / f"juelich_{SET}_1" / "scenarios" / "b3"
    assert _files(out / f"juelich_{SET}_0") == ["ATTRIBUTION.yaml", "map/0.png", "map/0.yaml", "scenarios/b1/ATTRIBUTION.yaml", "scenarios/b1/ped_1.csv", "scenarios/b1/ped_2.csv", "scenarios/b1/scenario.yaml", "scenarios/b2/ATTRIBUTION.yaml", "scenarios/b2/ped_4.csv", "scenarios/b2/scenario.yaml", "world.yaml"]
    assert sorted(scenario.name for scenario in _world(out, f"juelich_{SET}_0").scenario.listall()) == ["b1", "b2"]


def test_floor_is_the_layout_with_its_pillar_walled_in(importer, tmp_path: Path) -> None:
    import shapely

    out = tmp_path / "worlds"
    importer.import_item(f"{SET}/b1", out)
    level = _world(out, f"juelich_{SET}_0").load(validate=True).levels["0"]
    floor = shapely.union_all([shapely.Polygon([(point.x, point.y) for point in zone.polygon]) for zone in level.zones])
    assert floor.symmetric_difference(shapely.from_wkt(CORRIDOR)).area == pytest.approx(0.0, abs=1e-6)
    walls = shapely.MultiLineString([((wall.start.x, wall.start.y), (wall.end.x, wall.end.y)) for wall in level.all_walls])
    assert walls.length == pytest.approx(2 * (10 + 4) + 2 * (1 + 1))
    assert walls.covers(shapely.from_wkt(CORRIDOR).interiors[0])


def test_tracks_are_in_meters_whether_the_file_is_in_centimeters_or_says_meters(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import Track

    out = tmp_path / "worlds"
    walker = Track.load(importer.import_item(f"{SET}/b1", out) / "ped_1.csv")
    assert (walker.start, walker.end) == pytest.approx((0.0, 2.0))
    assert (walker.state(1.0).x, walker.state(1.0).y) == pytest.approx((2.0, 0.0), abs=1e-3)
    late = Track.load(out / f"juelich_{SET}_0" / "scenarios" / "b1" / "ped_2.csv")
    assert late.start == pytest.approx(16 / FPS)
    metric = Track.load(importer.import_item(f"{SET}/b3", out) / "ped_7.csv")
    assert (metric.state(0.5).x, metric.state(0.5).y) == pytest.approx((2.0, 3.0), abs=1e-3)


def test_scenario_loads_as_native_pedestrians_with_their_stays(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import track_file

    out = tmp_path / "worlds"
    directory = importer.import_item(f"{SET}/b1", out)
    scenario = _world(out, f"juelich_{SET}_0").scenario("b1").resolve_sync().load()
    assert scenario.robots == []
    assert [human.name for human in scenario.humans] == ["ped_1", "ped_2"]
    for human in scenario.humans:
        assert track_file(human) == directory / f"{human.name}.csv"
        (stay,) = human.stays
        assert (stay.legs[-1].position.x, stay.legs[-1].position.y) == pytest.approx((human.waypoints[-1].x, human.waypoints[-1].y), abs=1e-3)
    assert scenario.humans[0].stays[0].velocity == pytest.approx((1.0, 0.0), abs=1e-3)


def test_attribution_names_the_experimenters_and_the_archive(importer, tmp_path: Path) -> None:
    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    from task_generator.adapters.juelich import ARCHIVE, Juelich

    out = tmp_path / "worlds"
    importer.import_item(f"{SET}/b1", out)
    attribution = read_attribution(out / f"juelich_{SET}_0" / ATTRIBUTION_FILE)
    assert (attribution.license, attribution.url, attribution.authors) == ("CC-BY-4.0", ARCHIVE, "Armin Seyfried, Maik Boltes")
    assert attribution.citation.startswith("Bottleneck, Forschungszentrum Juelich")
    cited = Juelich(cache=seed(tmp_path / "doi", doi="10.34735/ped.2009.1")).attribution(f"{SET}/b1")
    assert (cited.url, cited.citation.endswith("doi:10.34735/ped.2009.1")) == ("https://doi.org/10.34735/ped.2009.1", True)


def test_layout_index_counts_distinct_layouts_in_archive_order() -> None:
    from task_generator.adapters.juelich import layout_index

    metadata = {"experiment": {"run": [{"run_name": "a", "wkt_geometry": ROOM}, {"run_name": "b", "wkt_geometry": CORRIDOR}, {"run_name": "c", "wkt_geometry": ROOM}]}}
    assert [layout_index(metadata, run) for run in ("a", "b", "c")] == [0, 1, 0]


@pytest.mark.network
def test_live_basigo_sets_list_their_runs(tmp_path: Path) -> None:
    from task_generator.adapters.juelich import Juelich, runs

    importer = Juelich(cache=tmp_path / "cache")
    assert len(runs(importer.metadata("basigo_bidirectional"))) == 15
    assert len(runs(importer.metadata("basigo_crossing_90"))) == 46
    assert all(runs(importer.metadata(name)) for name in ("hermes_bottleneck", "hermes_tjunction"))
