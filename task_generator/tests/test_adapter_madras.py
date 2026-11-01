"""MADRAS importer: TopView sequences into replay scenarios on one open floor around every tracked point, read by the native loaders."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

HEADER = ["# trajectory data of sequence TopView_{name}", "# recorded at 30 fps", "# id frame x/m y/m z/m id_global t/s x_RGF y_RGF"]
WALKERS = {"1A": [(1, 0, 60, (-2.0, 5.0), (2.0, 5.0)), (2, 30, 90, (0.0, 0.0), (0.0, 3.0))], "2B": [(3, 0, 30, (10.0, -4.0), (12.0, -4.0))]}


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")
    pytest.importorskip("arena_simulation_setup.shared.track")


def _sequence(name: str) -> str:
    rows = [line.format(name=name) for line in HEADER]
    for ident, first, last, (x0, y0), (x1, y1) in WALKERS[name]:
        for frame in range(first, last + 1):
            u = (frame - first) / (last - first)
            rows.append(f"{ident} {frame} {x0 + u * (x1 - x0):.3f} {y0 + u * (y1 - y0):.3f} 1.8 {110000 + ident} {frame / 30:.3f} 842499.35 6520341.33")
    return "\n".join(rows) + "\n"


def seed(cache: Path) -> Path:
    from task_generator.adapters.madras import ARCHIVE

    cache.mkdir(parents=True)
    with zipfile.ZipFile(cache / ARCHIVE, "w") as z:
        z.writestr("Data_Madras/Geometry.csv", "x,y\n")
        for name in WALKERS:
            z.writestr(f"Data_Madras/TopView_trajectories/TopView_{name}.txt", _sequence(name))
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from task_generator.adapters.madras import MADRAS

    return MADRAS(cache=seed(tmp_path / "cache"))


def _world(out: Path):
    from arena_simulation_setup.tree.World import WorldIdentifier

    from task_generator.adapters.madras import WORLD

    return WorldIdentifier(WORLD).load(out / WORLD)


def test_items_are_the_topview_sequences(importer) -> None:
    assert list(importer.items()) == ["topview_1a", "topview_2b"]
    with pytest.raises(KeyError, match="items: topview_1a, topview_2b"):
        importer.fetch("topview_9z")


def test_floor_is_the_box_around_every_sequence_with_a_margin(importer, tmp_path: Path) -> None:
    import shapely

    from task_generator.adapters.madras import MARGIN_M

    out = tmp_path / "worlds"
    importer.import_item("topview_1a", out)
    level = _world(out).load(validate=True).levels["0"]
    floor = shapely.union_all([shapely.Polygon([(point.x, point.y) for point in zone.polygon]) for zone in level.zones])
    assert floor.bounds == pytest.approx((-2.0 - MARGIN_M, -4.0 - MARGIN_M, 12.0 + MARGIN_M, 5.0 + MARGIN_M))
    assert not level.objects


def test_both_sequences_share_one_world_in_either_order(tmp_path: Path) -> None:
    from task_generator.adapters.madras import MADRAS, WORLD

    worlds = []
    for order in (("topview_1a", "topview_2b"), ("topview_2b", "topview_1a")):
        out = tmp_path / "-".join(order) / "worlds"
        importer = MADRAS(cache=seed(tmp_path / "-".join(order) / "cache"))
        for item in order:
            importer.import_item(item, out)
        assert sorted(scenario.name for scenario in _world(out).scenario.listall()) == ["topview_1a", "topview_2b"]
        worlds.append((out / WORLD / "world.yaml").read_bytes())
    assert worlds[0] == worlds[1]


def test_tracks_are_timed_by_frame_at_30_fps(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import Track

    scenario = importer.import_item("topview_1a", tmp_path / "worlds")
    walker = Track.load(scenario / "ped_1.csv")
    assert (walker.start, walker.end) == pytest.approx((0.0, 2.0))
    assert (walker.state(1.0).x, walker.state(1.0).y) == pytest.approx((0.0, 5.0), abs=1e-3)
    assert Track.load(scenario / "ped_2.csv").start == pytest.approx(1.0)


def test_scenario_loads_as_native_pedestrians_with_their_stays(importer, tmp_path: Path) -> None:
    out = tmp_path / "worlds"
    importer.import_item("topview_2b", out)
    scenario = _world(out).scenario("topview_2b").resolve_sync().load()
    (human,) = scenario.humans
    assert human.name == "ped_3"
    (stay,) = human.stays
    assert stay.velocity == pytest.approx((2.0, 0.0), abs=1e-3)


def test_attribution(importer, tmp_path: Path) -> None:
    from arena_assets.importers import ATTRIBUTION_FILE, read_attribution

    from task_generator.adapters.madras import RECORD, WORLD

    out = tmp_path / "worlds"
    importer.import_item("topview_1a", out)
    attribution = read_attribution(out / WORLD / ATTRIBUTION_FILE)
    assert (attribution.license, attribution.url) == ("CC-BY-4.0", f"https://zenodo.org/records/{RECORD}")
    assert "Tordeux" in attribution.authors


@pytest.mark.network
def test_live_archive_holds_nine_topview_sequences(tmp_path: Path) -> None:
    from task_generator.adapters.madras import MADRAS

    importer = MADRAS(cache=tmp_path / "cache")
    assert importer.fetch_archive().stat().st_size == 33_639_872
    assert list(importer.items()) == [f"topview_{name}" for name in ("1a", "1b", "1c", "2a", "2b", "2c", "2d", "2e", "2f")]
