"""GLOW importer: 5-minute periods of the time-sorted recording into replay scenarios on one open floor, read by the native loaders."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

START = datetime.datetime(2019, 11, 9, 18, 3, 0)


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_assets.importers")
    pytest.importorskip("arena_simulation_setup.shared.track")


def _walk(ident: int, offset_s: float, duration_s: float, start: tuple[float, float], end: tuple[float, float]) -> list[tuple[datetime.datetime, str]]:
    rows = []
    for k in range(int(duration_s * 30) + 1):
        u = k / (duration_s * 30)
        x, y = start[0] + u * (end[0] - start[0]), start[1] + u * (end[1] - start[1])
        stamp = START + datetime.timedelta(seconds=offset_s + k / 30)
        rows.append((stamp, f"{ident},{x + 0.01:.4f},{y:.4f},{x:.4f},{y:.4f},0.0,0.0"))
    return rows


def write_recording(path: Path, rows: list[tuple[datetime.datetime, str]]) -> Path:
    from task_generator.adapters.glow import HEADER

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([HEADER, *(f"{stamp.isoformat(sep=' ', timespec='microseconds')},{rest}" for stamp, rest in sorted(rows, key=lambda row: row[0]))]) + "\n")
    return path


def seed(cache: Path) -> Path:
    from task_generator.adapters.glow import RECORDING

    write_recording(cache / RECORDING, [*_walk(5, 0.0, 4.0, (0.0, 0.0), (4.0, 0.0)), *_walk(6, 110.0, 20.0, (1.0, 6.0), (1.0, -2.0)), *_walk(7, 300.0, 2.0, (8.0, 1.0), (9.0, 1.0))])
    return cache


@pytest.fixture()
def importer(tmp_path: Path):
    from task_generator.adapters.glow import GLOW

    return GLOW(cache=seed(tmp_path / "cache"))


def _world(out: Path):
    from arena_simulation_setup.tree.World import WorldIdentifier

    from task_generator.adapters.glow import WORLD

    return WorldIdentifier(WORLD).load(out / WORLD)


def test_items_are_the_five_minute_periods_with_rows(importer) -> None:
    assert list(importer.items()) == ["2019-11-09T1800", "2019-11-09T1805"]
    assert importer.periods()["bounds"] == pytest.approx([0.0, -2.0, 9.0, 6.0])
    with pytest.raises(KeyError, match="no GLOW period '2019-11-09T1810'"):
        importer.fetch("2019-11-09T1810")


def test_a_walk_across_a_period_boundary_is_cut_into_both_periods(importer, tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import Track

    out = tmp_path / "worlds"
    first = importer.import_item("2019-11-09T1800", out)
    second = importer.import_item("2019-11-09T1805", out)
    assert sorted(path.name for path in first.glob("*.csv")) == ["ped_5.csv", "ped_6.csv"]
    assert sorted(path.name for path in second.glob("*.csv")) == ["ped_6.csv", "ped_7.csv"]
    walker = Track.load(first / "ped_5.csv")
    assert (walker.start, walker.end) == pytest.approx((180.0, 184.0))
    assert (walker.state(182.0).x, walker.state(182.0).y) == pytest.approx((2.0, 0.0), abs=1e-3)
    assert Track.load(second / "ped_6.csv").start == pytest.approx(0.0)
    assert Track.load(second / "ped_7.csv").start == pytest.approx(180.0)


def test_floor_is_the_box_around_the_whole_recording_with_a_margin(importer, tmp_path: Path) -> None:
    import shapely

    from task_generator.adapters.glow import MARGIN_M

    out = tmp_path / "worlds"
    importer.import_item("2019-11-09T1805", out)
    level = _world(out).load(validate=True).levels["0"]
    floor = shapely.union_all([shapely.Polygon([(point.x, point.y) for point in zone.polygon]) for zone in level.zones])
    assert floor.bounds == pytest.approx((-MARGIN_M, -2.0 - MARGIN_M, 9.0 + MARGIN_M, 6.0 + MARGIN_M))


def test_scenario_loads_as_native_pedestrians_with_their_stays(importer, tmp_path: Path) -> None:
    out = tmp_path / "worlds"
    importer.import_item("2019-11-09T1800", out)
    scenario = _world(out).scenario("2019-11-09T1800").resolve_sync().load()
    assert [human.name for human in scenario.humans] == ["ped_5", "ped_6"]
    assert scenario.humans[0].stays[0].start == pytest.approx(180.0)
    assert scenario.humans[0].stays[0].velocity == pytest.approx((1.0, 0.0), abs=1e-3)


def test_index_refuses_a_recording_out_of_time_order(tmp_path: Path) -> None:
    from task_generator.adapters.glow import HEADER, index

    path = tmp_path / "shuffled.csv"
    path.write_text(f"{HEADER}\n2019-11-09 18:06:00.000000,1,0,0,0,0,0,0\n2019-11-09 18:01:00.000000,2,0,0,0,0,0,0\n")
    with pytest.raises(ValueError, match="not sorted by time"):
        index(path)
    path.write_text("time,id,x,y\n")
    with pytest.raises(ValueError, match="expected 'time,tracked_object"):
        index(path)


def test_attribution(importer) -> None:
    attribution = importer.attribution("2019-11-09T1800")
    assert attribution.license == "CC-BY-4.0"
    assert "Corbetta" in attribution.authors


@pytest.mark.network
def test_live_recording_starts_with_the_expected_header(tmp_path: Path) -> None:
    from task_generator.adapters.glow import GLOW, HEADER, RECORDING

    head = GLOW(cache=tmp_path / "cache").provider.get(RECORDING, tmp_path / "head.csv", range=(0, 200))
    assert head.read_text().splitlines()[0] == HEADER
