from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _gate():
    pytest.importorskip("arena_simulation_setup.shared.track")


def _two_segments():
    from arena_simulation_setup.shared.track import Track

    return Track.from_segments(
        [
            (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, math.pi / 2])),
            (np.array([5.0, 6.0]), np.array([4.0, 3.0]), np.array([2.0, 2.0]), np.array([3.0, -3.0])),
        ]
    )


def test_state_interpolates_inside_a_segment() -> None:
    state = _two_segments().state(0.25)
    assert (state.x, state.y, state.yaw) == pytest.approx((0.25, 0.0, 0.0))
    assert (state.vx, state.vy) == pytest.approx((1.0, 0.0))
    turning = _two_segments().state(1.5)
    assert (turning.x, turning.y, turning.yaw) == pytest.approx((1.0, 1.0, math.pi / 4))
    assert (turning.vx, turning.vy) == pytest.approx((0.0, 2.0))


def test_state_is_absent_outside_the_segments() -> None:
    track = _two_segments()
    assert track.state(-0.01) is None
    assert track.state(2.01) is None
    assert track.state(4.99) is None
    assert track.state(6.01) is None
    assert track.spans == [(0.0, 2.0), (5.0, 6.0)]
    assert (track.start, track.end) == (0.0, 6.0)


def test_state_on_the_last_sample_keeps_the_arriving_velocity() -> None:
    state = _two_segments().state(2.0)
    assert (state.x, state.y, state.yaw) == pytest.approx((1.0, 2.0, math.pi / 2))
    assert (state.vx, state.vy) == pytest.approx((0.0, 2.0))
    first = _two_segments().state(5.0)
    assert (first.x, first.y) == pytest.approx((4.0, 2.0))
    assert (first.vx, first.vy) == pytest.approx((-1.0, 0.0))


def test_heading_turns_the_short_way_across_pi() -> None:
    state = _two_segments().state(5.5)
    assert abs(state.yaw) == pytest.approx(math.pi, abs=1e-9)


def test_file_round_trip_keeps_rows_and_the_pen_lift(tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import HEADER, Track

    track = _two_segments()
    path = tmp_path / "scenario" / "ped.csv"
    track.save(path)
    lines = path.read_text().splitlines()
    assert lines[0] == HEADER
    assert lines[4] == "nan,nan,nan,nan"
    assert len(lines) == 7
    again = Track.load(path)
    for column in ("t", "x", "y", "yaw"):
        assert getattr(again, column) == pytest.approx(getattr(track, column), abs=1e-4)
    assert again.segment.tolist() == [0, 0, 0, 1, 1]
    assert again.dumps() == path.read_bytes()


def test_shift_moves_positions_only() -> None:
    track = _two_segments()
    moved = track.shifted(10.0, -2.0)
    assert moved.x == pytest.approx(track.x + 10.0)
    assert moved.y == pytest.approx(track.y - 2.0)
    assert np.array_equal(moved.t, track.t)
    assert np.array_equal(moved.yaw, track.yaw)


def test_times_must_increase(tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import HEADER, Track

    with pytest.raises(ValueError, match="increase strictly"):
        Track.from_segments([(np.array([0.0, 0.0]), np.zeros(2), np.zeros(2), np.zeros(2))])
    path = tmp_path / "back.csv"
    path.write_text(f"{HEADER}\n1.0,0,0,0\nnan,nan,nan,nan\n0.5,0,0,0\n")
    with pytest.raises(ValueError, match="increase strictly"):
        Track.load(path)


def test_header_is_checked(tmp_path: Path) -> None:
    from arena_simulation_setup.shared.track import Track

    path = tmp_path / "other.csv"
    path.write_text("frame,x,y\n0,0,0\n")
    with pytest.raises(ValueError, match="expected 't,x,y,yaw'"):
        Track.load(path)


def test_track_file_resolves_against_the_scenario_directory(tmp_path: Path) -> None:
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.shared.track import track_file
    from arena_simulation_setup.utils.cattrs import converter

    entry = {"name": "ped", "model": "arenian", "pose": [0.0, 0.0, 0.0]}
    relative = converter.structure({**entry, "track": "./ped.csv"}, DynamicObstacle)
    relative.included_from = tmp_path
    assert track_file(relative) == tmp_path / "ped.csv"
    absolute = converter.structure({**entry, "track": str(tmp_path / "abs.csv")}, DynamicObstacle)
    absolute.included_from = tmp_path / "elsewhere"
    assert track_file(absolute) == tmp_path / "abs.csv"
    assert track_file(converter.structure(dict(entry), DynamicObstacle)) is None


def test_hold_start_rests_on_the_first_sample_before_the_track_begins() -> None:
    from arena_simulation_setup.shared.track import Track, TrackState

    track = Track.from_segments([(np.array([2.0, 3.0]), np.array([1.0, 2.0]), np.array([4.0, 4.0]), np.array([0.5, 0.5])), (np.array([6.0, 7.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.0]))])
    assert track.state(1.0) is None
    assert track.state(1.0, hold_start=True) == TrackState(1.0, 4.0, 0.5, 0.0, 0.0)
    assert track.state(2.5, hold_start=True) == track.state(2.5)
    assert track.state(4.0, hold_start=True) is None
    assert track.state(8.0, hold_start=True) is None


def test_prespawn_is_a_bool_field_of_the_human() -> None:
    from arena_simulation_setup.shared import DynamicObstacle
    from arena_simulation_setup.utils.cattrs import converter

    entry = {"name": "ped", "model": "arenian", "pose": [0.0, 0.0, 0.0], "track": "./ped.csv"}
    assert converter.structure(dict(entry), DynamicObstacle).prespawn is False
    assert converter.structure({**entry, "prespawn": True}, DynamicObstacle).prespawn is True
