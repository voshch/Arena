from __future__ import annotations

import pytest


def _t(sec: int, nanosec: int):
    from arena_rclpy_mixins.Time import Time

    return Time(sec=sec, nanosec=nanosec)


def test_subtraction_may_go_negative():
    assert (_t(4, 0) - _t(5, 0)).to_nanoseconds() == -1_000_000_000


def test_subtraction_is_exact_at_wall_clock_magnitude():
    """One nanosecond at ~1.7e9 seconds sits below float eps."""
    assert (_t(1_700_000_000, 1) - _t(1_700_000_000, 0)).to_nanoseconds() == 1


def test_addition_carries_across_a_second_boundary():
    assert _t(4, 999_999_999) + _t(0, 1) == _t(5, 0)


def test_negative_results_stay_serializable():
    """nanosec is uint32 on the wire, so it has to normalize into [0, 1e9)."""
    delta = _t(0, 0) - _t(0, 500_000_000)
    assert delta == _t(-1, 500_000_000)
    assert 0 <= delta.nanosec < 1_000_000_000
    assert delta.to_msg().nanosec == 500_000_000


def test_negative_results_order_and_convert_correctly():
    delta = _t(0, 0) - _t(0, 500_000_000)
    assert delta < _t(0, 0)
    assert delta.to_seconds() == -0.5


def test_from_float_floors_negatives():
    from arena_rclpy_mixins.Time import Time

    assert Time.from_float(-0.5) == _t(-1, 500_000_000)
    assert Time.from_float(1.5) == _t(1, 500_000_000)


@pytest.mark.parametrize(
    ("later", "earlier"),
    [
        pytest.param((5, 0), (4, 999_999_999), id="greater_sec_wins_over_smaller_nanosec"),
        pytest.param((5, 500), (5, 200), id="within_the_same_second"),
    ],
)
def test_ordering_is_lexicographic_on_sec_and_nanosec(later, earlier):
    assert _t(*later) > _t(*earlier)
    assert _t(*earlier) < _t(*later)
    assert not _t(*later) < _t(*earlier)
    assert not _t(*earlier) > _t(*later)


def test_equal_times_are_neither_less_nor_greater():
    assert not _t(5, 500) < _t(5, 500)
    assert not _t(5, 500) > _t(5, 500)
    assert _t(5, 500) >= _t(5, 500)
    assert _t(5, 500) <= _t(5, 500)


def test_sorts_monotonically_across_second_boundaries():
    stamps = [_t(5, 0), _t(4, 999_999_999), _t(5, 1), _t(4, 0)]
    assert sorted(stamps) == [_t(4, 0), _t(4, 999_999_999), _t(5, 0), _t(5, 1)]
