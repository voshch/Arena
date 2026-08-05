from __future__ import annotations


def _t(sec: int, nanosec: int):
    from arena_rclpy_mixins.Time import Time

    return Time(sec=sec, nanosec=nanosec)


def test_greater_sec_wins_over_smaller_nanosec():
    """Ordering is lexicographic on (sec, nanosec)."""
    assert _t(5, 0) > _t(4, 999_999_999)
    assert not _t(5, 0) < _t(4, 999_999_999)


def test_orders_within_the_same_second():
    assert _t(5, 200) < _t(5, 500)
    assert _t(5, 500) > _t(5, 200)


def test_equal_times_are_neither_less_nor_greater():
    assert not _t(5, 500) < _t(5, 500)
    assert not _t(5, 500) > _t(5, 500)
    assert _t(5, 500) >= _t(5, 500)
    assert _t(5, 500) <= _t(5, 500)


def test_sorts_monotonically_across_second_boundaries():
    stamps = [_t(5, 0), _t(4, 999_999_999), _t(5, 1), _t(4, 0)]
    assert sorted(stamps) == [_t(4, 0), _t(4, 999_999_999), _t(5, 0), _t(5, 1)]
