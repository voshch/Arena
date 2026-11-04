from arena_runtime.env.phase_clock import PhaseClock


class _Clock:
    def __init__(self, *times: float) -> None:
        self._times = list(times)

    def __call__(self) -> float:
        return self._times.pop(0)


def test_laps_measure_time_since_previous_lap():
    clock = PhaseClock(_Clock(10.0, 10.5, 12.0, 12.25))
    clock.lap("hold")
    clock.lap("world")
    clock.lap("respawn")
    assert clock.laps == [("hold", 0.5), ("world", 1.5), ("respawn", 0.25)]
    assert clock.total == 2.25


def test_render_lists_total_then_laps_in_order():
    clock = PhaseClock(_Clock(0.0, 0.125, 1.0))
    clock.lap("set_up")
    clock.lap("robots")
    assert str(clock) == "1.00s (set_up 0.12, robots 0.88)"


def test_render_without_laps_is_zero_total():
    assert str(PhaseClock(_Clock(3.0))) == "0.00s ()"
