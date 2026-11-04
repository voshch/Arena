import time
from collections.abc import Callable


class PhaseClock:
    """Wall-clock laps of one multi-step operation, rendered as one log line."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start = self._last = clock()
        self.laps: list[tuple[str, float]] = []

    def lap(self, name: str) -> None:
        now = self._clock()
        self.laps.append((name, now - self._last))
        self._last = now

    @property
    def total(self) -> float:
        return self._last - self._start

    def __str__(self) -> str:
        return f"{self.total:.2f}s (" + ", ".join(f"{name} {seconds:.2f}" for name, seconds in self.laps) + ")"
