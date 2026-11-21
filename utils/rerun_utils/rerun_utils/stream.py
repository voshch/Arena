"""How the bridge logs to rerun: streams on the `sim_time` timeline at a capped rate per key, one-time data static."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import rerun as rr
from builtin_interfaces.msg import Time

TIMELINE = "sim_time"
UNCAPPED_FLUSH_TICK_S = 1 / 30


def recording(application_id: str, max_rate_hz: float) -> rr.RecordingStream:
    """The bridge's recording, made the global one, its batches flushed once per rate-cap period."""
    flush_tick = 1.0 / max_rate_hz if max_rate_hz > 0 else UNCAPPED_FLUSH_TICK_S
    stream = rr.RecordingStream(application_id, make_default=True, batcher_config=rr.ChunkBatcherConfig(flush_tick=flush_tick))
    rr.set_global_data_recording(stream)
    return stream


class Stream:
    """Runs each `offer` key at most `max_rate_hz` (0 = uncapped), the newest early thunk on `flush`."""

    def __init__(
        self,
        recording: rr.RecordingStream,
        max_rate_hz: float,
        sim_now: Callable[[], float],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._recording = recording
        self.period = 1.0 / max_rate_hz if max_rate_hz > 0 else 0.0
        self._sim_now = sim_now
        self._clock = clock
        self._last: dict[object, float] = {}
        self._waiting: dict[object, Callable[[], None]] = {}
        self._lock = threading.Lock()

    def offer(self, key: object, thunk: Callable[[], None]) -> None:
        with self._lock:
            now = self._clock()
            self._waiting.pop(key, None)
            if now - self._last.get(key, -float("inf")) >= self.period:
                self._last[key] = now
                thunk()
            else:
                self._waiting[key] = thunk

    def flush(self) -> None:
        """Run the waiting thunks whose slot has come."""
        with self._lock:
            now = self._clock()
            for key in [k for k in self._waiting if now - self._last.get(k, -float("inf")) >= self.period]:
                self._last[key] = now
                self._waiting.pop(key)()

    def subscriber[M](self, key: object, cb: Callable[[M], None]) -> Callable[[M], None]:
        """A subscription callback that converts and logs only the messages `offer` lets through."""
        return lambda msg: self.offer(key, lambda: cb(msg))

    def log(self, entity: str, *components: rr.AsComponents, stamp: Time | None = None) -> None:
        seconds = stamp.sec + stamp.nanosec * 1e-9 if stamp is not None else 0.0
        rr.set_time(TIMELINE, duration=seconds or self._sim_now(), recording=self._recording)
        rr.log(entity, *components, recording=self._recording)

    def fixed(self, entity: str, *components: rr.AsComponents) -> None:
        """Log static, which the gRPC server keeps for late viewers after it drops older temporal data."""
        rr.log(entity, *components, static=True, recording=self._recording)
