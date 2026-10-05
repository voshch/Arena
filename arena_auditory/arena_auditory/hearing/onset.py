"""Energy-onset detection over a running noise floor: peak-vs-median trigger, no ROS."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

_ONSET_HOP_S = 0.005
_MIN_FLOOR_SAMPLES = 10


class OnsetDetector:
    """Hop-by-hop peak-vs-floor trigger on the mono sum of a multichannel stream."""

    def __init__(
        self,
        sample_rate_hz: int,
        *,
        hop_s: float = 0.1,
        floor_window_s: float = 5.0,
        onset_db: float = 6.0,
    ) -> None:
        self.fs = int(sample_rate_hz)
        self.hop_s = float(hop_s)
        self.onset_db = float(onset_db)
        maxlen = max(int(round(floor_window_s / hop_s)), 1)
        self._floor_history: deque[float] = deque(maxlen=maxlen)

    def step(self, frame: np.ndarray) -> tuple[bool, float, float]:
        """(fired, peak_db, floor_db) for one hop of (samples, channels) audio."""
        peak_db = self._peak_db(frame)
        floor_db = float(np.median(self._floor_history)) if self._floor_history else peak_db
        fired = len(self._floor_history) >= _MIN_FLOOR_SAMPLES and (peak_db - floor_db) >= self.onset_db
        self._floor_history.append(peak_db)
        return fired, peak_db, floor_db

    def _peak_db(self, frame: np.ndarray) -> float:
        """dB of the loudest 5 ms hop of the mono sum."""
        mono = frame.sum(axis=1)
        n = mono.shape[0]
        hop = max(int(round(_ONSET_HOP_S * self.fs)), 1)
        n_hops = max(n // hop, 1)
        energy = np.array([np.sum(mono[i * hop : (i + 1) * hop] ** 2) for i in range(n_hops)])
        return 10.0 * math.log10(max(float(np.max(energy)), 1e-12))
