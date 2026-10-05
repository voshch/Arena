"""Audio stream time base pinned to frame stamps: gaps become silence, so sample index i is always start + i / fs."""

from __future__ import annotations

import attrs


@attrs.define
class AudioTimeline:
    fs: int
    max_gap_samples: int
    start_ns: int | None = None
    samples: int = 0
    gaps: int = 0
    rewinds: int = 0

    def observe(self, stamp_ns: int, n: int) -> int:
        """Account for a frame of ``n`` samples starting at ``stamp_ns``. Returns the silence to insert before it."""
        if self.start_ns is None:
            self.start_ns = stamp_ns
        expected_ns = self.start_ns + round(self.samples * 1_000_000_000 / self.fs)
        gap = round((stamp_ns - expected_ns) * self.fs / 1_000_000_000)
        fill = 0
        if gap > 1:
            fill = min(gap, self.max_gap_samples)
            self.gaps += 1
            if gap > self.max_gap_samples:
                self.start_ns += round((gap - fill) * 1_000_000_000 / self.fs)
        elif gap < -1:
            self.rewinds += 1
            self.start_ns = stamp_ns - round(self.samples * 1_000_000_000 / self.fs)
        self.samples += fill + n
        return fill

    def time_ns(self, sample: int) -> int:
        if self.start_ns is None:
            raise RuntimeError("no frame observed yet")
        return self.start_ns + round(sample * 1_000_000_000 / self.fs)
