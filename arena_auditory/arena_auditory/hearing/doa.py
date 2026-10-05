"""Bearing from the array geometry: onset-windowed GCC-PHAT delays over every mic pair, least-squares over azimuth, gated against ego noise."""

from __future__ import annotations

import itertools

import numpy as np

from arena_auditory.spatial_audio import gcc_phat, rectangular_array

_ONSET_HOP_S = 0.005


class ArrayBearing:
    """Far-field planar fit. ``mics`` in channel order, positions in the base frame."""

    def __init__(
        self,
        sample_rate_hz: int,
        *,
        positions_m: tuple[tuple[float, float, float], ...] | None = None,
        speed_of_sound_mps: float = 343.0,
        step_deg: float = 1.0,
        onset_window_s: float = 0.04,
        onset_lead_s: float = 0.01,
        min_tdoa_s: float = 1.5e-4,
        max_residual_s: float = 1.0e-4,
    ) -> None:
        pos = np.asarray(positions_m if positions_m is not None else tuple(m.position_m for m in rectangular_array()), dtype=np.float64)[:, :2]
        self.fs = int(sample_rate_hz)
        self.pairs = list(itertools.combinations(range(len(pos)), 2))
        self.candidates = np.radians(np.arange(0.0, 360.0, step_deg))
        units = np.stack([np.cos(self.candidates), np.sin(self.candidates)], axis=1)
        baselines = np.array([pos[a] - pos[b] for a, b in self.pairs])
        # signal-minus-reference delay: mic a hears later than b when it is farther along -u
        self.predicted = -(units @ baselines.T) / speed_of_sound_mps
        self.max_tau = float(np.linalg.norm(baselines, axis=1).max() / speed_of_sound_mps) * 1.05
        self.onset_window_s = float(onset_window_s)
        self.onset_lead_s = float(onset_lead_s)
        self.min_tdoa_s = float(min_tdoa_s)
        self.max_residual_s = float(max_residual_s)

    def _onset_window(self, frame: np.ndarray) -> np.ndarray:
        """Slice of ``frame`` around the loudest 5 ms hop of the mono sum."""
        n = frame.shape[0]
        hop = max(int(round(_ONSET_HOP_S * self.fs)), 1)
        mono = frame.sum(axis=1)
        n_hops = max(n // hop, 1)
        energy = np.array([np.sum(mono[i * hop : (i + 1) * hop] ** 2) for i in range(n_hops)])
        peak_s = int(np.argmax(energy)) * hop / self.fs
        start_s = max(peak_s - self.onset_lead_s, 0.0)
        end_s = min(start_s + self.onset_window_s, n / self.fs)
        return frame[int(round(start_s * self.fs)) : int(round(end_s * self.fs))]

    def bearing(self, frame: np.ndarray) -> tuple[float, float, bool]:
        """(azimuth rad CCW from +x, rms delay residual s, valid) for the onset window of a (samples, channels) frame."""
        window = self._onset_window(frame)
        measured = np.array([gcc_phat(window[:, a], window[:, b], sample_rate_hz=self.fs, max_tau_seconds=self.max_tau)[0] for a, b in self.pairs])
        err = ((self.predicted - measured[None, :]) ** 2).mean(axis=1)
        best = int(np.argmin(err))
        residual = float(np.sqrt(err[best]))
        valid = bool(np.max(np.abs(measured)) >= self.min_tdoa_s) and residual <= self.max_residual_s
        return float(self.candidates[best]), residual, valid
