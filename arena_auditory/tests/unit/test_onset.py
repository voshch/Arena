import numpy as np

from arena_auditory.hearing.onset import OnsetDetector

FS = 16000
HOP_S = 0.1
HOP_N = int(HOP_S * FS)


def _frame(amplitude: float, n: int = HOP_N, channels: int = 1) -> np.ndarray:
    return np.full((n, channels), amplitude, dtype=np.float64)


def test_silence_never_fires():
    det = OnsetDetector(FS, hop_s=HOP_S)
    for _ in range(30):
        fired, peak_db, floor_db = det.step(_frame(0.0))
        assert fired is False


def test_burst_fires_once_floor_established():
    det = OnsetDetector(FS, hop_s=HOP_S, onset_db=6.0)
    for _ in range(12):
        fired, _, _ = det.step(_frame(1.0))
        assert fired is False
    fired, peak_db, floor_db = det.step(_frame(10.0))
    assert fired is True
    assert peak_db - floor_db >= 6.0


def test_burst_before_floor_established_does_not_fire():
    det = OnsetDetector(FS, hop_s=HOP_S, onset_db=6.0)
    for _ in range(5):
        fired, _, _ = det.step(_frame(1.0))
        assert fired is False
    fired, _, _ = det.step(_frame(100.0))
    assert fired is False


def test_floor_tracks_level_change():
    det = OnsetDetector(FS, hop_s=HOP_S, floor_window_s=5.0, onset_db=6.0)
    for _ in range(60):
        _, low_peak_db, floor_db = det.step(_frame(1.0))
    assert abs(floor_db - low_peak_db) < 0.01
    for _ in range(60):
        _, high_peak_db, floor_db = det.step(_frame(3.0))
    assert abs(floor_db - high_peak_db) < 0.01
    assert high_peak_db - low_peak_db > 5.0
