from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("sounddevice")

from task_generator.auditory.audio_mixer import AudioMixer, Voice


def _sample(values: list[float]) -> SimpleNamespace:
    return SimpleNamespace(
        samples=np.asarray(values, dtype=np.float32)[:, None],
        path="<test>",
    )


def test_segmented_voice_plays_intro_loop_and_outro() -> None:
    voice = Voice(
        sample=_sample([0.1, 0.2]),
        voice_id="motor:1",
        loop_sample=_sample([0.3, 0.4, 0.5]),
        outro_sample=_sample([0.6, 0.7]),
        stage="intro",
    )
    mixer = AudioMixer.__new__(AudioMixer)
    mixer._voices = [voice]
    mixer._lock = threading.Lock()
    mixer._master_gain = 1.0

    first_block = np.empty((5, 1), dtype=np.float32)
    mixer._callback(first_block, 5, None, None)

    np.testing.assert_allclose(
        first_block[:, 0],
        [0.1, 0.2, 0.3, 0.4, 0.5],
    )
    assert voice.stage == "loop"
    assert voice.position == 0

    assert mixer.stop("motor:1")
    second_block = np.empty((5, 1), dtype=np.float32)
    mixer._callback(second_block, 5, None, None)

    np.testing.assert_allclose(
        second_block[:, 0],
        [0.3, 0.4, 0.5, 0.6, 0.7],
    )
    assert mixer._voices == []


def test_duplicate_sequence_replaces_keyed_voice() -> None:
    mixer = AudioMixer.__new__(AudioMixer)
    mixer._voices = []
    mixer._lock = threading.Lock()

    for _ in range(2):
        mixer.play_looping_sequence(
            _sample([0.1]),
            _sample([0.2]),
            _sample([0.3]),
            voice_id="motor:1",
        )

    assert len(mixer._voices) == 1
    assert mixer._voices[0].voice_id == "motor:1"


def test_keyed_single_loop_repeats_and_stops_at_boundary() -> None:
    mixer = AudioMixer.__new__(AudioMixer)
    mixer._voices = []
    mixer._lock = threading.Lock()
    mixer._master_gain = 1.0

    mixer.play(
        _sample([0.1, 0.2]),
        loop=True,
        voice_id="motor:1",
    )
    first_block = np.empty((5, 1), dtype=np.float32)
    mixer._callback(first_block, 5, None, None)
    np.testing.assert_allclose(
        first_block[:, 0],
        [0.1, 0.2, 0.1, 0.2, 0.1],
    )

    assert mixer.stop("motor:1")
    second_block = np.empty((4, 1), dtype=np.float32)
    mixer._callback(second_block, 4, None, None)
    np.testing.assert_allclose(
        second_block[:, 0],
        [0.2, 0.0, 0.0, 0.0],
    )
    assert mixer._voices == []


def test_render_source_shares_the_sample_mixer() -> None:
    class ConstantSource:
        finished = False

        @staticmethod
        def render(frames: int) -> np.ndarray:
            return np.full((frames, 1), 0.25, dtype=np.float32)

    mixer = AudioMixer.__new__(AudioMixer)
    mixer._voices = []
    mixer._render_voices = []
    mixer._lock = threading.Lock()
    mixer._master_gain = 1.0
    mixer.add_render_source(ConstantSource(), voice_id="procedural:1")

    block = np.empty((4, 1), dtype=np.float32)
    mixer._callback(block, 4, None, None)

    np.testing.assert_allclose(block[:, 0], [0.25] * 4)
    assert mixer.voice_count == 1
