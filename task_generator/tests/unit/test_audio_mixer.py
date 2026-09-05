from __future__ import annotations
from task_generator.auditory.audio_mixer import AudioMixer, Voice, auto_output_candidates

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")


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
    mixer = AudioMixer(channels=1)
    mixer._voices = [voice]

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
    mixer = AudioMixer(channels=1)

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
    mixer = AudioMixer(channels=1)

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

    mixer = AudioMixer(channels=1)
    mixer.add_render_source(ConstantSource(), voice_id="procedural:1")

    block = np.empty((4, 1), dtype=np.float32)
    mixer._callback(block, 4, None, None)

    np.testing.assert_allclose(block[:, 0], [0.25] * 4)
    assert mixer.voice_count == 1


def test_muted_sample_bus_stays_synchronized() -> None:
    mixer = AudioMixer(channels=1)

    mixer.play(
        _sample([0.1, 0.2]),
        loop=True,
        voice_id="motor:1",
        bus="motor",
    )
    mixer.set_bus_enabled("motor", False)

    muted_block = np.empty((3, 1), dtype=np.float32)
    mixer._callback(muted_block, 3, None, None)
    np.testing.assert_allclose(muted_block[:, 0], [0.0] * 3)

    mixer.set_bus_enabled("motor", True)
    audible_block = np.empty((2, 1), dtype=np.float32)
    mixer._callback(audible_block, 2, None, None)
    np.testing.assert_allclose(audible_block[:, 0], [0.2, 0.1])


def test_muted_render_bus_stays_synchronized() -> None:
    class CountingSource:
        finished = False

        def __init__(self) -> None:
            self.render_count = 0

        def render(self, frames: int) -> np.ndarray:
            self.render_count += 1
            return np.full((frames, 1), 0.25, dtype=np.float32)

    mixer = AudioMixer(channels=1)
    source = CountingSource()
    mixer.add_render_source(source, voice_id="robot:jackal:motor", bus="motor")
    mixer.set_bus_enabled("motor", False)

    muted_block = np.empty((4, 1), dtype=np.float32)
    mixer._callback(muted_block, 4, None, None)
    np.testing.assert_allclose(muted_block[:, 0], [0.0] * 4)
    assert source.render_count == 1

    mixer.set_bus_enabled("motor", True)
    audible_block = np.empty((4, 1), dtype=np.float32)
    mixer._callback(audible_block, 4, None, None)
    np.testing.assert_allclose(audible_block[:, 0], [0.25] * 4)
    assert source.render_count == 2


def _device(name: str, outputs: int = 2) -> dict[str, object]:
    return {"name": name, "max_output_channels": outputs}


@pytest.mark.parametrize(
    ("devices", "default_index", "expected"),
    [
        ([_device("hw:0,0"), _device("pulse"), _device("default")], 0, [1, 2, 0]),
        ([_device("hw:0,0"), _device("pipewire")], 0, [1, 0]),
        ([_device("default")], 0, [0]),
        ([_device("hw:0,0"), _device("hw:1,0")], 1, [1]),
        ([_device("pulse")], 0, [0]),
        ([], -1, []),
        ([_device("hw:0,0")], -1, []),
        ([_device("pulse", outputs=0), _device("default")], 0, [1]),
        ([_device("mic", outputs=0)], 0, []),
    ],
)
def test_auto_output_candidates(devices: list[dict[str, object]], default_index: int, expected: list[int]) -> None:
    assert auto_output_candidates(devices, default_index) == expected
