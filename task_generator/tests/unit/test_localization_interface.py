from __future__ import annotations

import numpy as np
import pytest

from task_generator.auditory.localization import (
    AudioLocalizationInput,
    LocalizationObservation,
    SourceHypothesis,
)


def test_audio_localization_input_tracks_geometry_and_shape() -> None:
    audio = np.zeros((4, 256), dtype=np.float32)
    data = AudioLocalizationInput(
        timestamp_ns=123456789,
        sample_rate_hz=16000,
        audio=audio,
        microphone_positions_m=(
            (0.0, 0.0, 0.2),
            (0.1, 0.0, 0.2),
            (0.0, 0.1, 0.2),
            (0.1, 0.1, 0.2),
        ),
        microphone_frame_ids=("mic_0", "mic_1", "mic_2", "mic_3"),
        robot_frame_id="base_link",
        world_frame_id="map",
        robot_position_world_m=(1.0, 2.0, 0.0),
        robot_yaw_world_rad=0.5,
    )

    assert data.channel_count == 4
    assert data.frame_count == 256
    assert data.microphone_frame_ids[0] == "mic_0"
    assert data.world_frame_id == "map"


def test_audio_localization_input_rejects_geometry_mismatch() -> None:
    with pytest.raises(ValueError, match="must match audio channels"):
        AudioLocalizationInput(
            timestamp_ns=1,
            sample_rate_hz=16000,
            audio=np.zeros((2, 128), dtype=np.float32),
            microphone_positions_m=((0.0, 0.0, 0.0),),
        )


def test_localization_observation_can_carry_multihypothesis_output() -> None:
    observation = LocalizationObservation(
        timestamp_ns=123456789,
        source_present=True,
        azimuth_rad=1.2,
        elevation_rad=None,
        confidence=0.87,
        generic_sound_probability=0.95,
        event_class="unknown",
        robot_frame_id="base_link",
        world_frame_id="map",
        source_hypotheses=(
            SourceHypothesis(
                frame_id="map",
                position_m=(2.0, 3.0, 0.0),
                weight=0.7,
                nlos_probability=0.2,
                path_type="direct",
            ),
            SourceHypothesis(
                frame_id="map",
                position_m=(5.0, 3.0, 0.0),
                weight=0.3,
                nlos_probability=0.8,
                path_type="reflection",
            ),
        ),
    )

    assert observation.source_present is True
    assert observation.source_hypotheses[0].path_type == "direct"
    assert observation.source_hypotheses[1].nlos_probability == 0.8