from __future__ import annotations

import os

import numpy as np
import pytest
from arena_auditory import offline_render, render_core
from scipy.io import wavfile

BLOCK_SIZE = 64
SAMPLE_RATE = 16000

TUNING = render_core.DrivetrainTuning(
    volume_db=-20.0,
    frequency_scale=1.0,
    tonal_gain_db=0.0,
    broadband_gain_db=-12.0,
    speed_exponent=1.5,
    velocity_smoothing_seconds=0.015,
)


def asset() -> np.ndarray:
    rng = np.random.default_rng(20240917)
    return np.ascontiguousarray((rng.standard_normal(SAMPLE_RATE) * 0.1).astype(np.float32))


def params() -> render_core.RenderParams:
    samples = asset()
    return render_core.RenderParams(
        block_size=BLOCK_SIZE,
        sample_rate=SAMPLE_RATE,
        resolve=lambda _key: samples,
    )


def block(index: int, *, gain: float = 0.5, velocity: float = 0.3) -> render_core.RenderInputs:
    return render_core.RenderInputs(
        block_index=index,
        clips=(),
        continuous=(
            render_core.ContinuousInput(
                source_id="fan",
                channel=1,
                asset_key="fan_a",
                gain=gain,
                delay_target=2.0 + 0.25 * index,
                loop=True,
                program_start=0,
            ),
        ),
        drivetrain=(
            render_core.DrivetrainInput(
                source_id="robot",
                deterministic_seed=11,
                gains=(0.5, 0.4, 0.3, 0.2),
                delay_samples=(1.5, 2.5, 3.5, 4.5),
                active_channels=(True, True, True, True),
                left_velocity=velocity,
                right_velocity=velocity + 0.1,
                active=True,
                tuning=TUNING,
                source_agent_id=1,
                source_agent_name="robot",
                sound_type="motor",
                asset_id="drivetrain",
            ),
        ),
    )


def trace(indices: list[int], **kwargs: float) -> list[render_core.RenderInputs]:
    return [block(index, **kwargs) for index in indices]


def test_thread_pin_is_set_by_import() -> None:
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_replay_matches_a_direct_render_block_loop() -> None:
    blocks = trace(list(range(24)))

    expected_params = params()
    expected_state = render_core.RenderState()
    expected = [render_core.render_block(expected_state, item, expected_params) for item in blocks]

    replayed = list(offline_render.replay(iter(blocks), params()))

    assert [item.block_index for item, _ in replayed] == [item.block_index for item in blocks]
    for (_, result), reference in zip(replayed, expected, strict=True):
        assert np.array_equal(result.raw, reference.raw)
        assert np.array_equal(result.motor, reference.motor)
        assert result.clipped_samples == reference.clipped_samples


def test_replay_carries_state_between_blocks() -> None:
    blocks = trace(list(range(8)))
    stateless = [render_core.render_block(render_core.RenderState(), item, params()) for item in blocks]
    replayed = [result for _, result in offline_render.replay(iter(blocks), params())]

    assert np.array_equal(replayed[0].raw, stateless[0].raw)
    assert not np.array_equal(replayed[-1].raw, stateless[-1].raw)


def test_replay_reanchors_the_cursor_from_the_block_index() -> None:
    blocks = trace([0, 1, 7, 8])
    state = render_core.RenderState()
    replayed = list(offline_render.replay(iter(blocks), params(), state=state))

    assert state.cursor == 9 * BLOCK_SIZE
    assert len(replayed) == 4

    expected_state = render_core.RenderState()
    expected = []
    for item in blocks:
        expected_state.cursor = item.block_index * BLOCK_SIZE
        expected.append(render_core.render_block(expected_state, item, params()))
    for (_, result), reference in zip(replayed, expected, strict=True):
        assert np.array_equal(result.raw, reference.raw)


def test_render_results_concatenates_blocks_in_order() -> None:
    blocks = trace(list(range(5)))
    results = list(offline_render.replay(iter(blocks), params()))
    audio = offline_render.render_results(results)

    assert audio.shape == (render_core.CHANNELS, 5 * BLOCK_SIZE)
    assert np.array_equal(audio[:, :BLOCK_SIZE], results[0][1].raw)
    assert np.array_equal(audio[:, -BLOCK_SIZE:], results[-1][1].raw)


def test_render_results_on_an_empty_trace() -> None:
    audio = offline_render.render_results([])
    assert audio.shape == (render_core.CHANNELS, 0)


def test_verify_results_accepts_an_exact_replay() -> None:
    results = list(offline_render.replay(iter(trace(list(range(6)))), params()))
    raw = [result.raw.copy() for _, result in results]
    motor = [result.motor.copy() for _, result in results]

    assert offline_render.verify_results(iter(results), raw, motor) is None


def test_verify_results_reports_the_first_divergence() -> None:
    results = list(offline_render.replay(iter(trace([4, 5, 6, 7])), params()))
    raw = [result.raw.copy() for _, result in results]
    motor = [result.motor.copy() for _, result in results]
    raw[2][3, 17] += np.float32(0.5)

    divergence = offline_render.verify_results(iter(results), raw, motor)

    assert divergence is not None
    assert divergence.stream == "raw_array"
    assert divergence.block == 6
    assert divergence.channel == 3
    assert divergence.sample == 17
    assert divergence.recorded == pytest.approx(divergence.replayed + 0.5, abs=1e-6)


def test_verify_results_checks_the_motor_stem_too() -> None:
    results = list(offline_render.replay(iter(trace([0, 1])), params()))
    raw = [result.raw.copy() for _, result in results]
    motor = [result.motor.copy() for _, result in results]
    motor[1][0, 5] = np.float32(1.0)

    divergence = offline_render.verify_results(iter(results), raw, motor)

    assert divergence is not None
    assert divergence.stream == "stem_motor"
    assert divergence.block == 1


def test_verify_results_tolerates_a_missing_stream() -> None:
    results = list(offline_render.replay(iter(trace([0, 1])), params()))
    raw = [result.raw.copy() for _, result in results]

    assert offline_render.verify_results(iter(results), raw, []) is None


def test_first_divergence_rejects_a_shape_mismatch() -> None:
    replayed = np.zeros((4, 8), dtype=np.float32)
    recorded = np.zeros((4, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="shape|replayed"):
        offline_render.first_divergence("raw_array", 3, replayed, recorded)


def test_remix_block_leaves_a_motorless_block_untouched() -> None:
    raw = np.linspace(-0.5, 0.5, 4 * 16, dtype=np.float32).reshape(4, 16)
    motor = np.zeros_like(raw)

    remixed = offline_render.remix_block(raw, motor, 12.0)

    assert remixed.dtype == np.float32
    assert np.array_equal(remixed, raw)


def test_remix_block_attenuates_pure_ego_noise() -> None:
    raw = np.full((4, 16), 0.4, dtype=np.float32)

    assert offline_render.remix_block(raw, raw, 20.0) == pytest.approx(0.04, abs=1e-6)
    assert offline_render.remix_block(raw, raw, 0.0) == pytest.approx(0.4, abs=1e-6)


def test_remix_block_keeps_the_pedestrian_stem() -> None:
    pedestrians = np.full((4, 16), 0.2, dtype=np.float32)
    motor = np.full((4, 16), 0.1, dtype=np.float32)

    remixed = offline_render.remix_block(pedestrians + motor, motor, 6.0206)

    assert remixed == pytest.approx(0.25, abs=1e-5)


def test_remix_results_reports_clipped_blocks() -> None:
    blocks = [block(3, gain=0.2), block(4, gain=40.0), block(5, gain=0.2)]
    results = list(offline_render.replay(iter(blocks), params()))
    assert [result.clipped_samples > 0 for _, result in results] == [False, True, False]

    audio, clipped = offline_render.remix_results(iter(results), 12.0)

    assert clipped == (4,)
    assert audio.shape == (render_core.CHANNELS, 3 * BLOCK_SIZE)


def test_remix_results_on_a_clean_trace() -> None:
    results = list(offline_render.replay(iter(trace([0, 1, 2])), params()))

    audio, clipped = offline_render.remix_results(iter(results), 6.0)

    assert clipped == ()
    assert audio.shape == (render_core.CHANNELS, 3 * BLOCK_SIZE)


def test_write_wav_round_trips_the_samples(tmp_path) -> None:
    audio = np.linspace(-0.9, 0.9, 4 * 100, dtype=np.float32).reshape(4, 100)
    path = tmp_path / "nested" / "episode.wav"

    offline_render.write_wav(path, audio, SAMPLE_RATE)
    rate, read_back = wavfile.read(path)

    assert rate == SAMPLE_RATE
    assert read_back.dtype == np.float32
    assert np.array_equal(read_back.T, audio)


def test_output_name_carries_the_attenuation(tmp_path) -> None:
    job = offline_render.EpisodeJob(
        path=tmp_path / "episode_007.mcap",
        mode="remix",
        assets=tmp_path / "assets.yaml",
        sounds=tmp_path,
        attenuation_db=12.5,
    )

    assert offline_render._output_name(job) == "episode_007.remix_12.5db.wav"


def test_audio_block_deinterleaves_the_payload() -> None:
    channels = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)

    block_out = offline_render.audio_block(
        "/robot/audio/raw_array",
        encoding="32FC1",
        interleaved=True,
        channel_count=4,
        frame_count=3,
        data=channels.T.reshape(-1).tolist(),
    )

    assert np.array_equal(block_out, channels)


def test_audio_block_rejects_a_foreign_encoding() -> None:
    with pytest.raises(ValueError, match="32FC1"):
        offline_render.audio_block(
            "/robot/audio/raw_array",
            encoding="16SC1",
            interleaved=True,
            channel_count=4,
            frame_count=1,
            data=[0.0, 0.0, 0.0, 0.0],
        )


def test_audio_block_rejects_a_truncated_payload() -> None:
    with pytest.raises(ValueError, match="malformed"):
        offline_render.audio_block(
            "/robot/audio/raw_array",
            encoding="32FC1",
            interleaved=True,
            channel_count=4,
            frame_count=3,
            data=[0.0] * 8,
        )


def clip_block(index: int, *, anchor: int) -> render_core.RenderInputs:
    delay, samples = render_core.prepare_clip(asset(), 62.0, -26.0, 3.4)
    return render_core.RenderInputs(
        block_index=index,
        clips=(
            render_core.ClipInput(
                channel=2,
                start=anchor + delay,
                asset_key="bark_a",
                samples=samples,
                anchor=anchor,
                received_volume_db=62.0,
                sensitivity_dbfs_at_94_dbspl=-26.0,
                delay_samples=3.4,
            ),
        ),
        continuous=block(index).continuous,
        drivetrain=block(index).drivetrain,
    )


def test_decoded_trace_replays_to_the_same_audio() -> None:
    blocks = [clip_block(2, anchor=3 * BLOCK_SIZE) if index == 2 else block(index) for index in range(12)]

    expected_state = render_core.RenderState()
    expected_params = params()
    expected = [render_core.render_block(expected_state, item, expected_params) for item in blocks]

    texts = [render_core.render_inputs_to_json(item) for item in blocks]
    replay_params = params()
    replayed = list(offline_render.replay(offline_render.decode_blocks(texts, replay_params.resolve), replay_params))

    for (_, result), reference in zip(replayed, expected, strict=True):
        assert np.array_equal(result.raw, reference.raw)
        assert np.array_equal(result.motor, reference.motor)
