from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest
from arena_auditory.render_core import (
    ClipInput,
    ContinuousInput,
    DrivetrainInput,
    DrivetrainTuning,
    RenderInputs,
    RenderParams,
    RenderState,
    prepare_clip,
    render_block,
    render_inputs_from_json,
    render_inputs_to_json,
)
from arena_auditory.spatial_audio import (
    CHANNEL_NAMES,
    apply_monitor_controls,
    calibrate_mems,
    fractional_delay,
    gcc_phat,
    geometric_delays_seconds,
    headphone_stereo,
    hearing_waveform,
    monitor_amplify,
    ramped_read,
    rectangular_array,
    streaming_fractional_delays,
    transform_array,
)


def _delays(source: tuple[float, float, float]) -> np.ndarray:
    microphones = rectangular_array()
    return geometric_delays_seconds(
        source,
        tuple(mic.position_m for mic in microphones),
    )


def test_four_mic_geometry_matches_jackal_chassis_inset() -> None:
    microphones = rectangular_array()
    assert tuple(mic.name for mic in microphones) == CHANNEL_NAMES
    assert microphones[0].position_m == (0.19, 0.135, 0.22)
    assert microphones[1].position_m == (0.19, -0.135, 0.22)
    assert microphones[2].position_m == (-0.19, 0.135, 0.22)
    assert microphones[3].position_m == (-0.19, -0.135, 0.22)
    assert [round(math.degrees(mic.yaw_rad)) for mic in microphones] == [45, -45, 135, -135]


def test_array_moves_rigidly_with_robot() -> None:
    microphones = rectangular_array()
    world = transform_array(
        microphones,
        robot_position_m=(2.0, -1.0, 0.1),
        robot_yaw_rad=math.pi / 2.0,
    )
    assert np.allclose(world[0], (2.0 - 0.135, -1.0 + 0.19, 0.32))
    original_distances = np.linalg.norm(
        np.asarray([mic.position_m for mic in microphones])[:, None, :]
        - np.asarray([mic.position_m for mic in microphones])[None, :, :],
        axis=2,
    )
    moved_distances = np.linalg.norm(np.asarray(world)[:, None, :] - np.asarray(world)[None, :, :], axis=2)
    assert np.allclose(original_distances, moved_distances)


def test_left_right_and_front_rear_geometric_arrivals() -> None:
    left = _delays((0.0, 5.0, 0.22))
    right = _delays((0.0, -5.0, 0.22))
    front = _delays((5.0, 0.0, 0.22))
    rear = _delays((-5.0, 0.0, 0.22))
    assert max(left[0], left[2]) < min(left[1], left[3])
    assert max(right[1], right[3]) < min(right[0], right[2])
    assert max(front[0], front[1]) < min(front[2], front[3])
    assert max(rear[2], rear[3]) < min(rear[0], rear[1])
    assert (left[1] - left[0]) == pytest.approx(0.000787, abs=2e-6)
    assert (front[2] - front[0]) == pytest.approx(0.001107, abs=2e-6)


def test_independent_channels_keep_tdoa_and_stereo_asymmetry() -> None:
    rate = 16000
    delays = _delays((0.0, 5.0, 0.22))
    pulse = np.hanning(64).astype(np.float32)
    audio = np.zeros((4, 512), dtype=np.float32)
    base = 80
    for channel, delay in enumerate(delays - delays.min()):
        start = base + round(float(delay) * rate)
        audio[channel, start:start + len(pulse)] = pulse * (1.0 - 0.08 * channel)
    assert not np.array_equal(audio[0], audio[1])
    estimate, confidence = gcc_phat(audio[1], audio[0], sample_rate_hz=rate, max_tau_seconds=0.002)
    assert estimate > 0.0
    assert abs(estimate - (delays[1] - delays[0])) < 1.0 / rate
    assert confidence > 0.0
    stereo = headphone_stereo(audio)
    assert not np.array_equal(stereo[0], stereo[1])


def test_detection_fusion_selects_strongest_without_phase_averaging() -> None:
    audio = np.zeros((4, 32), dtype=np.float32)
    audio[0] = 0.1
    audio[2] = -0.8
    assert np.array_equal(hearing_waveform(audio), audio[2])


def test_mems_calibration_and_silence_are_stable() -> None:
    tone = np.sin(np.linspace(0.0, 4.0 * math.pi, 1600, endpoint=False)).astype(np.float32)
    calibrated = calibrate_mems(tone, 94.0)
    measured_dbfs = 20.0 * math.log10(float(np.sqrt(np.mean(calibrated**2))))
    assert measured_dbfs == pytest.approx(-26.0, abs=0.02)
    assert np.array_equal(calibrate_mems(np.zeros(64, dtype=np.float32), 80.0), np.zeros(64, dtype=np.float32))


def test_fractional_delay_is_not_rounded_away() -> None:
    integer, delayed = fractional_delay(np.asarray([1.0, 0.0], dtype=np.float32), 3.25)
    assert integer == 3
    assert delayed[0] == pytest.approx(0.75)
    assert delayed[1] == pytest.approx(0.25)


def test_disable_mute_solo_and_reenable_controls() -> None:
    audio = np.arange(32, dtype=np.float32).reshape(4, 8)
    assert not np.any(apply_monitor_controls(audio, enabled=False))
    assert not np.any(apply_monitor_controls(audio, muted=True))
    assert np.array_equal(apply_monitor_controls(audio, enabled=True), audio)
    solo = apply_monitor_controls(audio, solo_channel="rear_left")
    assert np.array_equal(solo[2], audio[2])
    assert not np.any(solo[[0, 1, 3]])


def test_monitor_gain_does_not_change_calibrated_raw_audio() -> None:
    raw = np.asarray([0.001, -0.001, 0.02], dtype=np.float32)
    untouched = raw.copy()
    monitored = monitor_amplify(raw, gain_db=40.0, limit=0.98)
    np.testing.assert_allclose(monitored[:2], [0.1, -0.1], atol=1e-6)
    assert monitored[2] == pytest.approx(0.98)
    np.testing.assert_array_equal(raw, untouched)


def test_streaming_fractional_delays_retain_history_between_blocks() -> None:
    first, history = streaming_fractional_delays(
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.asarray([0.0, 1.5]),
    )
    np.testing.assert_allclose(first[0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(first[1], [0.0, 0.5, 1.5, 2.5])

    second, _ = streaming_fractional_delays(
        np.asarray([5.0, 6.0], dtype=np.float32),
        np.asarray([0.0, 1.5]),
        history,
    )
    np.testing.assert_allclose(second[0], [5.0, 6.0])
    np.testing.assert_allclose(second[1], [3.5, 4.5])


def test_ramped_read_constant_delay_is_a_plain_shift() -> None:
    loop = np.arange(8, dtype=np.float32)
    out = ramped_read(loop, 0.0, 3.0, 3.0, 8, loop=True)
    np.testing.assert_allclose(out, [5, 6, 7, 0, 1, 2, 3, 4])
    clip = ramped_read(loop, 0.0, 3.0, 3.0, 8, loop=False)
    np.testing.assert_allclose(clip, [0, 0, 0, 0, 1, 2, 3, 4])


def test_ramped_read_is_continuous_across_blocks_and_reaches_the_target() -> None:
    fs = 16000
    tone = np.sin(2.0 * np.pi * 440.0 * np.arange(fs, dtype=np.float64) / fs).astype(np.float32)
    block = 320
    delay, target = 100.0, 103.0
    blocks = []
    for index in range(3):
        blocks.append(ramped_read(tone, index * block, delay, target, block, loop=True))
        delay = target
    signal = np.concatenate(blocks)
    steps = np.abs(np.diff(signal))
    assert steps.max() < 1.05 * np.abs(np.diff(tone[: 2 * block])).max()
    settled = ramped_read(tone, 3 * block, target, target, block, loop=True)
    expected = tone[(np.arange(3 * block, 4 * block) - 103) % tone.size]
    np.testing.assert_allclose(settled, expected, atol=1e-6)


_RENDER_BLOCK = 8
_RENDER_RATE = 16000
_MOTOR_TUNING = DrivetrainTuning(
    volume_db=0.0,
    frequency_scale=1.0,
    tonal_gain_db=0.0,
    broadband_gain_db=-12.0,
    speed_exponent=1.5,
    velocity_smoothing_seconds=0.0,
)


def _asset_samples(asset_key: str) -> np.ndarray:
    """A per-key waveform whose samples are exact in float32."""
    steps = np.arange(48, dtype=np.int64)
    pattern = ((steps * 7 + len(asset_key) * 13) % 17) - 8
    return np.ascontiguousarray((pattern / 256.0).astype(np.float32))


def _render_params() -> RenderParams:
    return RenderParams(
        block_size=_RENDER_BLOCK,
        sample_rate=_RENDER_RATE,
        resolve=_asset_samples,
    )


def _drivetrain_input(**overrides: object) -> DrivetrainInput:
    fields: dict[str, object] = {
        "source_id": "jackal",
        "deterministic_seed": 7,
        "gains": (0.5, 0.25, 0.125, 0.0625),
        "delay_samples": (0.0, 1.5, 2.0, 3.25),
        "active_channels": (True, True, True, True),
        "left_velocity": 0.8,
        "right_velocity": 0.6,
        "active": True,
        "tuning": _MOTOR_TUNING,
        "source_agent_id": 0,
        "source_agent_name": "jackal",
        "sound_type": "motor",
        "asset_id": "drivetrain",
    }
    fields.update(overrides)
    return DrivetrainInput(**fields)


def _continuous_input(gain: float = 0.5) -> ContinuousInput:
    return ContinuousInput(
        source_id="fan",
        channel=1,
        asset_key="fan_loop",
        gain=gain,
        delay_target=1.5,
        loop=True,
        program_start=0,
    )


def _clip(channel: int, start: int, asset_key: str, samples: np.ndarray) -> ClipInput:
    """A clip whose scheduling parameters the accumulation never reads."""
    return ClipInput(
        channel=channel,
        start=start,
        asset_key=asset_key,
        samples=samples,
        anchor=start,
        received_volume_db=94.0,
        sensitivity_dbfs_at_94_dbspl=-26.0,
        delay_samples=0.0,
    )


def _clip_input() -> ClipInput:
    return _clip(0, 4, "thud", (_asset_samples("thud")[:9] * 4.0).astype(np.float32))


def _render_scenario(
    blocks: int,
    *,
    continuous_gain: float = 0.5,
    with_clip: bool = True,
    **drivetrain_overrides: object,
) -> tuple[np.ndarray, np.ndarray, int]:
    state = RenderState()
    params = _render_params()
    raw: list[np.ndarray] = []
    motor: list[np.ndarray] = []
    clipped = 0
    for index in range(blocks):
        result = render_block(
            state,
            RenderInputs(
                block_index=index,
                clips=(_clip_input(),) if with_clip and index == 0 else (),
                continuous=(_continuous_input(continuous_gain),),
                drivetrain=(_drivetrain_input(**drivetrain_overrides),),
            ),
            params,
        )
        raw.append(result.raw)
        motor.append(result.motor)
        clipped += result.clipped_samples
    return np.concatenate(raw, axis=1), np.concatenate(motor, axis=1), clipped


_GOLDEN_RAW = np.array(
    [
        [0.0, -1.2368853276711889e-05, -3.6360666854307055e-05, -4.039225314045325e-05, -0.1093534529209137, 0.00012824783334508538, 0.10957394540309906, -0.04678588733077049, 0.06238681450486183, -0.09390002489089966, 0.015591255389153957, 0.1251825988292694, -0.03098478354513645, -2.6669868020690046e-06, -0.0004496758047025651, -0.0008444450213573873],
        [-0.0068359375, -0.005859375, -0.004885904490947723, 0.008776879869401455, 0.005840186960995197, 0.0029249766375869513, 3.744910645764321e-05, -0.002847888972610235, -0.005787360016256571, -0.008795080706477165, 0.004817009903490543, 0.0019071826245635748, -0.0009393480722792447, -0.0037942954804748297, -0.006770300213247538, 0.0067228516563773155],
        [0.0, 0.0, 0.0, -3.092213319177972e-06, -9.090166713576764e-06, -1.0098063285113312e-05, 5.387149030866567e-06, 3.2061958336271346e-05, 4.973656905349344e-05, 2.2278529286268167e-05, -2.82963301287964e-05, -3.750628457055427e-05, -8.436130883637816e-06, 4.565057315630838e-05, 6.630394636886194e-05, -6.667467005172512e-07],
        [0.0, 0.0, 0.0, 0.0, -1.1595800515351584e-06, -3.795339125645114e-06, -4.923044798488263e-06, 7.579229759357986e-07, 1.2696627891273238e-05, 2.2658958187093958e-05, 1.4571519386663567e-05, -7.826307410141453e-06, -1.7601898434804752e-05, -7.851835107430816e-06, 1.6064448573160917e-05, 3.057029971387237e-05],
    ],
    dtype=np.float32,
)


def test_render_block_reproduces_the_recorded_accumulation() -> None:
    raw, _, clipped = _render_scenario(2)
    assert clipped == 0
    assert np.array_equal(raw, _GOLDEN_RAW)


def test_motor_stem_splits_the_raw_block_without_loss() -> None:
    raw, motor, clipped = _render_scenario(2)
    assert clipped == 0
    assert np.max(np.abs(motor)) > 0.0
    assert np.array_equal(raw, (raw - motor) + motor)


def test_motor_stem_is_silent_when_no_drivetrain_channel_is_active() -> None:
    raw, motor, _ = _render_scenario(
        2,
        gains=(0.0, 0.0, 0.0, 0.0),
        active_channels=(False, False, False, False),
        active=False,
    )
    assert np.array_equal(motor, np.zeros_like(motor))
    assert np.max(np.abs(raw)) > 0.0


def test_motor_volume_change_remixes_as_a_scalar_on_the_stem() -> None:
    attenuation_db = 6.0
    quiet = replace(_MOTOR_TUNING, volume_db=_MOTOR_TUNING.volume_db - attenuation_db)
    raw, motor, _ = _render_scenario(3, continuous_gain=0.02, with_clip=False)
    remixed_raw, remixed_motor, _ = _render_scenario(
        3,
        continuous_gain=0.02,
        with_clip=False,
        tuning=quiet,
    )
    gain = 10.0 ** (-attenuation_db / 20.0)
    assert np.max(np.abs(motor)) > 1e-6
    assert not np.array_equal(remixed_motor, motor)
    np.testing.assert_allclose(remixed_raw, (raw - motor) + gain * motor, rtol=1e-5, atol=1e-8)


def test_clipped_samples_counts_the_clamped_output() -> None:
    over = np.full(_RENDER_BLOCK, 2.0, dtype=np.float32)
    under = np.full(_RENDER_BLOCK, -3.0, dtype=np.float32)
    inside = np.full(_RENDER_BLOCK, 0.5, dtype=np.float32)
    result = render_block(
        RenderState(),
        RenderInputs(
            block_index=0,
            clips=(
                _clip(0, 0, "over", over),
                _clip(1, 0, "under", under),
                _clip(2, 0, "inside", inside),
            ),
            continuous=(),
            drivetrain=(),
        ),
        _render_params(),
    )
    assert result.clipped_samples == 2 * _RENDER_BLOCK
    assert np.array_equal(result.raw[0], np.ones(_RENDER_BLOCK, dtype=np.float32))
    assert np.array_equal(result.raw[1], -np.ones(_RENDER_BLOCK, dtype=np.float32))
    assert np.array_equal(result.raw[2], inside)
    assert np.array_equal(result.motor, np.zeros_like(result.motor))


def _traced_clip() -> ClipInput:
    """A clip built the way the live driver builds one, so the trace can rebuild it."""
    anchor, received_volume_db, sensitivity, delay_samples = 4, 88.0, -26.0, 3.25
    delay, samples = prepare_clip(_asset_samples("thud"), received_volume_db, sensitivity, delay_samples)
    return ClipInput(
        channel=0,
        start=anchor + delay,
        asset_key="thud",
        samples=samples,
        anchor=anchor,
        received_volume_db=received_volume_db,
        sensitivity_dbfs_at_94_dbspl=sensitivity,
        delay_samples=delay_samples,
    )


def _traced_inputs(block_index: int) -> RenderInputs:
    return RenderInputs(
        block_index=block_index,
        clips=(_traced_clip(),) if block_index == 0 else (),
        continuous=(_continuous_input(),),
        drivetrain=(_drivetrain_input(),),
    )


def test_render_inputs_json_round_trip_renders_the_same_blocks() -> None:
    params = _render_params()
    live_state, replay_state = RenderState(), RenderState()
    live_raw: list[np.ndarray] = []
    live_motor: list[np.ndarray] = []
    for index in range(3):
        inputs = _traced_inputs(index)
        replayed = render_inputs_from_json(render_inputs_to_json(inputs), _asset_samples)
        assert replayed.block_index == inputs.block_index
        assert replayed.continuous == inputs.continuous
        assert replayed.drivetrain == inputs.drivetrain
        if inputs.clips:
            assert np.max(np.abs(inputs.clips[0].samples)) > 0.0
            assert np.array_equal(replayed.clips[0].samples, inputs.clips[0].samples)
            assert replayed.clips[0].start == inputs.clips[0].start
        live = render_block(live_state, inputs, params)
        replay = render_block(replay_state, replayed, params)
        assert np.array_equal(live.raw, replay.raw)
        assert np.array_equal(live.motor, replay.motor)
        assert live.clipped_samples == replay.clipped_samples
        live_raw.append(live.raw)
        live_motor.append(live.motor)
    raw = np.concatenate(live_raw, axis=1)
    motor = np.concatenate(live_motor, axis=1)
    assert np.max(np.abs(raw[0])) > 0.0
    assert np.max(np.abs(raw[1])) > 0.0
    assert np.max(np.abs(motor)) > 0.0


def test_render_inputs_json_rejects_a_clip_that_replays_at_another_sample() -> None:
    payload = json.loads(render_inputs_to_json(_traced_inputs(0)))
    payload["clips"][0]["start"] += 1
    with pytest.raises(ValueError, match="trace recorded"):
        render_inputs_from_json(json.dumps(payload), _asset_samples)


def test_render_inputs_json_carries_no_sample_arrays() -> None:
    clip = _traced_clip()
    assert clip.samples.size > 4
    payload = json.loads(render_inputs_to_json(_traced_inputs(0)))
    assert [sorted(entry) for entry in payload["clips"]] == [
        [
            "anchor",
            "asset_key",
            "channel",
            "delay_samples",
            "received_volume_db",
            "sensitivity_dbfs_at_94_dbspl",
            "start",
        ]
    ]

    def _no_waveforms(node: object) -> None:
        if isinstance(node, dict):
            for item in node.values():
                _no_waveforms(item)
        elif isinstance(node, list):
            for item in node:
                _no_waveforms(item)
            assert all(isinstance(item, dict) for item in node) or len(node) <= len(CHANNEL_NAMES)

    _no_waveforms(payload)
