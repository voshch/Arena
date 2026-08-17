from __future__ import annotations

from pathlib import Path

import numpy as np

from task_generator.auditory.asset_lib import CachedSample
from task_generator.auditory.drivetrain import DrivetrainSpec, DrivetrainVoice
from task_generator.auditory.procedural_audio import (
    LoopingSampleRenderSource,
    PartitionedConvolver,
)


def _cached_sample(values: list[float]) -> CachedSample:
    samples = np.asarray(values, dtype=np.float32)[:, None]
    return CachedSample(
        sample_id="loop",
        path=Path("loop.wav"),
        samples=samples,
        sample_rate=8,
        channels=1,
        duration_sec=len(samples) / 8,
        normalization_dbfs=-6.0,
        tags=frozenset(),
        octave_band_levels_db={},
    )


def test_partitioned_convolver_matches_linear_convolution() -> None:
    rng = np.random.default_rng(7)
    block_size = 32
    signal = rng.standard_normal(block_size * 8).astype(np.float32)
    impulse = rng.standard_normal(75).astype(np.float32)
    convolver = PartitionedConvolver(impulse, block_size)

    rendered = np.concatenate(
        [
            convolver.process(signal[offset : offset + block_size])
            for offset in range(0, len(signal), block_size)
        ]
    )
    expected = np.convolve(signal, impulse)[: len(signal)]

    np.testing.assert_allclose(rendered, expected, rtol=2e-5, atol=2e-5)


def test_looping_sources_share_phase_but_keep_distinct_rir_delays() -> None:
    sample = _cached_sample([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    direct = LoopingSampleRenderSource(
        sample,
        block_size=4,
        loop=True,
        start_frame=2,
    )
    delayed = LoopingSampleRenderSource(
        sample,
        block_size=4,
        loop=True,
        start_frame=2,
    )
    direct.update(
        gain_db=0.0,
        active=True,
        impulse=np.asarray([1.0], dtype=np.float32),
        rir_signature=("direct",),
    )
    delayed.update(
        gain_db=0.0,
        active=True,
        impulse=np.asarray([0.0, 1.0], dtype=np.float32),
        rir_signature=("delayed",),
    )

    direct_block = direct.render(4)[:, 0]
    delayed_block = delayed.render(4)[:, 0]

    assert delayed_block[0] == 0.0
    np.testing.assert_allclose(delayed_block[1:], direct_block[:-1], rtol=2e-5, atol=2e-5)


def test_looping_source_keeps_program_phase_while_inaudible() -> None:
    source = LoopingSampleRenderSource(
        _cached_sample([1.0, 2.0, 3.0, 4.0]),
        block_size=2,
        loop=True,
    )
    source.update(
        gain_db=0.0,
        active=False,
        impulse=None,
        rir_signature=None,
    )
    source.render(2)
    source.update(
        gain_db=0.0,
        active=True,
        impulse=None,
        rir_signature=None,
    )

    resumed = source.render(2)[:, 0]

    assert resumed[-1] == 4.0


def test_drivetrain_runtime_tuning_changes_pitch_and_tonal_level() -> None:
    sample_rate = 8000
    spec = DrivetrainSpec(
        K=2.0 * np.pi * 100.0,
        partials_db=(0.0,),
        n_drivetrains=1,
        v_static=0.0,
        crossfade_s=0.0001,
        sample_rate=sample_rate,
    )
    frames = sample_rate

    baseline = DrivetrainVoice(spec, transfer=False, gain=1.0).render(
        1.0,
        frames,
    )
    tuned = DrivetrainVoice(spec, transfer=False, gain=1.0).render(
        1.0,
        frames,
        frequency_scale=1.5,
        tonal_gain_db=-12.0,
    )

    frequencies = np.fft.rfftfreq(frames, 1.0 / sample_rate)
    baseline_peak = frequencies[np.argmax(np.abs(np.fft.rfft(baseline)))]
    tuned_peak = frequencies[np.argmax(np.abs(np.fft.rfft(tuned)))]
    assert baseline_peak == 100.0
    assert tuned_peak == 150.0
    np.testing.assert_allclose(
        np.sqrt(np.mean(tuned**2)),
        np.sqrt(np.mean(baseline**2)) * 10.0 ** (-12.0 / 20.0),
        rtol=0.01,
    )
