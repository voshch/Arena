from __future__ import annotations

import numpy as np

from task_generator.auditory.procedural_audio import PartitionedConvolver


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
