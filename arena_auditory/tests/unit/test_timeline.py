from arena_auditory.hearing.timeline import AudioTimeline

FS = 16000
BLOCK = 320
BLOCK_NS = 20_000_000


def _contiguous(t: AudioTimeline, start_ns: int, blocks: int) -> None:
    for i in range(blocks):
        assert t.observe(start_ns + i * BLOCK_NS, BLOCK) == 0


def test_contiguous_stream_keeps_stamps():
    t = AudioTimeline(FS, max_gap_samples=FS)
    _contiguous(t, 5_000_000_000, 10)
    assert t.time_ns(t.samples) == 5_000_000_000 + 10 * BLOCK_NS
    assert t.gaps == 0


def test_gap_is_filled_with_silence():
    t = AudioTimeline(FS, max_gap_samples=FS)
    _contiguous(t, 0, 3)
    fill = t.observe(8 * BLOCK_NS, BLOCK)  # blocks 3..7 never arrived
    assert fill == 5 * BLOCK
    assert t.time_ns(t.samples) == 9 * BLOCK_NS
    assert t.gaps == 1


def test_long_gap_is_capped_and_reanchored():
    t = AudioTimeline(FS, max_gap_samples=FS)
    _contiguous(t, 0, 1)
    fill = t.observe(10_000_000_000, BLOCK)
    assert fill == FS
    assert t.time_ns(t.samples) == 10_000_000_000 + BLOCK_NS


def test_rewind_reanchors():
    t = AudioTimeline(FS, max_gap_samples=FS)
    _contiguous(t, 50_000_000_000, 5)
    assert t.observe(1_000_000_000, BLOCK) == 0
    assert t.rewinds == 1
    assert t.time_ns(t.samples) == 1_000_000_000 + BLOCK_NS


def test_jitter_within_a_sample_is_ignored():
    t = AudioTimeline(FS, max_gap_samples=FS)
    _contiguous(t, 0, 1)
    assert t.observe(BLOCK_NS + 40_000, BLOCK) == 0
    assert t.gaps == 0
