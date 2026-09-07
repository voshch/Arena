from arena_auditory.render_clock import RenderCursor

BLOCK = 20_000_000


def test_first_call_anchors_and_owes_one_block():
    cursor = RenderCursor(block_ns=BLOCK, max_catchup=10)
    assert cursor.owed(5_000_000_000) == (1, 0)
    assert cursor.start_ns == 5_000_000_000


def test_no_new_block_before_period_elapses():
    cursor = RenderCursor(block_ns=BLOCK, max_catchup=10)
    cursor.owed(0)
    assert cursor.owed(BLOCK - 1) == (0, 0)
    assert cursor.owed(BLOCK) == (1, 0)


def test_catch_up_within_cap():
    cursor = RenderCursor(block_ns=BLOCK, max_catchup=10)
    cursor.owed(0)
    assert cursor.owed(4 * BLOCK) == (4, 0)
    assert cursor.rendered == 5


def test_skip_past_cap():
    cursor = RenderCursor(block_ns=BLOCK, max_catchup=10)
    cursor.owed(0)
    assert cursor.owed(25 * BLOCK) == (10, 15)
    assert cursor.skipped == 15
    assert cursor.owed(25 * BLOCK) == (0, 0)
    assert cursor.owed(26 * BLOCK) == (1, 0)


def test_clock_going_backwards_owes_nothing():
    cursor = RenderCursor(block_ns=BLOCK, max_catchup=10)
    cursor.owed(3 * BLOCK)
    assert cursor.owed(0) == (0, 0)
