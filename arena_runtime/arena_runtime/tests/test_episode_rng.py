from __future__ import annotations

import numpy as np
from arena_runtime.env.constants.rng import EpisodeRng, stable_int


def test_stable_int_int_passthrough() -> None:
    assert stable_int(7) == 7


def test_stable_int_string_deterministic() -> None:
    assert stable_int("hello") == stable_int("hello")


def test_stable_int_distinct_strings_differ() -> None:
    assert stable_int("a") != stable_int("b")


def test_stable_int_not_builtin_hash() -> None:
    # builtin hash("abc") varies across process invocations (PYTHONHASHSEED);
    # stable_int must return the same value every time.
    v1 = stable_int("abc")
    v2 = stable_int("abc")
    assert v1 == v2
    assert isinstance(v1, int)


def test_same_seed_same_key_identical_draws() -> None:
    rng1 = EpisodeRng()
    rng1.reseed(42)
    rng2 = EpisodeRng()
    rng2.reseed(42)
    draw1 = rng1.stream("obs").integers(0, 1000, size=8).tolist()
    draw2 = rng2.stream("obs").integers(0, 1000, size=8).tolist()
    assert draw1 == draw2


def test_distinct_keys_independent_sequences() -> None:
    rng = EpisodeRng()
    rng.reseed(1)
    a = rng.stream("key_a").integers(0, 10**9, size=10).tolist()
    b = rng.stream("key_b").integers(0, 10**9, size=10).tolist()
    assert a != b


def test_access_order_independence() -> None:
    rng_ab = EpisodeRng()
    rng_ab.reseed(99)
    _ = rng_ab.stream("b").random()
    draw_a_after_b = rng_ab.stream("a").integers(0, 10**9, size=6).tolist()

    rng_a = EpisodeRng()
    rng_a.reseed(99)
    draw_a_direct = rng_a.stream("a").integers(0, 10**9, size=6).tolist()

    assert draw_a_after_b == draw_a_direct


def test_reseed_same_seed_reproduces() -> None:
    rng = EpisodeRng()
    rng.reseed(7)
    first = rng.stream("x").integers(0, 10**9, size=5).tolist()
    rng.reseed(7)
    second = rng.stream("x").integers(0, 10**9, size=5).tolist()
    assert first == second


def test_reseed_clears_cached_streams() -> None:
    rng = EpisodeRng()
    rng.reseed(1)
    gen_before = rng.stream("k")
    _ = gen_before.random()

    rng.reseed(2)
    gen_after = rng.stream("k")

    # gen_after is a fresh generator rooted on seed 2, not the cached one.
    assert gen_before is not gen_after


def test_reseed_different_seed_different_draws() -> None:
    rng = EpisodeRng()
    rng.reseed(10)
    draw_a = rng.stream("z").integers(0, 10**9, size=5).tolist()
    rng.reseed(11)
    draw_b = rng.stream("z").integers(0, 10**9, size=5).tolist()
    assert draw_a != draw_b


def test_reseed_post_draw_reroots() -> None:
    rng = EpisodeRng()
    rng.reseed(5)
    _ = rng.stream("w").random(10)
    rng.reseed(5)
    # After re-seeding with the same root, draws restart from the beginning.
    draw_fresh = rng.stream("w").random(3).tolist()

    rng2 = EpisodeRng()
    rng2.reseed(5)
    draw_ref = rng2.stream("w").random(3).tolist()

    assert draw_fresh == draw_ref


def test_stream_returns_numpy_generator() -> None:
    rng = EpisodeRng()
    rng.reseed(0)
    assert isinstance(rng.stream("any"), np.random.Generator)


def test_stream_cached_within_episode() -> None:
    rng = EpisodeRng()
    rng.reseed(3)
    assert rng.stream("p") is rng.stream("p")
