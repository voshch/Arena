from __future__ import annotations

import math

import numpy as np

from arena_auditory.hearing.policy import State, YieldMachine, YieldParams, compose_masks, mass_split, paint_lane


def test_compose_masks_takes_the_lowest_nonzero_limit() -> None:
    layers = [
        np.array([0, 50, 0], dtype=np.int8),
        np.array([30, 0, 0], dtype=np.int8),
        np.array([0, 0, 0], dtype=np.int8),
    ]
    out = compose_masks(*layers)
    assert list(out) == [30, 50, 0]


def test_compose_masks_single_layer_returns_itself() -> None:
    layer = np.array([5, 0, 10], dtype=np.int8)
    out = compose_masks(layer)
    assert list(out) == [5, 0, 10]


def test_compose_masks_all_zero_layers_stay_zero() -> None:
    out = compose_masks(np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8))
    assert list(out) == [0, 0, 0]


def test_paint_lane_marks_a_disc_around_each_point() -> None:
    mask = paint_lane((50, 50), (0.0, 0.0), 0.1, [(2.5, 2.5)], 0.3, 40)
    assert mask[25, 25] == 40
    assert mask[25, 30] == 0
    count = int((mask != 0).sum())
    expected = math.pi * 3.0**2
    assert 0.8 * expected <= count <= 1.2 * expected


def test_paint_lane_with_no_points_is_all_zero() -> None:
    mask = paint_lane((50, 50), (0.0, 0.0), 0.1, [], 0.3, 40)
    assert not mask.any()


def test_mass_split_ahead_behind_and_total() -> None:
    belief = np.zeros((100, 100))
    belief[50, 50] = 1.0
    belief[50, 10] = 1.0
    ahead, behind, total = mass_split(belief, (0.0, 0.0), 0.1, (5.0, 5.0), 1.0, (3.0, 5.0), (1.0, 0.0))
    assert ahead == 1.0
    assert behind == 1.0
    assert total == 2.0


def _params() -> YieldParams:
    return YieldParams(yield_fraction=0.5, release_fraction=0.25, min_yield_s=3.0, yield_timeout_s=15.0, recede_s=2.0)


def test_cruise_to_listen_on_approach() -> None:
    m = YieldMachine(params=_params())
    state = m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    assert state is State.LISTEN


def test_listen_to_yield_on_a_strong_contact_ahead() -> None:
    m = YieldMachine(params=_params())
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    state = m.step(1.0, in_approach=True, past_bend=False, frac_ahead=0.8, ahead=1.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    assert state is State.YIELD
    assert m.yield_count == 1
    assert m.contact is True


def test_yield_holds_until_min_yield_s_then_passes_once_behind_leads() -> None:
    m = YieldMachine(params=_params())
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.8, ahead=1.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    entered = m.entered_at
    state = m.step(entered + 1.0, in_approach=True, past_bend=False, frac_ahead=0.9, ahead=0.5, behind=1.0, event_age_s=0.0, receding_s=0.0)
    assert state is State.YIELD
    state = m.step(entered + 4.0, in_approach=True, past_bend=False, frac_ahead=0.9, ahead=0.5, behind=1.0, event_age_s=0.0, receding_s=0.0)
    assert state is State.PASS


def test_yield_to_pass_on_timeout() -> None:
    m = YieldMachine(params=_params())
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.8, ahead=1.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    state = m.step(16.0, in_approach=True, past_bend=False, frac_ahead=0.9, ahead=1.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    assert state is State.PASS


def test_yield_to_pass_when_contact_fades() -> None:
    m = YieldMachine(params=_params())
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.8, ahead=1.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    state = m.step(1.0, in_approach=True, past_bend=False, frac_ahead=0.1, ahead=1.0, behind=0.0, event_age_s=5.0, receding_s=0.0)
    assert state is State.PASS


def test_pass_to_cruise_once_past_the_bend() -> None:
    m = YieldMachine(params=_params())
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.8, ahead=1.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    m.step(1.0, in_approach=True, past_bend=False, frac_ahead=0.1, ahead=1.0, behind=0.0, event_age_s=5.0, receding_s=0.0)
    state = m.step(2.0, in_approach=True, past_bend=True, frac_ahead=0.1, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    assert state is State.CRUISE


def test_listen_to_cruise_when_approach_ends() -> None:
    m = YieldMachine(params=_params())
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    state = m.step(1.0, in_approach=False, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    assert state is State.CRUISE


def test_new_bend_resets_contact_and_drops_yield_or_pass_to_cruise() -> None:
    m = YieldMachine(params=_params())
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.0, ahead=0.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    m.step(0.0, in_approach=True, past_bend=False, frac_ahead=0.8, ahead=1.0, behind=0.0, event_age_s=0.0, receding_s=0.0)
    assert m.state is State.YIELD
    assert m.contact is True
    m.new_bend()
    assert m.state is State.CRUISE
    assert m.contact is False
