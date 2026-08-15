from __future__ import annotations

from arena_cam.fly import AXES, COMMANDS, intent_from_keys


def test_opposing_keys_cancel() -> None:
    assert intent_from_keys({"w", "s"}).forward == 0.0


def test_axes_combine_and_clamp() -> None:
    intent = intent_from_keys({"w", "d", "e", "left", "up", "]"})
    assert intent.forward == 1.0
    assert intent.right == 1.0
    assert intent.up == 1.0
    assert intent.yaw == 1.0
    assert intent.pitch == 1.0
    assert intent.fov == 1.0


def test_unknown_keys_are_ignored() -> None:
    intent = intent_from_keys({"w", "k", "f"})
    assert intent.forward == 1.0
    assert intent.right == 0.0


def test_modifiers_ride_along() -> None:
    assert intent_from_keys(set(), boost=True).scale() > 1.0
    assert intent_from_keys(set(), crawl=True).scale() < 1.0
    assert intent_from_keys(set()).scale() == 1.0


def test_held_and_tapped_keys_do_not_overlap() -> None:
    assert set(AXES) & set(COMMANDS) == set()
