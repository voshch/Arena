"""Pose profiles of the agent types the arena_humansim engine reports, resolved before the publish tick reads them."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim_msgs")

from arena_humansim_msgs.msg import AgentMeta

from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.humans.hands import Grip, held_pose


class _Logger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def get_child(self, _name: str) -> _Logger:
        return self

    def warning(self, message: str, /) -> None:
        self.warnings.append(message)


class _Node:
    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self) -> _Logger:
        return self.logger


def _adapter() -> ArenaHumanSimulator:
    adapter = object.__new__(ArenaHumanSimulator)
    adapter._NodeInterface__node = _Node()
    adapter._pose_profiles = {}
    adapter._agent_types = {}
    adapter._agent_type_paths = {}
    return adapter


def test_agent_meta_resolves_the_reported_types_with_their_grips(tmp_path: Path) -> None:
    caner = tmp_path / "caner.yaml"
    caner.write_text("name: caner\nextends: adult\npose:\n  hold: {r: 0.0125}\n")
    adapter = _adapter()
    adapter._remember_agent_type_path(str(caner))

    adapter._agent_meta_callback(AgentMeta(agent_id=[3, 4], agent_type=["caner", "adult"]))

    assert set(adapter._pose_profiles) == {str(caner), "adult"}
    assert adapter._pose_profiles[str(caner)].hold == {"r": Grip(0.0125)}
    assert adapter._pose_profiles["adult"].hold == {}
    assert adapter._pose_profile_for(str(caner)) is adapter._pose_profiles[str(caner)]
    assert set(held_pose(adapter._pose_profiles[str(caner)].hold)) >= {"r_curl_index", "r_curl_thumb"}
    assert adapter.node.logger.warnings == []


def test_an_interaction_grip_is_solved_with_its_type_and_dropped_alone_when_it_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import arena_runtime.humans as humans

    pusher = tmp_path / "pusher.yaml"
    pusher.write_text("name: pusher\nextends: adult\npose:\n  hold: {r: 0.0125}\n  while:\n    PUSH_WHEELCHAIR:\n      walk: {base: walk_pushwheelchair}\n      hold: {l: 0.014, r: 0.014}\n")
    solved: list[dict[str, Grip]] = []

    def fussy(hold: dict[str, Grip]) -> dict[str, float]:
        solved.append(dict(hold))
        if "l" in hold:
            raise ValueError("no grip")
        return held_pose(hold)

    monkeypatch.setattr(humans, "held_pose", fussy)
    adapter = _adapter()
    adapter._remember_agent_type_path(str(pusher))

    adapter._agent_meta_callback(AgentMeta(agent_id=[3], agent_type=["pusher"]))

    profile = adapter._pose_profiles[str(pusher)]
    assert {"l": Grip(0.014), "r": Grip(0.014)} in solved, "the push grip is solved when the type arrives, not in the publish tick"
    assert profile.hold == {"r": Grip(0.0125)}
    assert profile.while_in("PUSH_WHEELCHAIR").hold == {}
    assert profile.while_in("PUSH_WHEELCHAIR").walk == profile.during["PUSH_WHEELCHAIR"].walk != profile.walk, "the push keeps its own walk"
    assert len(adapter.node.logger.warnings) == 1 and "PUSH_WHEELCHAIR" in adapter.node.logger.warnings[0]
