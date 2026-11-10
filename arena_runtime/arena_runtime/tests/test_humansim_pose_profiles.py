"""Pose profiles of the agent types the arena_humansim engine reports, resolved before the publish tick reads them."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_humansim_msgs")

from arena_humansim.core.interaction_kinds import InteractionType
from arena_humansim_msgs.msg import AgentMeta
from arena_people_msgs.msg import Pedestrian

from arena_runtime.humans.arena_humansim.arena_humansim import ArenaHumanSimulator
from arena_runtime.humans import hands
from arena_runtime.humans.hands import Grip, grasp, held_pose
from arena_runtime.humans.profile import load_library


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


PUSHER = "name: pusher\nextends: adult\npose:\n  hold: {r: 0.0125}\n  while:\n    PUSH_WHEELCHAIR:\n      walk: {base: walk_pushwheelchair}\n      hold: {l: 0.014, r: 0.014}\n"


def test_an_interaction_grip_is_solved_when_its_type_arrives(tmp_path: Path) -> None:
    pusher = tmp_path / "pusher.yaml"
    pusher.write_text(PUSHER)
    adapter = _adapter()
    adapter._remember_agent_type_path(str(pusher))
    grasp.cache_clear()

    adapter._agent_meta_callback(AgentMeta(agent_id=[3], agent_type=["pusher"]))

    profile = adapter._pose_profiles[str(pusher)]
    assert profile.hold == {"r": Grip(0.0125)}
    assert profile.while_in("PUSH_WHEELCHAIR").hold == {"l": Grip(0.014), "r": Grip(0.014)}
    assert grasp.cache_info().misses == 3
    assert adapter.node.logger.warnings == []


def test_unsolvable_grips_leave_the_hands_unheld_and_the_push_keeps_its_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> None:
        raise OSError("no template hand")

    pusher = tmp_path / "pusher.yaml"
    pusher.write_text(PUSHER)
    monkeypatch.setattr(hands, "load_template", missing)
    grasp.cache_clear()
    adapter = _adapter()
    adapter._remember_agent_type_path(str(pusher))

    adapter._agent_meta_callback(AgentMeta(agent_id=[3], agent_type=["pusher"]))

    profile = adapter._pose_profiles[str(pusher)]
    pushing = profile.while_in("PUSH_WHEELCHAIR")
    assert profile.hold == {} and pushing.hold == {}
    assert pushing.walk.joints == load_library()["walk_pushwheelchair"].joints
    assert sorted("while in PUSH_WHEELCHAIR" in warning for warning in adapter.node.logger.warnings) == [False, True]


def test_a_ped_in_a_humansim_interaction_names_it_for_its_pose_profile() -> None:
    adapter = _adapter()
    pushing = Pedestrian(interaction_id=4, interaction_type=int(InteractionType.PUSH_WHEELCHAIR))
    free = Pedestrian(interaction_id=-1, interaction_type=int(InteractionType.PUSH_WHEELCHAIR))

    assert adapter._interaction(pushing) == "PUSH_WHEELCHAIR"
    assert adapter._interaction(free) is None
