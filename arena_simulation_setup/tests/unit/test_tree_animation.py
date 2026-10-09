"""Animation assets, offline: clip round trip, layout listing, and the clips a scenario's agents reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_simulation_setup.tree import SimplePathResolver
from arena_simulation_setup.tree.assets import Animation
from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier, AnimationView, referenced_clips, write_clip
from arena_simulation_setup.tree.World.Scenario import ScenarioView


def _frames(n: int = 4) -> list[dict]:
    return [{"t": 0.05 * k, "angles": {"l_elbow": 0.1 * k, "r_elbow": -0.1 * k}, "root_xy_yaw": [0.0, 0.0, 0.01 * k], "animation_state": 0} for k in range(n)]


def test_clip_round_trips_frames_and_meta(tmp_path: Path) -> None:
    frames = _frames()
    frames[0]["angles"].pop("r_elbow")  # a recording that predates a DOF reads it back as 0.0
    directory = write_clip(tmp_path / "Common" / "Animation" / "wave_test", frames, {"fps": 20.0, "loop": True, "joints": "upper_body"})

    view = AnimationIdentifier("wave_test").load(directory)
    assert isinstance(view, AnimationView) and view.name == "wave_test"
    assert view.meta == {"name": "wave_test", "path": "Animation/wave_test", "fps": 20.0, "loop": True, "joints": "upper_body"}
    clip = view.clip
    assert clip.joint_names == ("l_elbow", "r_elbow") and clip.angles.shape == (4, 2) and clip.fps == 20.0
    back = clip.frames()
    assert back[0]["angles"] == {"l_elbow": 0.0, "r_elbow": 0.0}
    assert [f["angles"] for f in back[1:]] == [f["angles"] for f in frames[1:]]
    assert [f["t"] for f in back] == [f["t"] for f in frames] and back[3]["root_xy_yaw"] == pytest.approx([0.0, 0.0, 0.03])
    assert view.extra("table.npz") == directory / "table.npz"


def test_clip_arrays_load_without_pickle(tmp_path: Path) -> None:
    directory = write_clip(tmp_path / "Animation" / "plain", _frames(), {})
    with np.load(directory / "plain.npz", allow_pickle=False) as z:
        assert set(z.files) == {"joint_names", "angles", "t", "root_xy_yaw", "animation_state"}


def test_identifier_layout_and_listing(tmp_path: Path) -> None:
    ident = AnimationIdentifier.parse("hug_v7_hero_w")
    assert ident.shortname == "Common/Animation/hug_v7_hero_w" and ident.relpath() == Path("Common/Animation/hug_v7_hero_w")
    for name in ("wave", "hug"):
        write_clip(tmp_path / "Common" / "Animation" / name, _frames(), {})
    resolver = SimplePathResolver(AnimationIdentifier, path=tmp_path)
    assert {i.name for i in resolver.listall()} == {"wave", "hug"}


def test_referenced_clips_reads_every_attention_clip_form() -> None:
    config = {
        "sequences": {
            "a": {"steps": {"call": {"attention": {"clip": "beckon_v3", "gaze": {"x": 1.0}}}}},
            "b": {"steps": {"wave": {"attention": {"clip": {"name": "wave", "when": "bound"}}}}},
            "c": {"steps": [{"attention": {"clip": {"when": "bound"}}}, {"attention": {"point": "target"}}]},  # no name: no reference
        },
        "desired_velocity": {"mean": 1.2, "clip_low": 0.5},  # clip_low is a parameter bound, not a clip
    }
    assert sorted(referenced_clips(config)) == ["beckon_v3", "wave"]


def test_referenced_clips_adds_interaction_defaults_unless_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Animation, "_interaction_clip", {"HUG": "hug", "QUEUE_USE": None}.get)
    steps = [
        {"interaction": "HUG"},
        {"interaction": "HUG", "attention": {"clip": "hug_v7_air_w"}},
        {"interaction": "QUEUE_USE"},
    ]
    assert list(referenced_clips(steps)) == ["hug", "hug_v7_air_w"]


def _scenario(tmp_path: Path, agent: dict) -> ScenarioView:
    scenario_dir = tmp_path / "sc"
    scenario_dir.mkdir()
    dynamic = [{"name": "caller", "model": "Hospital/nurse_female_caucasian_young", "pose": [0.0, 0.0, 0.0], "agent": agent, "waypoints": [[0.0, 0.0]]}]
    (scenario_dir / "scenario.yaml").write_text(yaml.safe_dump({"static": [], "dynamic": dynamic}))
    return ScenarioView(scenario_dir)


def test_scenario_identifiers_include_agent_file_clips(tmp_path: Path) -> None:
    view = _scenario(tmp_path, {"agent_type": "./caller.yaml"})
    (view.path / "caller.yaml").write_text(yaml.safe_dump({"extends": "adult", "sequences": {"s": {"steps": {"call": {"attention": {"clip": "beckon_v3"}}}}}}))
    clips = [i.shortname for i in view.identifiers() if isinstance(i, AnimationIdentifier)]
    assert clips == ["Common/Animation/beckon_v3"]


def test_scenario_identifiers_include_inline_agent_clips_and_skip_builtin_types(tmp_path: Path) -> None:
    view = _scenario(tmp_path, {"agent_type": "adult", "attention": {"clip": "wave"}})
    assert [i.name for i in view.identifiers() if isinstance(i, AnimationIdentifier)] == ["wave"]


def test_scenario_identifiers_raise_on_a_missing_agent_file(tmp_path: Path) -> None:
    """A strict publish check must not read an unreadable agent file as one that plays no clips."""
    view = _scenario(tmp_path, {"agent_type": "./missing.yaml"})
    with pytest.raises(FileNotFoundError):
        list(view.identifiers())
