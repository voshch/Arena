"""Animation assets, offline: clip round trip, layout listing, and the clips a scenario's agents reference."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_simulation_setup.tree import NetResolver, SimplePathResolver
from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier, AnimationView, add_anim_group, from_start_heading, pack_anim, referenced_clips, write_anim, write_clip
from arena_simulation_setup.tree.World.Scenario import ScenarioView


def _frames(n: int = 4) -> list[dict]:
    return [{"t": 0.05 * k, "angles": {"l_elbow": 0.1 * k, "r_elbow": -0.1 * k}, "root_xy_yaw": [0.0, 0.0, 0.01 * k], "animation_state": 0} for k in range(n)]


def test_clip_round_trips_frames_and_meta(tmp_path: Path) -> None:
    frames = _frames()
    frames[0]["angles"].pop("r_elbow")
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
    assert not (directory / "plain.npz").exists()
    with np.load(directory / "plain.anim", allow_pickle=False) as z:
        assert set(z.files) == {"anim/requires", "anim/settings", "clip/joint_names", "clip/angles", "clip/t", "clip/root_xy_yaw", "clip/animation_state"}


def _quat(axis: tuple[float, float, float], angle: float) -> list[float]:
    s = np.sin(angle / 2.0)
    return [axis[0] * s, axis[1] * s, axis[2] * s, float(np.cos(angle / 2.0))]


def test_tilted_clip_round_trips_its_pelvis_orientation(tmp_path: Path) -> None:
    frames = _frames()
    for k, frame in enumerate(frames):
        frame["root_xy_yaw"] = [0.0, 0.0, 0.5 * k]
        frame["root_quat"] = _quat((0.0, 1.0, 0.0), -0.4 * k)
    directory = write_clip(tmp_path / "Animation" / "fall", frames, {"fps": 20.0})
    view = AnimationIdentifier("fall").load(directory)
    assert view.meta["pelvis_tilt"] is True
    back = view.clip.frames()
    for frame, loaded in zip(frames, back, strict=True):
        assert abs(np.dot(frame["root_quat"], loaded["root_quat"])) == pytest.approx(1.0)
    with np.load(directory / "fall.anim") as z:  # the file keeps each frame's own heading frame, as the retarget writes it
        assert z["root/quat"][0] == pytest.approx(frames[0]["root_quat"])
        # frame 2: the pitch seen from its heading, 1 rad left of the first
        assert body_up(z["root/quat"][2]) == pytest.approx([np.cos(-1.0) * np.sin(-0.8), np.sin(-1.0) * np.sin(-0.8), np.cos(-0.8)])


def body_up(q: np.ndarray) -> np.ndarray:
    """The body z axis after the xyzw rotation q."""
    x, y, z, w = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def test_pelvis_orientation_is_turned_from_each_heading_to_the_first() -> None:
    lying = np.array([_quat((0.0, 1.0, 0.0), -np.pi / 2)] * 3)  # on its back, the head behind the heading
    turned = from_start_heading(lying, np.array([0.3, 0.3 + np.pi, 0.3 + np.pi / 2]))
    assert body_up(turned[0]) == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)
    assert body_up(turned[1]) == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert body_up(turned[2]) == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


def test_root_quat_without_the_pelvis_tilt_annotation_is_not_played(tmp_path: Path) -> None:
    directory = write_clip(tmp_path / "Animation" / "folded", _frames(), {})
    add_anim_group(directory / "folded.anim", "root", {"quat": np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))})
    clip = AnimationIdentifier("folded").load(directory).clip
    assert clip.root_quat is None and "root_quat" not in clip.frames()[0]


def _npz_clip(directory: Path, annotation: dict | None = None, n: int = 4) -> Path:
    directory.mkdir(parents=True)
    np.savez(directory / f"{directory.name}.npz", joint_names=np.array(["l_elbow"]), angles=np.full((n, 1), 0.5), t=0.05 * np.arange(n), root_xy_yaw=np.zeros((n, 3)), animation_state=np.zeros(n, dtype=np.int16))
    (directory / "annotation.yaml").write_text(yaml.safe_dump({"name": directory.name, "fps": 20.0, "loop": True, **(annotation or {})}))
    return directory


def _anim_arrays(n: int = 4, value: float = 0.25) -> dict[str, np.ndarray]:
    return {"clip/joint_names": np.array(["l_elbow"]), "clip/angles": np.full((n, 1), value), "clip/t": 0.05 * np.arange(n), "clip/root_xy_yaw": np.zeros((n, 3)), "clip/animation_state": np.zeros(n, dtype=np.int16)}


def test_version_1_anim_stays_readable(tmp_path: Path) -> None:
    directory = _npz_clip(tmp_path / "frozen")
    with (directory / "frozen.anim").open("wb") as file:
        np.savez(file, **{"anim/requires": np.array(1), "anim/settings": np.array('{"loop_from": 0.05}'), **_anim_arrays()})
    view = AnimationView(directory)
    assert view.meta["loop_from"] == 0.05 and view.meta["loop"] is True
    assert view.clip.angles[0, 0] == 0.25 and not view.clip.legacy


def test_annotation_anim_block_tunes_over_the_file_settings(tmp_path: Path) -> None:
    directory = _npz_clip(tmp_path / "tuned", {"loop_from": 0.0, "anim": {"loop_from": 0.1}})
    write_anim(directory / "tuned.anim", _anim_arrays(), {"loop_from": 0.05, "release_from": 0.15})
    meta = AnimationView(directory).meta
    assert meta["loop_from"] == 0.1 and meta["release_from"] == 0.15


def test_annotation_anim_block_is_ignored_without_an_anim(tmp_path: Path) -> None:
    directory = _npz_clip(tmp_path / "legacy", {"loop_from": 0.0, "anim": {"loop_from": 0.1}})
    assert AnimationView(directory).meta["loop_from"] == 0.0


def test_anim_reader_ignores_groups_it_does_not_know(tmp_path: Path) -> None:
    directory = _npz_clip(tmp_path / "later")
    write_anim(directory / "later.anim", {**_anim_arrays(), "clip.2/angles": np.zeros((9, 1)), "mesh/verts": np.zeros((3, 3))})
    assert AnimationView(directory).clip.angles.shape == (4, 1)


def test_anim_needing_a_newer_reader_falls_back_to_the_npz(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    directory = _npz_clip(tmp_path / "newer")
    write_anim(directory / "newer.anim", _anim_arrays(), {"loop": False}, requires=2)
    view = AnimationView(directory)
    with caplog.at_level(logging.WARNING):
        clip = view.clip
    assert clip.angles[0, 0] == 0.5 and clip.legacy and view.meta["loop"] is True
    assert "needs reader 2" in caplog.text


def test_npz_only_clip_is_marked_legacy(tmp_path: Path) -> None:
    assert AnimationView(_npz_clip(tmp_path / "old")).clip.legacy


def test_old_reader_keeps_its_files_when_an_anim_is_added(tmp_path: Path) -> None:
    directory = _npz_clip(tmp_path / "wave")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    write_anim(directory / "wave.anim", _anim_arrays(), {"loop_from": 0.05})
    assert {name: (directory / name).read_bytes() for name in before} == before
    assert AnimationView(directory).clip.angles[0, 0] == 0.25


def test_pack_anim_keeps_the_motion_and_takes_the_table_as_a_group(tmp_path: Path) -> None:
    source = _npz_clip(tmp_path / "src" / "point")
    np.savez(source / "table.npz", az=np.arange(3.0), template=np.array("point"))
    target = tmp_path / "out" / "point"
    pack_anim(source, target, {"loop_from": 0.05})
    assert sorted(path.name for path in target.iterdir()) == ["point.anim"]
    with np.load(target / "point.anim", allow_pickle=False) as z:
        assert z["table/az"].tolist() == [0.0, 1.0, 2.0] and str(z["table/template"]) == "point"
        assert z["clip/angles"][0, 0] == 0.5 and "anim/settings" in z.files


def test_add_anim_group_replaces_only_that_group(tmp_path: Path) -> None:
    path = write_anim(tmp_path / "x.anim", {**_anim_arrays(), "table/az": np.zeros(2)}, {"loop": True})
    add_anim_group(path, "table", {"el": np.ones(3)})
    with np.load(path, allow_pickle=False) as z:
        assert "table/az" not in z.files and z["table/el"].tolist() == [1.0, 1.0, 1.0]
        assert "clip/angles" in z.files and "anim/settings" in z.files


def test_bucket_fetches_only_anim_and_annotation_files() -> None:
    nets = [resolver for resolver in AnimationIdentifier._resolvers if isinstance(resolver, NetResolver)]
    assert nets and all(resolver._formats == ["anim", "yaml"] for resolver in nets)


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
            "c": {"steps": [{"attention": {"clip": {"when": "bound"}}}, {"attention": {"point": "target"}}]},
        },
        "desired_velocity": {"mean": 1.2, "clip_low": 0.5},
    }
    assert sorted(referenced_clips(config)) == ["beckon_v3", "wave"]


def test_referenced_clips_adds_interaction_defaults_unless_overridden() -> None:
    kinds = pytest.importorskip("arena_humansim.core.interaction_kinds")
    hug = kinds.InteractionType.HUG.kind.clip
    assert hug and not kinds.InteractionType.QUEUE_USE.kind.clip
    steps = [
        {"interaction": "HUG"},
        {"interaction": "HUG", "attention": {"clip": "hug_custom"}},
        {"interaction": "QUEUE_USE"},
    ]
    assert list(referenced_clips(steps)) == [hug, "hug_custom"]


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
    view = _scenario(tmp_path, {"agent_type": "./missing.yaml"})
    with pytest.raises(FileNotFoundError):
        list(view.identifiers())
