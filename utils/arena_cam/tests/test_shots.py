"""Expansion of bundles, slices and rigs against real verbs, with a take that records what the verbs ask for."""

from __future__ import annotations

import asyncio

import pytest

camera = pytest.importorskip("arena_cam.camera")

from arena_cam.mounts import Mount
from arena_cam.shots import check_bundle, expand_rig, resolve, substitute

HOLD = {"timeline": [{"hold": {"duration": 4.0}}]}
TWO_HOLDS = {"timeline": [{"hold": {"duration": 2.0}}, {"look": {"eye": [1, 0, 0], "target": [0, 0, 0]}}, {"hold": {"duration": 2.0}}]}


class Take:
    """The verb surface a timeline drives, recording each call instead of moving a camera."""

    def __init__(self) -> None:
        self.drives: list[tuple[float, tuple[float, float]]] = []
        self.looks = 0
        self.references: list[str] = []

    def ok(self) -> bool:
        return True

    async def drive(self, duration: float, _world_orientation: bool, _frame_at: object, span: tuple[float, float]) -> None:
        self.drives.append((duration, span))

    async def look(self, _eye: object, _target: object, _fov: float) -> None:
        self.looks += 1

    async def set_reference(self, entity: str, _pose: object, _mode: str) -> None:
        self.references.append(entity)


def _play(action: object) -> Take:
    take = Take()
    asyncio.run(action.run(take, camera._Cursor()))
    return take


def test_a_nested_asset_expands_in_place() -> None:
    bundle = {"outer": {"timeline": [{"inner": {}}, {"hold": {"duration": 1.0}}]}, "inner": HOLD}
    assert [action.verb for action in resolve("outer", {}, (), bundle)] == ["hold", "hold"]


def test_an_asset_cycle_is_named() -> None:
    bundle = {"a": {"timeline": [{"b": {}}]}, "b": {"timeline": [{"a": {}}]}}
    with pytest.raises(ValueError, match="shot cycle: a -> b -> a"):
        resolve("a", {}, (), bundle)


def test_a_name_outside_the_bundle_lists_both_vocabularies() -> None:
    with pytest.raises(ValueError, match=r"unknown camera name: 'ghost' \(verbs: .*hold.*shots: \['a'\]"):
        resolve("ghost", {}, (), {"a": HOLD})


def test_a_bundle_name_that_shadows_a_verb_is_refused() -> None:
    with pytest.raises(ValueError, match="shadow a camera verb"):
        check_bundle({"orbit": HOLD})


def test_a_context_token_survives_substitution_until_the_caller_passes_it() -> None:
    assert substitute({"entity": "${robot}/${env}"}, {}) == {"entity": "${robot}/${env}"}
    assert substitute("${robot}", {"robot": "base_link"}) == "base_link"


def test_any_other_missing_parameter_stays_an_error() -> None:
    with pytest.raises(ValueError, match="radius"):
        substitute("${radius}", {})


def test_a_param_default_can_hold_a_context_token() -> None:
    bundle = {"a": {"params": {"subject": "${robot}"}, "timeline": [{"track": {"entity": "${subject}"}}]}}
    (track,) = resolve("a", {}, (), bundle)
    assert track.entity == "${robot}"


def test_a_slice_inside_one_segment_keeps_the_whole_segments_progress() -> None:
    (sliced,) = resolve("a", {"slice": [1.0, 3.0]}, (), {"a": HOLD})
    assert sliced.duration == pytest.approx(2.0)
    assert _play(sliced).drives == [(4.0, (0.25, 0.75))]


def test_a_slice_skips_what_ends_before_it_and_stops_at_its_end() -> None:
    (sliced,) = resolve("a", {"slice": [1.0, 3.0]}, (), {"a": TWO_HOLDS})
    take = _play(sliced)
    assert take.drives == [(2.0, (0.5, 1.0)), (2.0, (0.0, 0.5))]
    assert take.looks == 1


def test_a_look_before_the_slice_moves_the_cursor_without_a_frame() -> None:
    (sliced,) = resolve("a", {"slice": [3.0, 4.0]}, (), {"a": TWO_HOLDS})
    take = _play(sliced)
    assert take.looks == 0
    assert take.drives == [(2.0, (0.5, 1.0))]


def test_a_reference_before_the_slice_still_applies() -> None:
    bundle = {"a": {"timeline": [{"hold": {"duration": 2.0}}, {"track": {"entity": "base_link"}}, {"hold": {"duration": 2.0}}]}}
    (sliced,) = resolve("a", {"slice": [3.0, None]}, (), bundle)
    take = _play(sliced)
    assert take.references == ["base_link"]
    assert take.drives == [(2.0, (0.5, 1.0))]


def test_an_open_slice_and_an_overrunning_one_end_with_the_timeline() -> None:
    for bounds in ([2.0, None], [2.0, 90.0]):
        (sliced,) = resolve("a", {"slice": bounds}, (), {"a": HOLD})
        assert sliced.duration == pytest.approx(2.0)
        assert _play(sliced).drives == [(4.0, (0.5, 1.0))]


def test_slices_of_slices_compose() -> None:
    bundle = {"inner": HOLD, "outer": {"timeline": [{"inner": {"slice": [1.0, 3.0]}}]}}
    (sliced,) = resolve("outer", {"slice": [0.5, 1.5]}, (), bundle)
    assert _play(sliced).drives == [(4.0, pytest.approx((0.375, 0.625)))]


@pytest.mark.parametrize("bounds", [[3.0, 1.0], [-1.0, 2.0], [1.0], "2,3"])
def test_bad_slice_bounds_are_refused(bounds: object) -> None:
    with pytest.raises(ValueError, match="slice"):
        resolve("a", {"slice": bounds}, (), {"a": HOLD})


def test_slice_on_a_verb_is_refused() -> None:
    with pytest.raises(ValueError, match="slice applies to a step"):
        resolve("hold", {"duration": 1.0, "slice": [0.0, 1.0]})


def test_slice_as_a_param_default_is_refused() -> None:
    with pytest.raises(ValueError, match="slice is a step key"):
        resolve("a", {}, (), {"a": {"params": {"slice": [0, 1]}, "timeline": []}})


RIG = {
    "params": {"duration": 3.0},
    "projection": "perspective",
    "cameras": {
        "nose": {"mount": {"frame": "${robot}", "at": [0.2, 0.0, 0.6], "pitch": 0.1, "fov": 1.2}},
        "wide": {"timeline": [{"hold": {"duration": "${duration}"}}]},
        "chase": {"source": "sim", "reference": {"entity": "${robot}", "mode": "yaw"}, "timeline": [{"hold": {"duration": 1.0}}]},
    },
}


def test_a_rig_expands_each_camera_with_timelines_first() -> None:
    rig = expand_rig("r", RIG, {"duration": 9.0}, {"r": RIG})
    assert [camera.name for camera in rig.cameras] == ["wide", "chase", "nose"]
    assert rig.cameras[0].actions[0].duration == 9.0
    assert [action.verb for action in rig.cameras[1].actions] == ["track", "hold"]
    assert rig.cameras[1].source == "sim"
    assert rig.cameras[2].mount == Mount("${robot}", 1.2, (0.2, 0.0, 0.6), 0.1)
    assert rig.cameras[2].source == "sim"
    assert rig.projection == "perspective"


def test_live_defaults_to_the_first_timeline_camera() -> None:
    assert expand_rig("r", RIG, {}, {"r": RIG}).live == "wide"
    assert expand_rig("r", {**RIG, "live": "chase"}, {}, {"r": RIG}).live == "chase"


def _rig(**cameras: object) -> dict:
    return {"cameras": cameras}


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({**_rig(a={"timeline": []}), "timeline": []}, "exclude each other"),
        (_rig(a={"timeline": [], "fov": 1.0}), "unknown keys"),
        (_rig(a={"timeline": [], "source": "rig"}), "source is one of"),
        (_rig(a={"timeline": []}, b={"mount": {"frame": "x"}, "timeline": []}), "mount excludes"),
        (_rig(a={"timeline": []}, b={"mount": {"at": [0, 0, 0]}}), "mount takes"),
        (_rig(a={"mount": {"frame": "x"}}), "needs a camera with a timeline"),
        ({**_rig(a={"timeline": []}), "live": "ghost"}, "live names one of"),
        ({**_rig(a={"timeline": []}, b={"mount": {"frame": "x"}}), "live": "b"}, "live names one of"),
        (_rig(a={"timeline": [{"projection": "orthographic"}]}), "cannot run in a rig"),
        (_rig(a={"timeline": [{"latch": {}}]}), "cannot run in a rig"),
    ],
)
def test_rig_refusals(spec: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        expand_rig("r", spec, {}, {"r": spec})


def test_a_rig_inside_a_slice_refuses_a_latch_too() -> None:
    spec = _rig(a={"timeline": [{"sub": {"slice": [0.0, 1.0]}}]})
    bundle = {"r": spec, "sub": {"timeline": [{"latch": {}}, {"hold": {"duration": 2.0}}]}}
    with pytest.raises(ValueError, match="cannot run in a rig"):
        expand_rig("r", spec, {}, bundle)


def test_a_rig_is_a_root_only() -> None:
    bundle = {"outer": {"timeline": [{"r": {}}]}, "r": RIG}
    with pytest.raises(ValueError, match="root only"):
        resolve("outer", {}, (), bundle)


def _camera() -> camera.Camera:
    from arena_cam.surfaces import TargetSelection

    return camera.Camera(TargetSelection(include_sim=True, viz_all=False, viz_env=None))


def test_a_loaded_rig_films_every_camera_and_plays_the_live_one() -> None:
    cam = _camera().load("r", {"r": RIG})
    assert [rig.name for rig in cam.cameras(recording=True)] == ["wide", "chase", "nose"]
    assert [rig.name for rig in cam.cameras(recording=False)] == ["wide"]
    assert cam.projection == "perspective"


def test_a_lone_timeline_is_one_unnamed_camera() -> None:
    (lone,) = _camera().add("hold", duration=1.0).cameras(recording=True)
    assert lone.name == ""


def test_a_rig_cannot_follow_queued_steps() -> None:
    with pytest.raises(ValueError, match="root only"):
        _camera().add("hold", duration=1.0).load("r", {"r": RIG})


def test_nothing_can_follow_a_rig() -> None:
    with pytest.raises(ValueError, match="root only"):
        _camera().load("r", {"r": RIG}).add("hold", duration=1.0)


def test_latch_next_to_mounts_is_refused() -> None:
    cam = _camera().add("latch")
    with pytest.raises(ValueError, match="latch cannot run next to mounts"):
        cam.record("take", lockstep=True, mounts=[Mount("base_link", 1.0)])


def test_the_bundled_assets_expand_without_a_world() -> None:
    assert {"establishing", "overhead", "shoulder", "tour"} <= set(camera.global_cameras())
    for name in camera.global_cameras():
        assert _camera().add(name).cameras(recording=False)[0].actions
