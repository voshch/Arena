from __future__ import annotations

import math

import pytest

from arena_cam import faces

_ROSTERS = {
    "/arena/env_0/humans/faces/tracked": ["env_0_agent_1", "env_0_agent_2"],
    "/arena/env_1/humans/faces/tracked": ["env_1_agent_2", "env_1_agent_3"],
}


def test_exact_face_id_resolves_to_its_env():
    assert faces.match(_ROSTERS, "env_1_agent_3") == ("/arena/env_1/humans/faces/tracked", "env_1_agent_3")


def test_face_frame_name_resolves_like_its_id():
    assert faces.match(_ROSTERS, "face_env_0_agent_2") == ("/arena/env_0/humans/faces/tracked", "env_0_agent_2")


def test_bare_agent_number_resolves_when_unique():
    assert faces.match(_ROSTERS, "1") == ("/arena/env_0/humans/faces/tracked", "env_0_agent_1")


def test_bare_agent_number_in_two_envs_is_ambiguous():
    with pytest.raises(LookupError, match="ambiguous"):
        faces.match(_ROSTERS, "2")


def test_unknown_face_lists_the_tracked_ones():
    with pytest.raises(LookupError, match="env_1_agent_3"):
        faces.match(_ROSTERS, "env_0_agent_9")


def test_agent_number_does_not_match_a_longer_number():
    with pytest.raises(LookupError):
        faces.match({"/humans/faces/tracked": ["env_agent_12"]}, "2")


def test_view_topic_sits_under_the_roster_namespace():
    assert faces.view_topic("/arena/env_0/humans/faces/tracked", "env_0_agent_1") == "/arena/env_0/humans/faces/env_0_agent_1/view"


def test_tracked_topics_ignores_other_humans_topics():
    topics = ["/arena/env_0/humans/faces/tracked", "/arena/env_0/humans/bodies/tracked", "/arena/env_0/humans/tf"]
    assert faces.tracked_topics(topics) == ["/arena/env_0/humans/faces/tracked"]


def test_face_id_only_reads_face_frames():
    assert faces.face_id("face_env_0_agent_1") == "env_0_agent_1"
    assert faces.face_id("env_0/jackal/base_link") is None


def test_fov_inverts_a_square_pinhole():
    msgs = pytest.importorskip("arena_people_msgs.msg")
    sensor_msgs = pytest.importorskip("sensor_msgs.msg")
    f = 256.0 / math.tan(math.radians(30.0))
    info = sensor_msgs.CameraInfo(width=512, height=512, k=[f, 0.0, 256.0, 0.0, f, 256.0, 0.0, 0.0, 1.0])
    assert faces.fov(msgs.FaceView(info=info, clip_near=0.15)) == pytest.approx(math.radians(60.0))


def test_pov_takes_face_duration_and_fov():
    pytest.importorskip("arena_cam.camera")
    from arena_cam.shots import resolve

    (action,) = resolve("pov", {"face": 3, "duration": 4, "fov": 0.9})
    assert (action.face, action.duration, action.fov) == ("3", 4.0, 0.9)


def test_pov_without_a_face_names_the_missing_param():
    pytest.importorskip("arena_cam.camera")
    from arena_cam.shots import resolve

    with pytest.raises(ValueError, match="'face'"):
        resolve("pov", {"duration": 4})


def test_pov_reads_a_cli_number_as_an_agent_number():
    pytest.importorskip("arena_cam.camera")
    from arena_cam.shots import resolve

    (action,) = resolve("pov", {"face": 1.0})
    assert action.face == "1"
