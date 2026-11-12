from __future__ import annotations

from arena_bringup.viz_backends import viz_commands

NS = "/arena/env_{}/arena_env_node"


def _commands(env: int, **viz_args: str) -> dict[str, list[str]]:
    return dict(viz_commands(NS.format(env), env, viz_args))


def test_viz_cam_adds_the_envs_camera_server_with_the_request() -> None:
    commands = _commands(2, cam="wide")
    assert commands["cam_/arena/env_2/arena_env_node"] == ["ros2", "run", "arena_cam", "cam", "serve", "--env", "2", "--setup=wide"]


def test_env_0_also_drives_the_sim_camera() -> None:
    assert _commands(0, cam="0")["cam_/arena/env_0/arena_env_node"][-2:] == ["--setup=0", "--sim"]


def test_the_cam_key_stays_off_the_viz_backend() -> None:
    commands = _commands(1, cam="wide", view="robot")
    rviz = commands["rviz_/arena/env_1/arena_env_node"]
    assert "view:=robot" in rviz
    assert not any(arg.startswith("cam:=") for arg in rviz)


def test_no_cam_key_starts_no_server() -> None:
    assert set(_commands(0)) == {"rviz_/arena/env_0/arena_env_node"}


def test_viz_cam_record_reaches_the_server_as_its_record_dir() -> None:
    commands = _commands(2, **{"cam": "three_view", "cam.record": "mvp"})
    assert commands["cam_/arena/env_2/arena_env_node"][-2:] == ["--setup=three_view", "--record=mvp"]


def test_env_0_records_through_the_sim_camera() -> None:
    assert _commands(0, **{"cam": "0", "cam.record": "mvp"})["cam_/arena/env_0/arena_env_node"][-3:] == ["--setup=0", "--record=mvp", "--sim"]


def test_the_record_key_stays_off_the_viz_backend() -> None:
    rviz = _commands(1, **{"cam": "wide", "cam.record": "mvp"})["rviz_/arena/env_1/arena_env_node"]
    assert not any(arg.startswith("cam.record:=") for arg in rviz)


def test_a_record_dir_without_cam_starts_no_server() -> None:
    assert set(_commands(0, **{"cam.record": "mvp"})) == {"rviz_/arena/env_0/arena_env_node"}
