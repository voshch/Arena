"""The task generator's camera play against a stand-in camera server, resolving through real scenario tiers."""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest
import yaml

pytest.importorskip("arena_cam_msgs.srv")

from arena_runtime.env.episode_node import EpisodeNode, take_label
from arena_simulation_setup.tree.assets import Camera
from arena_simulation_setup.tree.World import point_scenario_assets


class Client:
    """The service client of `cam/play`: records what it is sent and answers with `response`."""

    def __init__(self, *, ready: bool = True, refuse: str = "") -> None:
        self._ready = ready
        self._refuse = refuse
        self.sent: list = []
        self.client = self
        self.srv_name = "/arena/env_0/cam/play"

    def service_is_ready(self) -> bool:
        return self._ready

    async def call_timeout(self, request: object) -> object:
        from arena_cam_msgs.srv import Play

        self.sent.append(request)
        return Play.Response(success=not self._refuse, message=self._refuse)


class Generator:
    """What `_play_camera` reads off the node."""

    _play_camera = EpisodeNode._play_camera

    def __init__(self, client: Client) -> None:
        self._cam_client = client
        self._cam_absent = False
        self._episodes = types.SimpleNamespace(current=types.SimpleNamespace(world="corridor", episode_id=7))
        self.warnings: list[str] = []

    def get_logger(self) -> Generator:
        return self

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _scenario(directory: Path, **specs: dict) -> Path:
    (directory / "cameras").mkdir(parents=True)
    for name, spec in specs.items():
        (directory / "cameras" / f"{name}.yaml").write_text(yaml.safe_dump(spec))
    return directory


def _play(generator: Generator, request: str, params: str = "", *, episode_take: bool = False) -> tuple[bool, str]:
    return asyncio.run(EpisodeNode._play_camera(generator, request, params, episode_take=episode_take))


@pytest.fixture(autouse=True)
def _scenario_tier(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(Camera, "_NETWORK", [])
    monkeypatch.setattr(Camera, "_LISTED", {})
    yield
    point_scenario_assets(None)


def test_the_scenario_asset_and_what_it_reaches_are_sent(tmp_path: Path) -> None:
    wide = {"timeline": [{"part": {"slice": [0, 1]}}, {"hold": {"duration": 1.0}}]}
    point_scenario_assets(_scenario(tmp_path / "scenario", wide=wide, part={"timeline": [{"hold": {"duration": 2.0}}]}))
    generator = Generator(Client())
    assert _play(generator, "wide", "{radius: 3}") == (True, "wide")
    (play,) = generator._cam_client.sent
    assert play.root == "wide"
    assert sorted(yaml.safe_load(play.bundle)) == ["part", "wide"]
    assert play.params == "{radius: 3}"


def test_a_scenario_swap_sends_the_new_scenarios_asset(tmp_path: Path) -> None:
    first = _scenario(tmp_path / "first", wide={"params": {"tag": "first"}, "timeline": []})
    second = _scenario(tmp_path / "second", wide={"params": {"tag": "second"}, "timeline": []})
    generator = Generator(Client())
    for scenario in (first, second):
        point_scenario_assets(scenario)
        _play(generator, "wide")
    assert [yaml.safe_load(play.bundle)["wide"]["params"]["tag"] for play in generator._cam_client.sent] == ["first", "second"]


def test_a_rigs_camera_order_survives_the_wire(tmp_path: Path) -> None:
    (tmp_path / "scenario" / "cameras").mkdir(parents=True)
    (tmp_path / "scenario" / "cameras" / "rig.yaml").write_text("cameras:\n  zebra: {timeline: []}\n  apple: {timeline: []}\n")
    point_scenario_assets(tmp_path / "scenario")
    generator = Generator(Client())
    _play(generator, "rig")
    assert list(yaml.safe_load(generator._cam_client.sent[0].bundle)["rig"]["cameras"]) == ["zebra", "apple"]


def test_a_name_a_swapped_scenario_lacks_falls_back_to_its_first(tmp_path: Path) -> None:
    point_scenario_assets(_scenario(tmp_path / "scenario", front={"timeline": []}))
    generator = Generator(Client())
    assert _play(generator, "wide") == (True, "front")


def test_an_index_picks_from_the_scenario(tmp_path: Path) -> None:
    point_scenario_assets(_scenario(tmp_path / "scenario", a={"timeline": []}, b={"timeline": []}))
    generator = Generator(Client())
    assert _play(generator, "1") == (True, "b")


def test_a_missing_server_is_logged_once(tmp_path: Path) -> None:
    point_scenario_assets(_scenario(tmp_path / "scenario", wide={"timeline": []}))
    generator = Generator(Client(ready=False))
    assert _play(generator, "wide") == (False, "no camera server")
    assert _play(generator, "wide") == (False, "no camera server")
    assert len(generator.warnings) == 1
    assert generator._cam_client.sent == []


def test_a_play_between_episodes_resolves_nothing(tmp_path: Path) -> None:
    from arena_cam_msgs.srv import Play

    point_scenario_assets(_scenario(tmp_path / "scenario", wide={"timeline": []}))
    generator = Generator(Client())
    generator._episodes = types.SimpleNamespace(current=types.SimpleNamespace(outcome_state=0))
    response = asyncio.run(EpisodeNode._cb_cam_play(generator, Play.Request(root="wide"), Play.Response()))
    assert not response.success
    assert "next episode start" in response.message
    assert generator._cam_client.sent == []


def test_a_refused_play_comes_back_with_its_message(tmp_path: Path) -> None:
    point_scenario_assets(_scenario(tmp_path / "scenario", wide={"timeline": []}))
    assert _play(Generator(Client(refuse="bad shot")), "wide") == (False, "bad shot")


def test_the_episode_start_play_carries_world_scenario_and_episode_as_its_take(tmp_path: Path) -> None:
    point_scenario_assets(_scenario(tmp_path / "oncoming", wide={"timeline": []}))
    generator = Generator(Client())
    _play(generator, "wide", episode_take=True)
    assert generator._cam_client.sent[0].take == "corridor/oncoming/ep007"


def test_a_play_forwarded_during_an_episode_carries_no_take(tmp_path: Path) -> None:
    from arena_cam_msgs.srv import Play
    from arena_runtime_msgs.action import RunEpisode

    point_scenario_assets(_scenario(tmp_path / "oncoming", wide={"timeline": []}))
    generator = Generator(Client())
    generator._episodes.current.outcome_state = RunEpisode.Result.RUNNING
    response = asyncio.run(EpisodeNode._cb_cam_play(generator, Play.Request(root="wide"), Play.Response()))
    assert response.success
    assert generator._cam_client.sent[0].take == ""


def test_a_take_without_a_scenario_skips_its_segment() -> None:
    assert take_label("corridor", "", 12) == "corridor/ep012"


def test_a_take_numbers_past_999_episodes() -> None:
    assert take_label("w", "s", 1234) == "w/s/ep1234"
