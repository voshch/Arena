"""`arena cam serve` against a stand-in runtime: play, replace mid-take, stop, refusals and riding a lockstep run."""

from __future__ import annotations

import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import rclpy
import yaml
from arena_cam.record import PARTIAL
from arena_cam_msgs.srv import Play
from standin import Runtime
from std_srvs.srv import Trigger

FPS = 10
PERIOD_NS = 1_000_000_000 // FPS

LONG = {"reference": {"entity": "first_link"}, "timeline": [{"hold": {"duration": 60.0}}]}
NEXT = {"reference": {"entity": "second_link"}, "timeline": [{"hold": {"duration": 60.0}}]}
SHORT = {"timeline": [{"hold": {"duration": 0.2}}]}
SHORT_RIG = {"cameras": {"a": {"timeline": [{"hold": {"duration": 0.3}}]}, "b": {"timeline": [{"hold": {"duration": 0.3}}]}}}
LONG_RIG = {"cameras": {"a": {"timeline": [{"hold": {"duration": 60.0}}]}, "b": {"timeline": [{"hold": {"duration": 60.0}}]}}}
RIG = {"live": "b", "cameras": {"a": {"reference": {"entity": "a_link"}, "timeline": [{"hold": {"duration": 60.0}}]}, "b": {"reference": {"entity": "b_link"}, "timeline": [{"hold": {"duration": 60.0}}]}}}


@pytest.fixture(autouse=True)
def _rclpy():
    rclpy.init()
    yield
    rclpy.shutdown()


class Server:
    """The server process and the service clients a task generator would use."""

    def __init__(self, runtime: Runtime, record: Path | None = None) -> None:
        self._record = record
        self._node = runtime.node
        self.play_client = self._node.create_client(Play, "/arena/env_0/cam/play")
        self.stop_client = self._node.create_client(Trigger, "/arena/env_0/cam/stop")

    def __enter__(self) -> Server:
        record = [] if self._record is None else [f"--record={self._record}"]
        self._process = subprocess.Popen([sys.executable, "-m", "arena_cam.cli", "serve", "--env", "0", "--sim", "--fps", str(FPS), *record])
        assert self.play_client.wait_for_service(timeout_sec=60.0)
        return self

    def __exit__(self, *_exc: object) -> None:
        try:
            self._process.send_signal(signal.SIGINT)
            self._process.wait(timeout=30)
        finally:
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait()

    def _call(self, client: object, request: object) -> object:
        future = client.call_async(request)
        deadline = time.monotonic() + 15.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert future.done()
        return future.result()

    def play(self, root: str, bundle: dict, params: str = "", take: str = "") -> Play.Response:
        return self._call(self.play_client, Play.Request(root=root, bundle=yaml.safe_dump(bundle, sort_keys=False), params=params, take=take))

    def stop(self) -> Trigger.Response:
        return self._call(self.stop_client, Trigger.Request())


def test_a_new_play_replaces_the_running_take() -> None:
    with Runtime() as runtime, Server(runtime) as server:
        assert server.play("long", {"long": LONG}).success
        assert runtime.wait(lambda: "first_link" in runtime.references and runtime.keyframes)
        assert server.play("next", {"next": NEXT}).success
        assert runtime.wait(lambda: "second_link" in runtime.references)
        streamed = len(runtime.keyframes)
        assert runtime.wait(lambda: len(runtime.keyframes) > streamed)


def test_each_take_starts_in_the_world_frame() -> None:
    with Runtime() as runtime, Server(runtime) as server:
        assert server.play("long", {"long": LONG}).success
        assert runtime.wait(lambda: "first_link" in runtime.references)
        assert server.play("next", {"next": NEXT}).success
        assert runtime.wait(lambda: "second_link" in runtime.references)
    assert runtime.references == ["world", "first_link", "world", "second_link"]


def test_stop_ends_the_take_and_the_server_stays_up() -> None:
    with Runtime() as runtime, Server(runtime) as server:
        assert server.play("long", {"long": LONG}).success
        assert runtime.wait(lambda: len(runtime.keyframes) > 3)
        assert server.stop().success
        assert runtime.stable(lambda: len(runtime.keyframes))
        streamed = len(runtime.keyframes)
        assert runtime.stable(lambda: len(runtime.keyframes))
        assert len(runtime.keyframes) == streamed
        assert server.play("next", {"next": NEXT}).success
        assert runtime.wait(lambda: "second_link" in runtime.references)


def test_a_finished_take_leaves_the_server_ready() -> None:
    with Runtime() as runtime, Server(runtime) as server:
        assert server.play("short", {"short": SHORT}).success
        assert runtime.wait(lambda: len(runtime.keyframes) > 3)
        assert runtime.stable(lambda: len(runtime.keyframes))
        assert server.play("next", {"next": NEXT}).success
        assert runtime.wait(lambda: "second_link" in runtime.references)


@pytest.mark.parametrize(
    ("root", "bundle", "message"),
    [
        ("ghost", {"long": LONG}, "ghost"),
        ("orbit", {"orbit": LONG}, "shadow a camera verb"),
        ("bad", {"bad": {"timeline": [{"nope": {}}]}}, "unknown camera name"),
        ("bad", {"bad": {"cameras": {"a": {"mount": {"frame": "x"}}}}}, "needs a camera with a timeline"),
    ],
)
def test_a_bad_bundle_is_refused_and_the_take_keeps_running(root: str, bundle: dict, message: str) -> None:
    with Runtime() as runtime, Server(runtime) as server:
        assert server.play("long", {"long": LONG}).success
        assert runtime.wait(lambda: len(runtime.keyframes) > 3)
        response = server.play(root, bundle)
        assert not response.success
        assert message in response.message
        streamed = len(runtime.keyframes)
        assert runtime.wait(lambda: len(runtime.keyframes) > streamed)


def test_a_rig_plays_its_live_camera() -> None:
    with Runtime() as runtime, Server(runtime) as server:
        assert server.play("rig", {"rig": RIG}).success
        assert runtime.wait(lambda: "b_link" in runtime.references)
        assert "a_link" not in runtime.references


def test_a_take_rides_an_active_run_under_its_env_and_never_holds() -> None:
    with Runtime(run_period_ns=PERIOD_NS) as runtime, Server(runtime) as server:
        assert server.play("long", {"long": LONG}).success
        assert runtime.wait(lambda: len(runtime.beats) > 5)
        assert ("cam", "/arena/env_0") in runtime.channels
        assert runtime.holds == []
        assert runtime.steps == 0
        beats = list(runtime.beats)
        assert [b - a for a, b in zip(beats, beats[1:], strict=False)] == [PERIOD_NS] * (len(beats) - 1)


def test_a_replaced_take_unregisters_its_channel_first() -> None:
    with Runtime(run_period_ns=PERIOD_NS) as runtime, Server(runtime) as server:
        assert server.play("long", {"long": LONG}).success
        assert runtime.wait(lambda: len(runtime.beats) > 3)
        assert server.play("next", {"next": NEXT}).success
        assert runtime.wait(lambda: runtime.channels.count(("cam", "/arena/env_0")) == 2)
    assert runtime.channels[:3] == [("cam", "/arena/env_0"), ("", "/arena/env_0"), ("cam", "/arena/env_0")]


def test_with_no_run_a_take_plays_on_the_wall_clock_without_a_channel() -> None:
    with Runtime() as runtime, Server(runtime) as server:
        assert server.play("long", {"long": LONG}).success
        assert runtime.wait(lambda: len(runtime.keyframes) > 3)
        assert runtime.channels == []
        assert runtime.holds == []


def test_a_recording_server_files_a_finished_rig_take_under_its_label(tmp_path: Path) -> None:
    with Runtime(run_period_ns=PERIOD_NS) as runtime, Server(runtime, record=tmp_path) as server:
        assert server.play("rig", {"rig": SHORT_RIG}, take="w/s/ep001").success
        assert runtime.wait(lambda: (tmp_path / "w/s/ep001-a.mp4").is_file() and (tmp_path / "w/s/ep001-b.mp4").is_file())
        assert not list(tmp_path.rglob(f"*{PARTIAL}*"))


def test_a_take_replaced_mid_rig_keeps_only_partial_files(tmp_path: Path) -> None:
    with Runtime(run_period_ns=PERIOD_NS) as runtime, Server(runtime, record=tmp_path) as server:
        assert server.play("rig", {"rig": LONG_RIG}, take="w/ep001").success
        assert runtime.wait(lambda: len(runtime.captures) > 4)
        assert server.play("rig", {"rig": LONG_RIG}, take="w/ep002").success
        assert runtime.wait(lambda: (tmp_path / f"w/ep001{PARTIAL}-a.mp4").is_file() and (tmp_path / f"w/ep002{PARTIAL}-a.mp4").is_file())
    assert not (tmp_path / "w/ep001-a.mp4").exists()


def test_without_a_run_the_server_plays_and_records_nothing(tmp_path: Path) -> None:
    with Runtime() as runtime, Server(runtime, record=tmp_path) as server:
        assert server.play("rig", {"rig": SHORT_RIG}, take="w/ep001").success
        assert runtime.wait(lambda: len(runtime.keyframes) > 0)
        time.sleep(1.0)
    assert not list(tmp_path.rglob("*.mp4"))


def test_a_finished_take_is_never_recorded_over(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    (tmp_path / "w/ep001-a.mp4").write_bytes(b"old")
    with Runtime(run_period_ns=PERIOD_NS) as runtime, Server(runtime, record=tmp_path) as server:
        assert server.play("rig", {"rig": SHORT_RIG}, take="w/ep001").success
        assert runtime.wait(lambda: len(runtime.beats) > 0)
        time.sleep(1.0)
    assert (tmp_path / "w/ep001-a.mp4").read_bytes() == b"old"
    assert not list(tmp_path.rglob(f"*{PARTIAL}*"))


def test_a_play_without_a_take_records_nothing(tmp_path: Path) -> None:
    with Runtime(run_period_ns=PERIOD_NS) as runtime, Server(runtime, record=tmp_path) as server:
        assert server.play("rig", {"rig": SHORT_RIG}).success
        assert runtime.wait(lambda: len(runtime.beats) > 0)
        time.sleep(1.0)
    assert not list(tmp_path.rglob("*.mp4"))


def test_partial_files_a_cut_take_left_are_recorded_over(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    for view in ("a", "b"):
        (tmp_path / f"w/ep001{PARTIAL}-{view}.mp4").write_bytes(b"stale")
    with Runtime(run_period_ns=PERIOD_NS) as runtime, Server(runtime, record=tmp_path) as server:
        assert server.play("rig", {"rig": SHORT_RIG}, take="w/ep001").success
        assert runtime.wait(lambda: (tmp_path / "w/ep001-a.mp4").is_file() and (tmp_path / "w/ep001-b.mp4").is_file())
    assert (tmp_path / "w/ep001-a.mp4").read_bytes() != b"stale"
    assert not list(tmp_path.rglob(f"*{PARTIAL}*"))
