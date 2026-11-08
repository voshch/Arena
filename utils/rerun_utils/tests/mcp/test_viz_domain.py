from __future__ import annotations

import asyncio
import os
import stat
import threading
import urllib.parse
import uuid
from collections.abc import Coroutine, Iterator
from pathlib import Path

import pytest

pytest.importorskip("arena_mcp")
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.msg")

from rerun_utils import mcp as viz
from rerun_utils import views

from arena_mcp.core import ROS, ToolError

STAND_IN = """#!/usr/bin/env python3
import signal, socket, sys, time
args = dict(a.split(":=", 1) for a in sys.argv[1:] if ":=" in a)
ports = [int(args["web_port"]), int(args["grpc_port"])]
socks = []
for port in ports:
    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen()
    socks.append(s)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    time.sleep(0.1)
"""


class Rig:
    def __init__(self, loop: asyncio.AbstractEventLoop, tg: str) -> None:
        self.loop = loop
        self.tg = tg

    def run[T](self, coro: Coroutine[object, object, T]) -> T:
        return self.loop.run_until_complete(coro)


@pytest.fixture(scope="module")
def rig():
    from arena_rclpy_mixins.qos import latched
    from arena_runtime_msgs.msg import EnvRecord, EnvRegistry
    from rclpy.executors import SingleThreadedExecutor

    owned = rclpy.ok()
    if not owned:
        rclpy.init()
    tg = f"/viz_{uuid.uuid4().hex[:8]}/env_0/arena_env_node"
    node = rclpy.create_node("arena")
    publisher = node.create_publisher(EnvRegistry, "/arena/state/envs", latched(1))
    publisher.publish(EnvRegistry(envs=[EnvRecord(env_id=0, fqn=tg, ready=True)]))
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(ROS.latest(EnvRegistry, "/arena/state/envs", timeout=30.0))
        yield Rig(loop, tg)
    finally:
        executor.shutdown()
        node.destroy_node()
        loop.run_until_complete(ROS.shutdown())
        loop.close()
        if rclpy.ok():
            rclpy.shutdown()
        if owned:
            rclpy.init()


@pytest.fixture(autouse=True)
def stand_in_ros2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ros2"
    script.write_text(STAND_IN)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "log"))
    yield
    for view in views.load_all():
        views.forget(view.env_id)
        try:
            os.killpg(view.pid, 9)
        except ProcessLookupError:
            pass


def test_viz_domain_is_served_from_its_entry_point():
    from arena_mcp.server import discover

    (loaded,) = discover(["viz"])
    assert loaded.served, loaded.reason
    assert {spec.name for spec in loaded.domain.tools} == {"viz_rerun_start", "viz_rerun_stop", "viz_rerun_list"}


def test_viewer_url_points_the_web_viewer_at_the_grpc_proxy():
    url = urllib.parse.urlsplit(views.viewer_url(9090, 9876))
    assert (url.hostname, url.port, url.path) == ("127.0.0.1", 9090, "/")
    assert urllib.parse.parse_qs(url.query) == {"url": ["rerun+http://127.0.0.1:9876/proxy"]}


def test_start_list_idempotent_start_and_stop_follow_one_view(rig: Rig):
    started = rig.run(viz.viz_rerun_start())
    assert started["started"] is True
    assert started["env_id"] == 0
    assert started["ns"] == rig.tg
    web, grpc = started["web_port"], started["grpc_port"]
    assert web != grpc
    assert views.accepts(web) and views.accepts(grpc)
    assert f"ns:={rig.tg}" in Path(f"/proc/{started['pid']}/cmdline").read_bytes().decode().split("\0")
    assert {f"web_port:={web}", f"grpc_port:={grpc}"} <= set(Path(f"/proc/{started['pid']}/cmdline").read_bytes().decode().split("\0"))
    assert started["url"] == views.viewer_url(web, grpc)

    assert viz.viz_rerun_list() == {"views": [{**{k: v for k, v in started.items() if k != "started"}, "alive": True}]}

    again = rig.run(viz.viz_rerun_start(env_id=0))
    assert again["started"] is False
    assert again["pid"] == started["pid"]

    assert rig.run(viz.viz_rerun_stop()) == {"env_id": 0, "stopped": True}
    assert viz.viz_rerun_list() == {"views": []}
    assert not views.accepts(web)
    assert rig.run(viz.viz_rerun_stop(env_id=0)) == {"env_id": 0, "stopped": False}


def test_start_honors_requested_ports(rig: Rig):
    web, grpc = views.free_port(), views.free_port()
    started = rig.run(viz.viz_rerun_start(web_port=web, grpc_port=grpc))
    assert (started["web_port"], started["grpc_port"]) == (web, grpc)
    rig.run(viz.viz_rerun_stop())


def test_list_prunes_a_view_whose_process_died(rig: Rig):
    started = rig.run(viz.viz_rerun_start())
    os.killpg(started["pid"], 9)
    for _ in range(50):
        if not views.load_all():
            break
        rig.run(asyncio.sleep(0.1))
    assert viz.viz_rerun_list() == {"views": []}
    assert not (views.state_dir() / "env_0.json").exists()


def test_start_names_the_log_when_the_bridge_exits_at_once(rig: Rig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    failing = tmp_path / "failing"
    failing.mkdir()
    script = failing / "ros2"
    script.write_text("#!/bin/sh\nexit 3\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{failing}{os.pathsep}{os.environ['PATH']}")
    with pytest.raises(ToolError, match="exited with 3"):
        rig.run(viz.viz_rerun_start())
    assert viz.viz_rerun_list() == {"views": []}
