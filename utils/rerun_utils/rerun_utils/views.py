"""Rerun bridge processes per env: the launch argv, the shared state files and the viewer URL."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
import subprocess
import time
import urllib.parse
from pathlib import Path

from arena_bringup.viz_backends import rerun_launch_argv

from arena_mcp.core import proc

LOOPBACK = "127.0.0.1"
START_TIMEOUT_S = 60.0


class StartTimeout(TimeoutError):
    def __init__(self, pid: int, message: str) -> None:
        super().__init__(message)
        self.pid = pid


def viewer_url(web_port: int, grpc_port: int) -> str:
    proxy = f"rerun+http://{LOOPBACK}:{grpc_port}/proxy"
    return f"http://{LOOPBACK}:{web_port}/?url={urllib.parse.quote(proxy, safe='')}"


def state_dir() -> Path:
    return Path("/tmp/arena-viz") / os.environ.get("ROS_DOMAIN_ID", "0")


def log_dir() -> Path:
    if os.environ.get("ROS_LOG_DIR"):
        return Path(os.environ["ROS_LOG_DIR"])
    return Path(os.environ.get("ROS_HOME") or Path.home() / ".ros") / "log"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind((LOOPBACK, 0))
        return sock.getsockname()[1]


def accepts(port: int) -> bool:
    with contextlib.suppress(OSError), socket.create_connection((LOOPBACK, port), timeout=0.5):
        return True
    return False


@dataclasses.dataclass(frozen=True)
class View:
    env_id: int
    ns: str
    pid: int
    started: int | None
    web_port: int
    grpc_port: int
    log: str

    @property
    def url(self) -> str:
        return viewer_url(self.web_port, self.grpc_port)

    @property
    def alive(self) -> bool:
        return proc.alive(self.pid, self.started)

    def public(self) -> dict[str, object]:
        return {
            "env_id": self.env_id,
            "ns": self.ns,
            "pid": self.pid,
            "web_port": self.web_port,
            "grpc_port": self.grpc_port,
            "url": self.url,
            "alive": self.alive,
        }


def _path(env_id: int) -> Path:
    return state_dir() / f"env_{env_id}.json"


def load(env_id: int) -> View | None:
    """The live view of env_id, pruning its state file when the process is gone."""
    path = _path(env_id)
    try:
        view = View(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return None
    if view.alive:
        return view
    path.unlink(missing_ok=True)
    return None


def load_all() -> list[View]:
    if not state_dir().is_dir():
        return []
    views = (load(int(p.stem.removeprefix("env_"))) for p in sorted(state_dir().glob("env_*.json")))
    return [v for v in views if v is not None]


def forget(env_id: int) -> None:
    _path(env_id).unlink(missing_ok=True)


def spawn(env_id: int, ns: str, web_port: int, grpc_port: int, timeout: float = START_TIMEOUT_S) -> View:
    """Start the bridge detached, record it, and wait until its web port accepts connections."""
    web_port = web_port or free_port()
    grpc_port = grpc_port or free_port()
    log = log_dir() / f"rerun_view_env_{env_id}_{time.strftime('%Y%m%d-%H%M%S')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        process = subprocess.Popen(
            rerun_launch_argv(ns, web_port, grpc_port),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    view = View(env_id, ns, process.pid, proc.start_time(process.pid), web_port, grpc_port, str(log))
    state_dir().mkdir(parents=True, exist_ok=True)
    _path(env_id).write_text(json.dumps(dataclasses.asdict(view)))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            forget(env_id)
            raise RuntimeError(f"the rerun bridge exited with {process.returncode}, log {log}")
        if accepts(web_port):
            return view
        time.sleep(0.25)
    forget(env_id)
    raise StartTimeout(process.pid, f"web port {web_port} did not accept connections within {timeout:.0f} s, log {log}")
