import json
import os
import pathlib
import re
import shlex
import socket
import stat
import sys
import threading
import time
import typing
import urllib.error
import urllib.request
from collections.abc import Iterator

import pytest

from arena_cli import cli, fork, web

STUB_DASHBOARD = r"""
import http.server, json, socket, sys, threading

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, kind, body):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html", b"<html><head><title>stub</title></head></html>")
        elif self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for name in ("run", "live"):
                self.wfile.write(f"event: {name}\ndata: {{}}\n\n".encode())
                self.wfile.flush()
        else:
            self._send(200, "application/json", json.dumps({"path": self.path, "running": True, "container": sys.argv[1]}).encode())

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._send(200, "application/json", json.dumps({"path": self.path, "body": json.loads(body), "header": self.headers.get("X-Arena-Dashboard")}).encode())

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
print(f"viz: http://127.0.0.1:{server.server_address[1]}/runtime", flush=True)
threading.Thread(target=server.serve_forever, daemon=True).start()
for _ in sys.stdin:
    pass
"""

FAKE_DOCKER = r"""#!/usr/bin/env python3
import json, os, subprocess, sys

containers = json.loads(os.environ["FAKE_CONTAINERS"])
args = sys.argv[1:]
if os.environ.get("FAKE_DOCKER_LOG"):
    with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
        log.write(json.dumps(args) + "\n")
if args[0] == "ps":
    print("\n".join(c["Id"] for c in containers))
elif args[0] == "inspect":
    print(json.dumps([c for c in containers if c["Id"] in args[1:]]))
elif args[0] == "exec":
    rest = [a for a in args[1:] if a != "-i"]
    container, script = rest[0], rest[-1].split("; ", 1)[1]
    if "stub_dashboard.py" not in script:
        print(f"fake exec in {container}: {script}", file=sys.stderr)
        sys.exit(3)
    os.environ["FAKE_CONTAINER"] = container
    sys.exit(subprocess.call(["bash", "--norc", "--noprofile", "-c", script]))
"""


def _container(cid: str, name: str, fork: str, status: str = "running", network: str = "host") -> dict:
    return {"Id": cid, "Name": f"/{name}", "State": {"Status": status}, "HostConfig": {"NetworkMode": network}, "Config": {"Env": ["PATH=/usr/bin", *([f"ARENA_FORK={fork}"] if fork else [])]}}


@pytest.fixture
def fake_docker(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(FAKE_DOCKER.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    stub = tmp_path / "stub_dashboard.py"
    stub.write_text(STUB_DASHBOARD)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(tmp_path / "docker.log"))
    monkeypatch.setenv("FAKE_CONTAINERS", json.dumps([_container("c1", "arena-ws-arena-1", ""), _container("c2", "arena-ws-p1-arena-1", "p1"), _container("c3", "arena-ws-old-arena-1", "old", "exited")]))
    return stub


@pytest.fixture
def hub(fake_docker: pathlib.Path) -> Iterator[tuple[str, web.Hub]]:
    state = web.Hub(f'exec {sys.executable} {fake_docker} "$FAKE_CONTAINER"', None)
    server = web.HubServer(("127.0.0.1", 0), state)
    state.start()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()
        state.close()


def _call(url: str, body: object | None = None, header: bool = True) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **({"X-Arena-Dashboard": "1"} if header else {})}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_parse_containers_names_forks_by_arena_fork_and_lists_dev_first() -> None:
    targets = web.parse_containers([_container("b", "arena-ws-zeta-arena-1", "zeta"), _container("a", "arena-ws-arena-1", ""), _container("c", "arena-ws-alpha-arena-1", "alpha", "exited")])
    assert [(t.id, t.kind, t.container, t.state) for t in targets] == [("dev", "dev", "arena-ws-arena-1", "running"), ("alpha", "fork", "arena-ws-alpha-arena-1", "exited"), ("zeta", "fork", "arena-ws-zeta-arena-1", "running")]


def test_targets_list_running_and_stopped_containers_unconnected(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    status, data = _call(f"{base}/hub/targets")
    assert status == 200 and data["errors"] == []
    assert [(t["id"], t["kind"], t["container"], t["dashboard"]["state"]) for t in data["targets"]] == [("dev", "dev", "running", "down"), ("old", "fork", "exited", "down"), ("p1", "fork", "running", "down")]
    assert [t["url"] for t in data["targets"]] == ["/t/dev", "/t/old", "/t/p1"]


def test_proxy_starts_the_dashboard_and_forwards_paths_bodies_headers_events_and_pages(hub: tuple[str, web.Hub]) -> None:
    base, state = hub
    status, got = _call(f"{base}/t/p1/api/runtime?x=1")
    post_status, posted = _call(f"{base}/t/p1/api/runtime/hold", {"seconds": 2})
    with urllib.request.urlopen(f"{base}/t/p1/events", timeout=30) as stream:
        events = stream.read().decode()
    with urllib.request.urlopen(f"{base}/t/p1/", timeout=30) as page:
        html = page.read()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        _, targets = _call(f"{base}/hub/targets")
        if next(t for t in targets["targets"] if t["id"] == "p1")["runtime"] is not None:
            break
        time.sleep(0.1)

    assert status == 200 and got == {"path": "/api/runtime?x=1", "running": True, "container": "arena-ws-p1-arena-1"}
    assert post_status == 200 and posted == {"path": "/api/runtime/hold", "body": {"seconds": 2}, "header": "1"}
    assert events == "event: run\ndata: {}\n\nevent: live\ndata: {}\n\n"
    assert b'<head>\n<meta name="arena-hub" content="1"><meta name="arena-root" content=""><title>stub</title>' in html
    p1 = next(t for t in targets["targets"] if t["id"] == "p1")
    assert p1["dashboard"]["state"] == "up" and p1["runtime"]["running"] is True
    assert state.targets["p1"].dashboard is not None


def test_closing_the_hub_ends_the_dashboards_through_stdin(hub: tuple[str, web.Hub]) -> None:
    base, state = hub
    _call(f"{base}/t/dev/api/runtime")
    proc = state.targets["dev"].dashboard
    assert proc is not None and proc.poll() is None
    state.close()
    assert proc.wait(10) == 0
    assert not state.state_dir.exists()


def test_stopped_and_unknown_containers_answer_with_errors(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    assert _call(f"{base}/t/old/api/runtime") == (409, {"error": "old is exited", "hint": "start it with arena fork exec or source arena"})
    assert _call(f"{base}/t/nope/api/runtime")[0] == 404


def test_hub_controls_need_the_header_and_valid_input(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    forged = _call(f"{base}/hub/targets/p1/runtime_up", {"sim": "flatsim"}, header=False)
    bad_sim = _call(f"{base}/hub/targets/p1/runtime_up", {"sim": "Bad Sim"})
    bad_args = _call(f"{base}/hub/targets/p1/runtime_up", {"sim": "dummy", "args": "world:=a nonsense"})
    bad_name = _call(f"{base}/hub/fork_new", {"name": "Bad Name"})
    no_confirm = _call(f"{base}/hub/targets/p1/fork_down", {})
    not_fork = _call(f"{base}/hub/targets/dev/fork_down", {"confirm": "dev"})
    unknown = _call(f"{base}/hub/targets/p1/explode", {})
    assert forged[0] == 403 and "X-Arena-Dashboard" in forged[1]["error"]
    assert bad_sim[0] == 400 and "sim must be a simulator name" in bad_sim[1]["error"]
    assert bad_args == (400, {"error": "launch arguments are key:=value, got nonsense", "hint": None})
    assert bad_name[0] == 400 and "fork names" in bad_name[1]["error"]
    assert no_confirm[0] == 400 and "confirm='p1'" in no_confirm[1]["error"]
    assert not_fork == (400, {"error": "dev is not a fork", "hint": None})
    assert unknown[0] == 404 and "runtime_up" in unknown[1]["hint"]


def test_actions_run_in_the_container_and_land_in_the_activity_log(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    status, started = _call(f"{base}/hub/targets/p1/runtime_up", {"sim": "dummy"})
    deadline = time.monotonic() + 20
    items: list[dict] = []
    while time.monotonic() < deadline:
        items = _call(f"{base}/hub/activity")[1]["items"]
        if items and items[0]["state"] != "running":
            break
        time.sleep(0.2)
    assert status == 200 and started["what"] == "runtime up (dummy) in p1" and started["state"] == "running"
    assert items[-1]["what"] == "runtime up (dummy) in p1" and items[-1]["state"] == "failed"
    assert items[-1]["output"] == "fake exec in arena-ws-p1-arena-1: arena runtime --bg sim:=dummy headless:=true"


def test_actions_stream_their_output_and_time_out(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    state = web.Hub("true", None)
    try:
        slow = state.run_action("slow", "a", ["bash", "-c", "echo building; sleep 2; echo done"], 30)
        deadline = time.monotonic() + 5
        while "building" not in slow.output and time.monotonic() < deadline:
            time.sleep(0.05)
        assert slow.state == "running" and slow.output == "building\n" and slow.started <= time.time()
        stuck = state.run_action("stuck", "b", ["bash", "-c", "echo waiting; sleep 30"], 0.5)
        deadline = time.monotonic() + 10
        while (slow.state == "running" or stuck.state == "running") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert (slow.state, slow.output) == ("done", "building\ndone")
        assert (stuck.state, stuck.output) == ("failed", "waiting\nno answer within 0.5s")
        assert state.busy == set()
    finally:
        state.close()


def test_runtime_up_passes_extra_args_quoted_and_leaves_an_empty_sim_to_the_container(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    _call(f"{base}/hub/targets/p1/runtime_up", {"sim": "", "args": "world:=map_empty 'robot.name:=a b'"})
    deadline = time.monotonic() + 20
    items: list[dict] = []
    while time.monotonic() < deadline:
        items = _call(f"{base}/hub/activity")[1]["items"]
        if items and items[0]["state"] != "running":
            break
        time.sleep(0.2)
    assert items[0]["what"] == "runtime up (default sim) in p1"
    assert items[0]["output"] == "fake exec in arena-ws-p1-arena-1: arena runtime --bg headless:=true world:=map_empty 'robot.name:=a b'"


def _docker_log() -> list[list[str]]:
    path = pathlib.Path(os.environ["FAKE_DOCKER_LOG"])
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _docker_calls(verb: str) -> int:
    return sum(1 for args in _docker_log() if args[0] == verb)


def _idle(base: str) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        items = _call(f"{base}/hub/activity")[1]["items"]
        if all(item["state"] != "running" for item in items):
            return
        time.sleep(0.1)


def _words(argv: list[str], values: dict[str, str]) -> list[str]:
    out: list[str] = []
    for arg in argv:
        arg = re.sub(r"\[(.*?)(?: \.\.\.)?\]", r"\1", arg)
        for key, value in values.items():
            arg = arg.replace(f"<{key}>", value)
        out.extend(shlex.split(arg) if " " in arg else [arg])
    return out


def test_action_commands_equal_what_the_actions_run(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    actions = _call(f"{base}/hub/targets")[1]["actions"]
    assert list(actions) == ["connect", "runtime_up", "runtime_down", "fork_down", "fork_new"]
    assert all({"label", "doc"} <= set(entry) for entry in actions.values())

    def last_exec(trigger: typing.Callable[[], object], container: str) -> list[str]:
        before = len(_docker_log())
        trigger()
        _idle(base)
        return next(args for args in reversed(_docker_log()[before:]) if args[0] == "exec" and container in args)

    values = {"container": "arena-ws-p1-arena-1", "sim": "dummy", "key": "world", "value": "map_empty", "name": "zz"}
    dev = {**values, "container": "arena-ws-arena-1"}
    runs = {
        "connect": (lambda: _call(f"{base}/t/p1/api/runtime"), values),
        "runtime_up": (lambda: _call(f"{base}/hub/targets/p1/runtime_up", {"sim": "dummy", "args": "world:=map_empty"}), values),
        "runtime_down": (lambda: _call(f"{base}/hub/targets/p1/runtime_down", {}), values),
        "fork_down": (lambda: _call(f"{base}/hub/targets/p1/fork_down", {"confirm": "p1"}), {**dev, "name": "p1"}),
        "fork_new": (lambda: _call(f"{base}/hub/fork_new", {"name": "zz"}), dev),
    }
    for action, (trigger, filled) in runs.items():
        logged = last_exec(trigger, filled["container"])
        assert _words(shlex.split(actions[action]["command"]), filled) == _words(["docker", *logged], {}), action


def test_action_docs_come_from_the_cli_help(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    actions = _call(f"{base}/hub/targets")[1]["actions"]
    assert actions["runtime_up"]["doc"] == web.usage_doc(cli.RUNTIME_HELP, "arena runtime [")
    assert actions["runtime_down"]["doc"] == web.usage_doc(cli.RUNTIME_HELP, "arena runtime down")
    assert actions["fork_down"]["doc"] == fork.COMMANDS["down"].short
    assert actions["fork_new"]["doc"] == fork.COMMANDS["new"].short
    assert [actions[a]["label"] for a in ("runtime_up", "runtime_down", "fork_down", "fork_new")] == ["Start Runtime", "Stop Runtime", "Delete Fork", "New Fork"]
    assert "arena runtime --bg [sim:=<sim>] headless:=true [<key>:=<value> ...]" in actions["runtime_up"]["command"]


def test_targets_answer_from_the_cache_without_running_discovery(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    before = _docker_calls("ps")
    ages = [_call(f"{base}/hub/targets")[1]["age_s"] for _ in range(5)]
    assert before >= 1 and _docker_calls("ps") == before
    assert all(0 <= age < web.DISCOVER_S for age in ages)


def test_poke_after_an_action_lists_a_new_container_before_the_discovery_interval(hub: tuple[str, web.Hub], monkeypatch: pytest.MonkeyPatch) -> None:
    base, _ = hub
    assert [t["id"] for t in _call(f"{base}/hub/targets")[1]["targets"]] == ["dev", "old", "p1"]
    monkeypatch.setenv("FAKE_CONTAINERS", json.dumps([_container("c1", "arena-ws-arena-1", ""), _container("c2", "arena-ws-p1-arena-1", "p1"), _container("c4", "arena-ws-p2-arena-1", "p2")]))
    assert _call(f"{base}/hub/targets/p1/fork_down", {"confirm": "p1"})[0] == 200
    deadline = time.monotonic() + web.DISCOVER_S / 2
    ids: list[str] = []
    while time.monotonic() < deadline:
        ids = [t["id"] for t in _call(f"{base}/hub/targets")[1]["targets"]]
        if "p2" in ids:
            break
        time.sleep(0.1)
    assert ids == ["dev", "p1", "p2"]


def test_a_reverse_proxy_mount_serves_the_page_under_its_prefix(fake_docker: pathlib.Path) -> None:
    state = web.Hub(f'exec {sys.executable} {fake_docker} "$FAKE_CONTAINER"', None)
    server = web.HubServer(("127.0.0.1", 0), state, "/arena/")
    state.start()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert web.normalize_prefix("/arena/") == "/arena" and web.normalize_prefix("") == ""
        with urllib.request.urlopen(f"{base}/arena/", timeout=10) as page:
            html = page.read()
        assert b'<meta name="arena-root" content="/arena">' in html
        assert _call(f"{base}/arena/hub/targets")[0] == 200
        outside = _call(f"{base}/hub/targets")
        assert outside[0] == 404 and "/arena" in outside[1]["hint"]
    finally:
        server.shutdown()
        server.server_close()
        state.close()


def test_root_serves_the_marked_dashboard_page(hub: tuple[str, web.Hub]) -> None:
    base, _ = hub
    if not web.PAGE.is_file():
        pytest.skip("arena_evaluation is not checked out next to _meta")
    with urllib.request.urlopen(f"{base}/", timeout=10) as page:
        html = page.read()
    assert b'<meta name="arena-hub" content="1">' in html and b"function targetsPage()" in html


def test_open_browser_runs_xdg_open_only_with_a_display(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    opened = tmp_path / "opened"
    opener = bindir / "xdg-open"
    opener.write_text(f"#!/bin/sh\necho \"$1\" > {opened}\n")
    opener.chmod(opener.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(bindir))
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert web.open_browser("http://127.0.0.1:1/") == "no DISPLAY or WAYLAND_DISPLAY"
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert web.open_browser("http://127.0.0.1:8760/") is None
    deadline = time.monotonic() + 5
    while not opened.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert opened.read_text() == "http://127.0.0.1:8760/\n"
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert web.open_browser("http://127.0.0.1:8760/") == "xdg-open is not installed"


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("FileNotFoundError: [Errno 2] No such file or directory: '/opt/arena_ws/install/arena_evaluation/lib/arena_evaluation/evaluation_cli'", "no arena_evaluation dashboard in this container"),
        ("evaluation_cli: error: argument command: invalid choice: 'serve' (choose from 'list', 'status')", "predates serve --runtime"),
        ("evaluation_cli: error: unrecognized arguments: --exit-on-eof", "predates serve --runtime"),
        ("Traceback (most recent call last):\n  File x\nRuntimeError: boom\n[ros2run]: Process exited with failure 1\n", "RuntimeError: boom"),
        ("", "the dashboard exited without output"),
    ],
)
def test_explain_turns_dashboard_failures_into_one_line(output: str, expected: str) -> None:
    explained = web.explain(output)
    assert expected in explained and "\n" not in explained and len(explained) <= 200


def test_sigterm_ends_the_hub_and_removes_its_state(tmp_path: pathlib.Path) -> None:
    import subprocess

    env = {**os.environ, "XDG_RUNTIME_DIR": str(tmp_path), "ARENA_WS_DIR": str(tmp_path), "PYTHONPATH": str(pathlib.Path(web.__file__).resolve().parents[1])}
    proc = subprocess.Popen([sys.executable, "-m", "arena_cli", "web", "--port", "0", "--no-open"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    assert proc.stdout is not None and proc.stdout.readline().startswith("arena web: http://127.0.0.1:")
    assert [p.name for p in tmp_path.iterdir()] == [f"arena-web-{proc.pid}"]
    proc.terminate()
    assert proc.wait(15) == 0
    assert list(tmp_path.iterdir()) == []


def test_arena_web_is_a_listed_verb_with_its_flags() -> None:
    verb = cli.registered_verb("web")
    assert "web" in cli.SECTIONS["Attach"] and not verb.hidden
    assert verb.short.startswith("Serve the web app")
    assert {"--port", "--listen", "--prefix", "--only", "--no-open"} <= set(verb.complete.flags)


def test_main_refuses_a_taken_port_and_leaves_no_state(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        assert web.main(["--port", str(taken.getsockname()[1])]) == 2
    assert "pick another with --port" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
