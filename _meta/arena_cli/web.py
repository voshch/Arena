"""`arena web`: one web hub over this machine's Arena runtimes, the dev container and its forks, proxying each container's dashboard."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime
import http.client
import http.server
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import typing
import urllib.parse

from arena_cli import cli, fork

DEFAULT_PORT = 8760
SERVICE_LABEL = "com.docker.compose.service=arena"
VIZ_PREFIX = "viz: "
CONTROL_HEADER = "X-Arena-Dashboard"
CONNECT_TIMEOUT_S = 90.0
PROXY_TIMEOUT_S = 90.0
STATUS_TIMEOUT_S = 6.0
REFRESH_S = 1.0
DISCOVER_S = 10.0
SOURCED = "source /opt/arena_ws/source >/dev/null 2>&1"
SERVE_COMMAND = "exec ros2 run arena_evaluation evaluation_cli serve --runtime --port 0 --exit-on-eof"
FORK_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SIM_RE = re.compile(r"^[a-z0-9_]*$")
LAUNCH_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:=")
MAX_BODY_BYTES = 65536
PAGE = pathlib.Path(__file__).resolve().parents[2] / "arena_evaluation" / "arena_evaluation" / "arena_evaluation" / "benchmark" / "dashboard.html"
PLACEHOLDER_RE = re.compile(r"[\w=.:@/-]*<\w+>[\w=.:@/-]*|\[[^\]]*\]")
HUB_META = b'<meta name="arena-hub" content="1">'


class HubError(Exception):
    """A hub request answered with an HTTP error status, a message and an optional next step."""

    def __init__(self, status: int, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.hint = hint


@dataclasses.dataclass
class Target:
    """One arena container: the dev container or a fork, with the dashboard the hub runs inside it."""

    container: str
    name: str
    kind: str
    state: str
    dashboard: subprocess.Popen[str] | None = None
    log: pathlib.Path | None = None
    port: int | None = None
    error: str | None = None
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)

    @property
    def id(self) -> str:
        """The fork name, or dev, the one way the hub, the API and the page address a target."""
        return self.name

    @property
    def url(self) -> str:
        return f'/t/{urllib.parse.quote(self.id)}'

    def exec_argv(self, script: str, *, stdin: bool = False) -> list[str]:
        return ['docker', 'exec', *(['-i'] if stdin else []), self.container, '/entrypoint.sh', 'bash', '--norc', '-c', f'{SOURCED}; {script}']

    def dashboard_state(self) -> str:
        if self.dashboard is not None and self.dashboard.poll() is None and self.port is not None:
            return "up"
        if self.dashboard is not None and self.dashboard.poll() is None:
            return "starting"
        return "failed" if self.error else "down"


def parse_containers(inspected: list[dict[str, typing.Any]]) -> list[Target]:
    """Targets from `docker inspect` output: the arena service containers, forks named by their ARENA_FORK."""
    targets = []
    for entry in inspected:
        env = dict(item.split("=", 1) for item in entry["Config"]["Env"] if "=" in item)
        fork = env.get("ARENA_FORK", "")
        targets.append(Target(container=entry["Name"].lstrip("/"), name=fork or "dev", kind="fork" if fork else "dev", state=entry["State"]["Status"]))
    return sorted(targets, key=lambda target: (target.kind != "dev", target.name))


def list_argv() -> list[str]:
    return ["docker", "ps", "-a", "-q", "--filter", f"label={SERVICE_LABEL}"]


def inspect_argv(ids: list[str]) -> list[str]:
    return ["docker", "inspect", *ids]


def runtime_up_argv(sim: str, tokens: list[str], template: bool = False) -> list[str]:
    if template:
        return ["arena", "runtime", "--bg", f"[sim:={sim}]", "headless:=true", f"[{' '.join(tokens)} ...]"]
    return ["arena", "runtime", "--bg", *([f"sim:={sim}"] if sim else []), "headless:=true", *tokens]


def runtime_down_argv() -> list[str]:
    return ["arena", "runtime", "down"]


def fork_down_argv(name: str) -> list[str]:
    return ["arena", "fork", "down", name]


def fork_new_argv(name: str) -> list[str]:
    return ["arena", "fork", "new", name]


def render(argv: list[str]) -> str:
    """argv as a shell command, with <placeholder> words left unquoted."""
    return " ".join(arg if PLACEHOLDER_RE.fullmatch(arg) else shlex.quote(arg) for arg in argv)


def usage_doc(help_text: str, usage_prefix: str) -> str:
    """The description beside the usage line that starts with usage_prefix in a verb's help."""
    for line in help_text.splitlines():
        usage, _, doc = line.strip().partition("   ")
        if usage.startswith(usage_prefix) and doc.strip():
            return doc.strip()
    raise ValueError(f"no usage line {usage_prefix!r} in the help")


def action_table(serve_command: str) -> dict[str, dict[str, str]]:
    """The hub's actions with the commands they run, built by the same argv functions with placeholders."""
    ghost = Target(container="<container>", name="<target>", kind="dev", state="running")
    runtime = cli.registered_verb("runtime")
    up = runtime_up_argv("<sim>", ["<key>:=<value>"], template=True)
    down = runtime_down_argv()
    fork_down = fork_down_argv("<name>")
    fork_new = fork_new_argv("<name>")

    def entry(title: str, argv: list[str], doc: str) -> dict[str, str]:
        return {"label": title, "doc": doc, "command": render(ghost.exec_argv(render(argv)))}

    return {
        "connect": {"label": "Connect", "doc": "Start the dashboard of a container and open it through the hub.", "command": render(ghost.exec_argv(serve_command, stdin=True))},
        "runtime_up": entry(runtime.title, up, usage_doc(cli.RUNTIME_HELP, f"{' '.join(up[:2])} [")),
        "runtime_down": entry(runtime.subtitles[down[2]], down, usage_doc(cli.RUNTIME_HELP, " ".join(down))),
        "fork_down": entry(fork.COMMANDS[fork_down[2]].title, fork_down, fork.COMMANDS[fork_down[2]].short),
        "fork_new": entry(fork.COMMANDS[fork_new[2]].title, fork_new, fork.COMMANDS[fork_new[2]].short),
    }


def discover() -> list[Target]:
    listed = subprocess.run(list_argv(), capture_output=True, text=True, timeout=30, check=False)
    if listed.returncode != 0:
        raise HubError(502, f"docker ps failed: {listed.stderr.strip() or listed.returncode}")
    ids = listed.stdout.split()
    if not ids:
        return []
    inspected = subprocess.run(inspect_argv(ids), capture_output=True, text=True, timeout=30, check=False)
    if inspected.returncode != 0:
        raise HubError(502, f"docker inspect failed: {inspected.stderr.strip()}")
    return parse_containers(json.loads(inspected.stdout))


@dataclasses.dataclass
class Activity:
    t: str
    what: str
    target: str
    state: str = "running"
    output: str = ""
    started: float = dataclasses.field(default_factory=time.time)


class Hub:
    """The hub's state: this machine's targets, the dashboards it runs inside them, and its activity log."""

    def __init__(self, serve_command: str, only: str | None) -> None:
        self.state_dir = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp") / f"arena-web-{os.getpid()}"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.refreshed_at = 0.0
        self.polled_at: float | None = None
        self.runtimes: dict[str, dict[str, typing.Any]] = {}
        self.wake = threading.Event()
        self.discover_due = threading.Event()
        self.halt = threading.Event()
        self.thread: threading.Thread | None = None
        self.serve_command = serve_command
        self.actions = action_table(serve_command)
        self.only = re.compile(only) if only else None
        self.targets: dict[str, Target] = {}
        self.errors: dict[str, str] = {}
        self.activity: list[Activity] = []
        self.busy: set[str] = set()
        self.lock = threading.Lock()

    def refresh(self) -> None:
        try:
            found = discover()
        except (HubError, subprocess.TimeoutExpired, ValueError) as exc:
            with self.lock:
                self.errors["docker"] = str(exc)
            return
        listed = {target.id for target in found}
        with self.lock:
            self.errors.pop("docker", None)
            for target in found:
                if self.only is not None and target.kind == "fork" and not self.only.search(target.id):
                    continue
                known = self.targets.get(target.id)
                if known is None:
                    self.targets[target.id] = target
                else:
                    known.state, known.container = target.state, target.container
            gone = [target for target_id, target in list(self.targets.items()) if target_id not in listed]
            for target in gone:
                del self.targets[target.id]
        for target in gone:
            self.stop(target)

    def poll(self) -> None:
        """Fetch /api/runtime of every target whose dashboard is up and store the answers."""
        with self.lock:
            up = [target for target in self.targets.values() if target.dashboard_state() == "up"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            answers = list(pool.map(_runtime, up))
        with self.lock:
            self.runtimes = {target.id: answer for target, answer in zip(up, answers, strict=True) if answer is not None}
            self.polled_at = time.monotonic()
            self.refreshed_at = time.time()

    def start(self) -> None:
        """Fill the cache once, then keep it fresh from a background thread until close."""
        self.refresh()
        self.poll()
        self.thread = threading.Thread(target=self._run, name="arena-web-refresh", daemon=True)
        self.thread.start()

    def poke(self, discover: bool = False) -> None:
        """Wake the background thread for an immediate pass, discovering containers too if asked."""
        if discover:
            self.discover_due.set()
        self.wake.set()

    def _run(self) -> None:
        discovered = time.monotonic()
        while not self.halt.is_set():
            self.wake.wait(REFRESH_S)
            if self.halt.is_set():
                break
            self.wake.clear()
            if self.discover_due.is_set() or time.monotonic() - discovered >= DISCOVER_S:
                self.discover_due.clear()
                self.refresh()
                discovered = time.monotonic()
            self.poll()

    def target(self, target_id: str) -> Target:
        with self.lock:
            target = self.targets.get(target_id)
        if target is None:
            self.refresh()
            with self.lock:
                target = self.targets.get(target_id)
        if target is None:
            raise HubError(404, f"no arena container {target_id!r}", "the Targets page lists them")
        return target

    def connect(self, target: Target) -> int:
        """The local port of the target's dashboard, started inside the container on first use."""
        with target.lock:
            if target.dashboard_state() == "up":
                assert target.port is not None
                return target.port
            if target.state != "running":
                raise HubError(409, f"{target.id} is {target.state}", "start it with arena fork exec or source arena")
            target.error = None
            target.log = self.state_dir / f"{re.sub(r'[^A-Za-z0-9._-]', '_', target.id)}.log"
            with target.log.open("w") as log:
                proc = subprocess.Popen(target.exec_argv(self.serve_command, stdin=True), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, start_new_session=True)
            target.dashboard = proc
            line = _first_line(proc, CONNECT_TIMEOUT_S)
            if not line.startswith(VIZ_PREFIX):
                self.stop(target)
                target.error = explain(line or _tail(target.log))
                raise HubError(502, target.error, f"the dashboard's log is {target.log}")
            target.port = urllib.parse.urlsplit(line[len(VIZ_PREFIX) :].strip()).port
            assert target.port is not None
        self.poke()
        return target.port

    def stop(self, target: Target) -> None:
        proc, target.dashboard, target.port = target.dashboard, None, None
        self.poke()
        if proc is None or proc.poll() is not None:
            return
        with contextlib.suppress(OSError):
            assert proc.stdin is not None
            proc.stdin.close()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 15)
            proc.wait(5)

    def close(self) -> None:
        self.halt.set()
        self.wake.set()
        if self.thread is not None:
            self.thread.join(15)
        with self.lock:
            targets = list(self.targets.values())
        for target in targets:
            self.stop(target)
        shutil.rmtree(self.state_dir, ignore_errors=True)

    def summary(self) -> dict[str, typing.Any]:
        if self.polled_at is None:
            self.refresh()
            self.poll()
        with self.lock:
            targets = list(self.targets.values())
            runtimes = dict(self.runtimes)
            errors = dict(self.errors)
            busy = sorted(self.busy)
            polled_at = self.polled_at
            refreshed_at = self.refreshed_at
        rows = []
        for target in targets:
            state = target.dashboard_state()
            rows.append({"id": target.id, "url": target.url, "name": target.name, "kind": target.kind, "container": target.state, "dashboard": {"state": state, "error": target.error}, "runtime": runtimes.get(target.id) if state == "up" else None})
        age = time.monotonic() - polled_at if polled_at is not None else 0.0
        return {"targets": rows, "errors": [f"{name}: {error}" for name, error in errors.items()], "busy": busy, "actions": self.actions, "age_s": round(age, 1), "refreshed_at": refreshed_at}

    def run_action(self, what: str, key: str, argv: list[str], timeout: float, after: typing.Callable[[], None] | None = None) -> Activity:
        with self.lock:
            if key in self.busy:
                raise HubError(409, f"{key} is busy with another action", "the Activity list shows it")
            self.busy.add(key)
            entry = Activity(t=datetime.datetime.now().strftime("%H:%M:%S"), what=what, target=key)
            self.activity.insert(0, entry)
            del self.activity[50:]

        def work() -> None:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            timer = threading.Timer(timeout, proc.kill)
            timer.start()
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    entry.output = (entry.output + line)[-600:]
                rc = proc.wait()
                if not timer.is_alive():
                    entry.state, entry.output = "failed", f"{entry.output.strip()}\nno answer within {timeout:g}s".strip()
                else:
                    entry.state = "done" if rc == 0 else "failed"
                    entry.output = entry.output.strip()
            finally:
                timer.cancel()
                with self.lock:
                    self.busy.discard(key)
                if after is not None:
                    after()

        threading.Thread(target=work, name=f"arena-web-{what}", daemon=True).start()
        return entry

    def act(self, target_id: str, action: str, body: dict[str, typing.Any]) -> Activity:
        target = self.target(target_id)
        if action == "connect":
            self.connect(target)
            return Activity(t=datetime.datetime.now().strftime("%H:%M:%S"), what="connect", target=target.id, state="done")
        if action == "runtime_up":
            sim = body.get("sim", "")
            if not isinstance(sim, str) or not SIM_RE.match(sim.strip()):
                raise HubError(400, f"sim must be a simulator name (lowercase letters, digits, '_') or empty for the container's default, got {sim!r}")
            tokens = _launch_tokens(body.get("args", ""))
            argv = runtime_up_argv(sim.strip(), tokens)
            return self.run_action(f"runtime up ({sim.strip() or 'default sim'}) in {target.id}", target.id, target.exec_argv(shlex.join(argv)), 120, self.poke)
        if action == "runtime_down":
            return self.run_action(f"runtime down in {target.id}", target.id, target.exec_argv(shlex.join(runtime_down_argv())), 120, self.poke)
        if action == "fork_down":
            if target.kind != "fork":
                raise HubError(400, f"{target.id} is not a fork")
            if body.get("confirm") != target.name:
                raise HubError(400, f"deleting a fork needs confirm={target.name!r} in the body")
            dev = self._dev()
            self.stop(target)
            return self.run_action(f"fork down {target.name}", target.id, dev.exec_argv(shlex.join(fork_down_argv(target.name))), 300, self._poke_discover)
        raise HubError(404, f"no target action {action!r}", "actions: connect, runtime_up, runtime_down, fork_down")

    def fork_new(self, body: dict[str, typing.Any]) -> Activity:
        name = body.get("name", "")
        if not isinstance(name, str) or not FORK_NAME_RE.match(name):
            raise HubError(400, f"fork names are lowercase letters, digits and '-', got {name!r}")
        dev = self._dev()
        return self.run_action(f"fork new {name}", "fork-new", dev.exec_argv(shlex.join(fork_new_argv(name))), 900, self._poke_discover)

    def _poke_discover(self) -> None:
        self.poke(True)

    def _dev(self) -> Target:
        with self.lock:
            dev = next((target for target in self.targets.values() if target.kind == "dev"), None)
        if dev is None or dev.state != "running":
            raise HubError(409, "no running dev container, the fork verbs run there", "source arena")
        return dev


def _runtime(target: Target) -> dict[str, typing.Any] | None:
    with contextlib.suppress(OSError, ValueError, http.client.HTTPException):
        connection = http.client.HTTPConnection("127.0.0.1", target.port, timeout=STATUS_TIMEOUT_S)
        connection.request("GET", "/api/runtime")
        response = connection.getresponse()
        if response.status == 200:
            return json.loads(response.read())
    return None


def _launch_tokens(args: object) -> list[str]:
    """key:=value launch arguments from a free-text field, one per whitespace-separated (shell-quoted) token."""
    if not isinstance(args, str):
        raise HubError(400, "args must be text of key:=value launch arguments")
    try:
        tokens = shlex.split(args)
    except ValueError as exc:
        raise HubError(400, f"args do not parse: {exc}") from exc
    bad = [token for token in tokens if not LAUNCH_TOKEN_RE.match(token)]
    if bad:
        raise HubError(400, f"launch arguments are key:=value, got {', '.join(bad)}")
    return tokens


def _first_line(proc: subprocess.Popen[str], timeout: float) -> str:
    out: list[str] = []
    reader = threading.Thread(target=lambda: out.append(proc.stdout.readline() if proc.stdout else ""), daemon=True)
    reader.start()
    reader.join(timeout)
    return out[0] if out else ""


def explain(output: str) -> str:
    """One line on why a container's dashboard did not start, from what it printed."""
    if "evaluation_cli" in output and ("No such file" in output or "No executable found" in output or "Package 'arena_evaluation' not found" in output):
        return "no arena_evaluation dashboard in this container, it predates arena web: update and build it (arena update, arena build)"
    if "invalid choice: 'serve'" in output or "unrecognized arguments: --runtime" in output or "unrecognized arguments: --exit-on-eof" in output:
        return "this container's arena_evaluation predates serve --runtime: update and build it (arena update, arena build)"
    lines = [line.strip() for line in output.splitlines() if line.strip() and not line.startswith("[ros2run]")]
    return (lines[-1] if lines else "the dashboard exited without output")[:200]


def _tail(log: pathlib.Path | None) -> str:
    if log is None or not log.is_file():
        return ""
    return log.read_text(errors="replace")[-600:]


class HubServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], hub: Hub, prefix: str = "") -> None:
        self.hub = hub
        self.prefix = normalize_prefix(prefix)
        super().__init__(address, _Handler)


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    @property
    def hub(self) -> Hub:
        return typing.cast(HubServer, self.server).hub

    def log_message(self, format: str, *args: typing.Any) -> None:
        pass

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    @property
    def prefix(self) -> str:
        return typing.cast(HubServer, self.server).prefix

    def _dispatch(self, method: str) -> None:
        parts = [urllib.parse.unquote(part) for part in urllib.parse.urlsplit(self.path).path.split("/") if part]
        if self.prefix:
            mounted = [part for part in self.prefix.split("/") if part]
            if parts[: len(mounted)] != mounted:
                self._json({"error": f"this hub is served under {self.prefix}", "hint": f"open {self.prefix}/"}, 404)
                return
            parts = parts[len(mounted) :]
        try:
            if parts[:1] == ["t"] and len(parts) >= 2:
                self._proxy(method, parts[1], parts[2:])
            elif method == "GET" and not parts:
                self._page()
            elif parts[:1] == ["hub"]:
                self._hub(method, parts[1:])
            else:
                raise HubError(404, f"no endpoint {self.path}", "/ is the Targets page, /t/<container>/ a target's dashboard")
        except HubError as exc:
            self._json({"error": exc.message, "hint": exc.hint}, exc.status)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _page(self) -> None:
        if not PAGE.is_file():
            raise HubError(500, f"the dashboard page is missing at {PAGE}", "check out the arena_evaluation submodule")
        self._send(200, "text/html; charset=utf-8", _mark(PAGE.read_bytes(), self.prefix))

    def _body(self) -> dict[str, typing.Any]:
        if self.headers.get(CONTROL_HEADER) != "1":
            raise HubError(403, f"control requests need the {CONTROL_HEADER}: 1 header", "the page sends it, other origins cannot")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise HubError(413, f"request body over {MAX_BODY_BYTES} bytes")
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except ValueError as exc:
            raise HubError(400, f"the request body is not JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise HubError(400, "the request body must be a JSON object")
        return body

    def _hub(self, method: str, rest: list[str]) -> None:
        hub = self.hub
        if method == "GET" and rest == ["targets"]:
            self._json(hub.summary())
        elif method == "GET" and rest == ["activity"]:
            with hub.lock:
                self._json({"items": [dataclasses.asdict(item) for item in hub.activity]})
        elif method == "POST" and len(rest) == 3 and rest[0] == "targets":
            self._json(dataclasses.asdict(hub.act(rest[1], rest[2], self._body())))
        elif method == "POST" and rest == ["fork_new"]:
            self._json(dataclasses.asdict(hub.fork_new(self._body())))
        else:
            raise HubError(404, f"no hub endpoint {method} /hub/{'/'.join(rest)}")

    def _proxy(self, method: str, container: str, rest: list[str]) -> None:
        hub = self.hub
        target = hub.target(container)
        port = hub.connect(target)
        upstream = f"/{'/'.join(rest)}" if rest else "/"
        if query := urllib.parse.urlsplit(self.path).query:
            upstream += f"?{query}"
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
        headers = {key: value for key, value in self.headers.items() if key in ("Content-Type", CONTROL_HEADER)}
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=PROXY_TIMEOUT_S)
        try:
            connection.request(method, upstream, body=body, headers=headers)
            response = connection.getresponse()
        except OSError as exc:
            hub.stop(target)
            raise HubError(502, f"the dashboard of {target.id} did not answer: {exc}", "reload to reconnect") from exc
        content_type = response.getheader("Content-Type", "application/octet-stream")
        if content_type.startswith("text/event-stream"):
            self.send_response(response.status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            while chunk := response.read1(65536):
                self.wfile.write(chunk)
                self.wfile.flush()
            return
        data = response.read()
        self._send(response.status, content_type, _mark(data, self.prefix) if content_type.startswith("text/html") else data)

    def _json(self, body: object, code: int = 200) -> None:
        self._send(code, "application/json", json.dumps(body, default=str).encode())

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def normalize_prefix(prefix: str) -> str:
    """A mount path as `/sub`, empty when the hub owns the root."""
    trimmed = prefix.strip().strip("/")
    return f"/{trimmed}" if trimmed else ""


def _mark(html: bytes, prefix: str) -> bytes:
    root = f'<meta name="arena-root" content="{prefix}">'.encode()
    return html.replace(b"<head>", b"<head>\n" + HUB_META + root, 1)


def open_browser(url: str) -> str | None:
    """Open url in the desktop's browser through xdg-open, returning why it was not opened."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return "no DISPLAY or WAYLAND_DISPLAY"
    opener = shutil.which("xdg-open")
    if opener is None:
        return "xdg-open is not installed"
    subprocess.Popen([opener, url], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arena web",
        description="One web hub over this machine's Arena runtimes: the dev container and its forks, each with its runtime and benchmark pages. Runs until Ctrl-C.",
    )
    parser.add_argument("--listen", default="127.0.0.1", help="address to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--only", default=None, metavar="REGEX", help="only forks whose name matches REGEX, dev containers always show")
    parser.add_argument("--serve-command", default=SERVE_COMMAND, help="the command each container runs for its dashboard, in a sourced shell")
    parser.add_argument("--prefix", default="", metavar="PATH", help="serve under this path when a reverse proxy mounts the hub there, e.g. /arena")
    parser.add_argument("--no-open", action="store_true", help="do not open the hub in the browser (xdg-open) once it listens")
    args = parser.parse_args(argv)
    previous = {signum: signal.signal(signum, signal.default_int_handler) for signum in (signal.SIGTERM, signal.SIGHUP)}
    hub = Hub(args.serve_command, args.only)
    server: HubServer | None = None
    try:
        server = HubServer((args.listen, args.port), hub, args.prefix)
        hub.start()
        link = f"http://{'127.0.0.1' if args.listen in ('', '0.0.0.0') else args.listen}:{server.server_address[1]}{server.prefix}/"
        print(f"arena web: {link} over this machine, {len(hub.targets)} containers", flush=True)
        if not args.no_open and (reason := open_browser(link)) is not None:
            print(f"arena web: not opening a browser, {reason}", file=sys.stderr)
        for source, error in hub.errors.items():
            print(f"arena web: {source}: {error}", file=sys.stderr)
        server.serve_forever(poll_interval=0.5)
    except OSError as exc:
        if server is not None:
            raise
        print(f"arena web: cannot listen on {args.listen}:{args.port}: {exc.strerror}, pick another with --port", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        hub.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0
