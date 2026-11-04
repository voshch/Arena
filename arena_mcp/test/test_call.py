from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

DEMO = '''
import asyncio
import sys

from mcp.server.mcpserver import Image
from mcp.types import InputRequiredResult

from arena_mcp.core import JOBS, Domain, ToolError

domain = Domain("demo", """Tools the call tests run.""")


@domain.tool(read_only=True)
def demo_echo(text: str, count: int = 1, tags: list[str] | None = None, shout: bool = False, note: str | None = "kept") -> dict[str, object]:
    """Echo the arguments.

    Second line of the description.
    """
    return {"text": text, "count": count, "tags": tags, "shout": shout, "note": note}


@domain.tool(read_only=True)
def demo_text() -> str:
    """A plain text result."""
    return "just text"


@domain.tool(read_only=True)
def demo_picture() -> Image:
    """An image result."""
    return Image(data=b"png bytes", format="png")


@domain.tool()
def demo_ask() -> InputRequiredResult:
    """Ask the client for more input."""
    return InputRequiredResult(request_state="pending")


@domain.tool()
def demo_fail() -> dict[str, object]:
    """Fail with a hint."""
    raise ToolError("broken", hint="fix it")


@domain.tool()
async def demo_task(hold_s: float = 0.0, fail: bool = False) -> dict[str, object]:
    """Start an in-process job."""

    async def body(handle) -> dict[str, object]:
        handle.log("task started")
        handle.progress(step=1)
        await asyncio.sleep(hold_s)
        if fail:
            raise ToolError("task broke")
        return {"done": True}

    return JOBS.start_task("demo", "demo task", body).as_dict()


@domain.tool()
async def demo_process() -> dict[str, object]:
    """Start a process job that runs until stopped."""
    argv = [sys.executable, "-c", "import time; print('process started', flush=True); time.sleep(120)"]
    return (await JOBS.start_process("demo", "demo process", argv)).as_dict()
'''


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    packages = tmp_path / "packages"
    info = packages / "arena_call_demo-0.dist-info"
    info.mkdir(parents=True)
    (packages / "arena_call_demo.py").write_text(DEMO)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: arena-call-demo\nVersion: 0\n")
    (info / "entry_points.txt").write_text("[arena_mcp.domains]\ndemo = arena_call_demo:domain\n")
    return {**os.environ, "PYTHONPATH": f"{packages}:{os.environ.get('PYTHONPATH', '')}", "ARENA_DATA_DIR": str(tmp_path / "data")}


def call(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "arena_mcp", "call", *args], env=env, capture_output=True, text=True, check=False, timeout=180, stdin=subprocess.DEVNULL)


@pytest.fixture
def started(env: dict[str, str]) -> Iterator[list[subprocess.Popen[str]]]:
    processes: list[subprocess.Popen[str]] = []
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.communicate()


def start(env: dict[str, str], started: list[subprocess.Popen[str]], *args: str) -> subprocess.Popen[str]:
    process = subprocess.Popen([sys.executable, "-m", "arena_mcp", "call", *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)
    started.append(process)
    return process


def read_until(process: subprocess.Popen[str], marker: str) -> str:
    watchdog = threading.Timer(120, process.kill)
    watchdog.start()
    seen = ""
    try:
        for line in process.stderr:
            seen += line
            if marker in line:
                return seen
    finally:
        watchdog.cancel()
    pytest.fail(f"never printed {marker!r}, stderr:\n{seen}")


def test_values_take_the_type_of_their_parameter(env: dict[str, str]) -> None:
    done = call(env, "demo_echo", "text=5", "count=2", 'tags=["a", "b"]', "shout=true", "note=null")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"text": "5", "count": 2, "tags": ["a", "b"], "shout": True, "note": None}


def test_a_value_keeps_every_equals_sign_after_the_first(env: dict[str, str]) -> None:
    done = call(env, "demo_echo", "text=a=b=c")
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["text"] == "a=b=c"


def test_a_value_of_the_wrong_type_fails_in_validation(env: dict[str, str]) -> None:
    done = call(env, "demo_echo", "text=a", "count=many")
    assert done.returncode == 1
    assert done.stdout == ""
    assert "count" in done.stderr


def test_plain_text_goes_to_stdout_as_is(env: dict[str, str]) -> None:
    done = call(env, "demo_text")
    assert (done.returncode, done.stdout) == (0, "just text\n"), done.stderr


def test_bare_call_says_how_to_find_tools(env: dict[str, str]) -> None:
    done = call(env)
    assert done.returncode == 0
    assert "arena mcp call TOOL [KEY=VALUE ...]" in done.stdout and "arena mcp ls DOMAIN" in done.stdout


def test_tool_help_lists_each_parameter(env: dict[str, str]) -> None:
    done = call(env, "demo_echo", "--help")
    assert done.returncode == 0, done.stderr
    assert "Second line of the description." in done.stdout
    lines = {line.split()[0]: line for line in done.stdout.split("Parameters:")[1].splitlines() if line.strip()}
    assert "string, required" in lines["text"]
    assert "integer, default 1" in lines["count"]
    assert "array of string | null, default null" in lines["tags"]
    assert 'default "kept"' in lines["note"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["demo_ech"], "no tool 'demo_ech', closest: demo_echo"),
        (["nope"], "no tool 'nope'"),
        (["demo"], "'demo' is a domain, its tools:"),
        (["demo_echo", "text"], "'text' is not KEY=VALUE"),
        (["demo_echo", "text=a", "colour=red"], "demo_echo has no parameter 'colour'"),
        (["demo_echo", "text=a", "text=b"], "'text' is given twice"),
        (["demo_echo", "count=1"], "demo_echo needs text="),
    ],
)
def test_bad_usage_exits_2_and_says_what_exists(env: dict[str, str], args: list[str], message: str) -> None:
    done = call(env, *args)
    assert done.returncode == 2
    assert done.stdout == ""
    assert message in done.stderr


def test_tool_error_exits_1_with_its_hint(env: dict[str, str]) -> None:
    done = call(env, "demo_fail")
    assert done.returncode == 1
    assert done.stdout == ""
    assert "broken" in done.stderr and "hint: fix it" in done.stderr


def test_a_tool_asking_for_input_exits_1_and_says_so(env: dict[str, str]) -> None:
    done = call(env, "demo_ask")
    assert done.returncode == 1
    assert done.stdout == ""
    assert "demo_ask asks for more input" in done.stderr


def test_image_result_is_saved_and_its_path_printed(env: dict[str, str]) -> None:
    done = call(env, "demo_picture")
    assert done.returncode == 0, done.stderr
    path = Path(done.stdout.strip())
    assert path.is_relative_to(Path(env["ARENA_DATA_DIR"]) / "mcp") and path.parent.name == "media" and path.suffix == ".png"
    assert path.read_bytes() == b"png bytes"


def test_task_job_runs_to_its_end_with_the_log_on_stderr(env: dict[str, str]) -> None:
    done = call(env, "demo_task")
    assert done.returncode == 0, done.stderr
    assert "task started" in done.stderr
    status = json.loads(done.stdout)
    assert status["state"] == "succeeded" and status["result"] == {"done": True}


def test_failed_task_job_exits_1(env: dict[str, str]) -> None:
    done = call(env, "demo_task", "fail=true")
    assert done.returncode == 1
    status = json.loads(done.stdout)
    assert status["state"] == "failed" and status["error"] == "task broke"


def test_interrupt_cancels_a_task_job(env: dict[str, str], started: list[subprocess.Popen[str]]) -> None:
    process = start(env, started, "demo_task", "hold_s=120")
    read_until(process, "task started")
    process.send_signal(signal.SIGINT)
    stdout, _ = process.communicate(timeout=60)
    assert process.returncode == 130
    assert json.loads(stdout)["state"] == "cancelled"


def test_interrupt_stops_a_process_job(env: dict[str, str], started: list[subprocess.Popen[str]]) -> None:
    process = start(env, started, "demo_process")
    read_until(process, "process started")
    process.send_signal(signal.SIGINT)
    stdout, _ = process.communicate(timeout=60)
    assert process.returncode == 130
    status = json.loads(stdout)
    assert status["state"] == "cancelled"
    with pytest.raises(ProcessLookupError):
        os.kill(status["pid"], 0)


def test_a_second_process_sees_a_running_task_job_and_its_end(env: dict[str, str], started: list[subprocess.Popen[str]]) -> None:
    process = start(env, started, "demo_task", "hold_s=25")
    seen = read_until(process, "following ")
    job_id = seen.rsplit("following ", 1)[1].split(",")[0]
    running = call(env, "jobs_status", f"job_id={job_id}")
    assert running.returncode == 0, running.stderr
    assert json.loads(running.stdout)["state"] == "running"
    waited = call(env, "jobs_wait", f"job_id={job_id}", "timeout_s=120")
    assert waited.returncode == 0, waited.stderr
    assert json.loads(waited.stdout)["state"] == "succeeded"
    stdout, _ = process.communicate(timeout=60)
    assert process.returncode == 0
    assert json.loads(stdout)["state"] == "succeeded"


def test_a_task_job_whose_owner_died_is_lost(env: dict[str, str], started: list[subprocess.Popen[str]]) -> None:
    process = start(env, started, "demo_task", "hold_s=120")
    seen = read_until(process, "following ")
    job_id = seen.rsplit("following ", 1)[1].split(",")[0]
    process.kill()
    process.wait(timeout=30)
    status = call(env, "jobs_status", f"job_id={job_id}")
    assert status.returncode == 0, status.stderr
    assert json.loads(status.stdout)["state"] == "lost"


@pytest.fixture
def subscriber(env: dict[str, str]) -> Iterator[tuple[str, list[str]]]:
    rclpy = pytest.importorskip("rclpy")
    from std_msgs.msg import String

    context = rclpy.Context()
    rclpy.init(context=context)
    name = f"call_test_{os.getpid()}"
    node = rclpy.create_node(name, context=context)
    received: list[str] = []
    node.create_subscription(String, f"/{name}/probe", lambda msg: received.append(msg.data), 10)
    executor = rclpy.executors.SingleThreadedExecutor(context=context)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    yield name, received
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown(context=context)
    spinner.join(timeout=10)


def test_a_fresh_process_sees_the_graph_and_reaches_its_subscribers(env: dict[str, str], subscriber: tuple[str, list[str]]) -> None:
    name, received = subscriber
    graph = call(env, "ros_graph")
    assert graph.returncode == 0, graph.stderr
    assert f"/{name}" in json.loads(graph.stdout)["nodes"]
    published = call(env, "ros_publish", f"topic=/{name}/probe", "type_name=std_msgs/msg/String", 'message={"data": "hello"}')
    assert published.returncode == 0, published.stderr
    assert json.loads(published.stdout)["subscribers"] == 1
    deadline = time.monotonic() + 10
    while not received and time.monotonic() < deadline:
        time.sleep(0.05)
    assert received == ["hello"]


def test_completion_values_carry_tools_and_keys(env: dict[str, str]) -> None:
    script = "import json; from arena_mcp.call import completion_values; print(json.dumps(completion_values()))"
    done = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False, timeout=180)
    assert done.returncode == 0, done.stderr
    values, deps = json.loads(done.stdout.splitlines()[-1])
    assert values["mcp.tools"]["demo_echo"] == "Echo the arguments."
    assert values["mcp.args.demo_echo"]["count="] == "integer"
    assert any(dep.endswith("arena_call_demo.py") for dep in deps)
