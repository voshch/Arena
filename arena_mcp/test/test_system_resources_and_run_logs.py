from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from arena_mcp.core import ToolError
from arena_mcp.domains.system import system_logs, system_resources


def _log(path: Path, text: str, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (mtime, mtime))
    os.utime(path.parent, (mtime, mtime))
    return path


@pytest.fixture
def benchmarks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ARENA_DATA_DIR", str(tmp_path))
    return tmp_path / "benchmarks"


def test_lane_roslogs_are_listed_and_searched(benchmarks: Path) -> None:
    now = time.time()
    _log(benchmarks / "shared" / "roslog.1" / "arena_env_0_1_2.log", "lane one\n", now - 20)
    _log(benchmarks / "shared" / "roslog.2" / "arena_env_0_3_4.log", "lane two\n", now - 10)
    resources = system_resources(run_id="shared")
    assert resources["run_id"] == "shared"
    assert resources["roslog_dirs"] == [str(benchmarks / "shared" / "roslog.1"), str(benchmarks / "shared" / "roslog.2")]
    assert resources["recent_roslogs"] == ["roslog.2/arena_env_0_3_4.log", "roslog.1/arena_env_0_1_2.log"]
    logs = system_logs(node="arena_env_0", run_id="shared", lines=10)
    assert logs["path"] == str(benchmarks / "shared" / "roslog.2" / "arena_env_0_3_4.log")
    assert logs["log"] == "lane two"


def test_resources_default_to_the_run_whose_roslogs_changed_last(benchmarks: Path) -> None:
    now = time.time()
    _log(benchmarks / "old" / "roslog" / "a.log", "", now - 100)
    _log(benchmarks / "new" / "roslog" / "b.log", "", now - 5)
    (benchmarks / "no_logs").mkdir()
    resources = system_resources()
    assert resources["run_id"] == "new"
    assert resources["recent_roslogs"] == ["roslog/b.log"]


def test_resources_without_benchmarks_have_no_run(benchmarks: Path) -> None:
    resources = system_resources()
    assert resources["run_id"] is None
    assert resources["roslog_dirs"] == []


@pytest.mark.parametrize("run_id", ["..", ".", "", "a/b", "../benchmarks"])
def test_run_ids_outside_the_benchmarks_root_are_refused(benchmarks: Path, run_id: str) -> None:
    with pytest.raises(ToolError, match="not a run id"):
        system_resources(run_id=run_id)
    with pytest.raises(ToolError, match="not a run id"):
        system_logs(node="arena_env_0", run_id=run_id)


def test_missing_runs_and_nodes_are_named(benchmarks: Path) -> None:
    _log(benchmarks / "run" / "roslog" / "arena_env_0_1_2.log", "x\n", time.time())
    with pytest.raises(ToolError, match="no roslog directory for run 'absent'"):
        system_resources(run_id="absent")
    with pytest.raises(ToolError, match="no roslog directory for run 'absent'"):
        system_logs(node="arena_env_0", run_id="absent")
    with pytest.raises(ToolError, match="no log of planner_server in the roslogs of run run"):
        system_logs(node="planner_server", run_id="run")
    with pytest.raises(ToolError, match="run_id needs node"):
        system_logs(run_id="run")


def test_node_logs_without_run_id_stay_in_the_ros_log_dir(benchmarks: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("rclpy")
    _log(benchmarks / "run" / "roslog" / "arena_env_0_1_2.log", "benchmark\n", time.time())
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path / "ros_log"))
    (tmp_path / "ros_log").mkdir()
    with pytest.raises(ToolError, match=f"under {tmp_path / 'ros_log'}"):
        system_logs(node="arena_env_0")


def test_stack_processes_match_by_program_and_list_the_outermost(tmp_path: Path) -> None:
    node = tmp_path / "node.py"
    node.write_text("import time\ntime.sleep(60)\n")
    launch = tmp_path / "launch.py"
    started = tmp_path / "started"
    launch.write_text(f"import subprocess, sys, time\nsubprocess.Popen([sys.executable, {str(node)!r}, '--ros-args'])\nopen({str(started)!r}, 'w').close()\ntime.sleep(60)\n")
    package = tmp_path / "modules" / "arena_bringup"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "supervisor.py").write_text("import time\ntime.sleep(60)\n")
    pipe = tmp_path / "arena_runtime" / "README.md"
    pipe.parent.mkdir()
    os.mkfifo(pipe)
    stack = subprocess.Popen([sys.executable, str(launch), "--ros-args"], start_new_session=True)
    supervisor = subprocess.Popen([sys.executable, "-m", "arena_bringup.supervisor"], cwd=package.parent)
    bystanders = [
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "/opt/arena_runtime/controller_server.log"]),
        subprocess.Popen(["cat", str(pipe)]),
    ]
    try:
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert started.exists()
        resources = system_resources()
        listed = {entry["pid"]: entry["cmdline"] for entry in resources["arena_processes"]}
        assert stack.pid in listed
        assert supervisor.pid in listed
        assert not any(bystander.pid in listed for bystander in bystanders)
        assert not any(line.startswith(f"{sys.executable} {node}") for line in listed.values())
        assert resources["process_count"] >= len(listed) + 1
    finally:
        os.killpg(stack.pid, signal.SIGKILL)
        for process in [supervisor, *bystanders]:
            process.kill()
        for process in [stack, supervisor, *bystanders]:
            process.wait()
