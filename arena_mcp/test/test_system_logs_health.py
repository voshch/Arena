from __future__ import annotations

import os
from pathlib import Path

from arena_mcp.domains.system import system_health, system_logs


def test_system_health_and_logs_read_benchmark_roslog_dir(tmp_path: Path) -> None:
    prev = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    try:
        roslog = tmp_path / "benchmarks" / "demo_run" / "roslog"
        roslog.mkdir(parents=True)
        log_file = roslog / "arena_env_0_123_456.log"
        log_file.write_text("line1\nGazebo control timeout\n", encoding="utf-8")

        health = system_health(run_id="demo_run")
        assert health["roslog_dir"] == str(roslog)
        assert "arena_env_0_123_456.log" in health["recent_roslogs"]
        assert isinstance(health["process_count"], int)

        log_res = system_logs(node="arena_env_0", run_id="demo_run", lines=10)
        assert log_res["run_id"] == "demo_run"
        assert "Gazebo control timeout" in log_res["log"]
    finally:
        if prev is None:
            os.environ.pop("ARENA_DATA_DIR", None)
        else:
            os.environ["ARENA_DATA_DIR"] = prev
