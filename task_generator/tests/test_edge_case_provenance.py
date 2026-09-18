from __future__ import annotations

import json

from task_generator.tasks.obstacles.edge_case.provenance import CaseLog, CaseRecord, default_record_dir


def _record(**kwargs) -> CaseRecord:
    base = {"run_seed": "abc123", "episode_id": 3, "world": "arena_arena_002", "seed": 99}
    return CaseRecord(**{**base, **kwargs})


def test_record_serialises_to_one_json_line() -> None:
    line = _record(case_id="h_004", steps=("A:blackout",), effects=({"type": "rally"},)).to_json()
    assert "\n" not in line
    doc = json.loads(line)
    assert doc["case_id"] == "h_004" and doc["steps"] == ["A:blackout"] and doc["effects"][0]["type"] == "rally"


def test_log_appends_rather_than_overwrites(tmp_path) -> None:
    log = CaseLog(tmp_path)
    log.write(_record(episode_id=1))
    log.write(_record(episode_id=2))
    assert [r["episode_id"] for r in log.read_all()] == [1, 2]


def test_log_creates_its_directory(tmp_path) -> None:
    assert CaseLog(tmp_path / "nested" / "deeper").write(_record()).is_file()


def test_read_all_on_missing_file_is_empty(tmp_path) -> None:
    assert CaseLog(tmp_path / "absent").read_all() == []


def test_baseline_and_aborted_are_distinguishable(tmp_path) -> None:
    log = CaseLog(tmp_path)
    log.write(_record(status="baseline"))
    log.write(_record(status="aborted", reason="no free point on the robot's route"))
    base, aborted = log.read_all()
    assert base["status"] == "baseline" and base["effects"] == []
    assert aborted["status"] == "aborted" and "free point" in aborted["reason"]


def test_geometry_survives_serialisation(tmp_path) -> None:
    log = CaseLog(tmp_path)
    log.write(_record(injected=1, spawn_pose=(1.0, 2.0), ped_goal=(3.0, 4.0), robot_leg=((0.0, 0.0), (9.0, 0.0)), plans=({"mode": "hold"},)))
    (row,) = log.read_all()
    assert row["spawn_pose"] == [1.0, 2.0] and row["robot_leg"] == [[0.0, 0.0], [9.0, 0.0]] and row["plans"][0]["mode"] == "hold"


def test_default_record_dir_follows_arena_data_dir(monkeypatch) -> None:
    monkeypatch.setenv("ARENA_DATA_DIR", "/opt/arena_ws/data")
    assert default_record_dir().as_posix() == "/opt/arena_ws/data/edge_case"


def test_default_record_dir_falls_back_without_the_env(monkeypatch) -> None:
    monkeypatch.delenv("ARENA_DATA_DIR", raising=False)
    assert default_record_dir().name == "arena_edge_case"
