from __future__ import annotations

import sys
from pathlib import Path

import pytest

from arena_runtime.env.task_layer import TASK_LAYER_GROUP, task_layer, task_layer_entry_point


def _register(root: Path, distribution: str, value: str) -> None:
    info = root / f"{distribution}-0.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.0.0\n")
    (info / "entry_points.txt").write_text(f"[{TASK_LAYER_GROUP}]\n{distribution} = {value}\n")


def test_the_one_registered_entry_point_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "fake_tasks.py").write_text("MARK = 1\n")
    _register(tmp_path, "fake_tasks", "fake_tasks")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    assert task_layer().MARK == 1


def test_no_registered_entry_point_names_the_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    with pytest.raises(LookupError, match=f"{TASK_LAYER_GROUP}.*none"):
        task_layer_entry_point()


def test_two_registered_entry_points_are_listed_instead_of_picked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _register(tmp_path, "first", "first")
    _register(tmp_path, "second", "second")
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    with pytest.raises(LookupError, match="first.*second"):
        task_layer_entry_point()
