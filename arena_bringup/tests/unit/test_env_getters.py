from __future__ import annotations

import pytest


def test_get_arena_ws_dir_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_WS_DIR", "/tmp/arena_ws")
    from arena_bringup import get_arena_ws_dir

    assert get_arena_ws_dir() == "/tmp/arena_ws"


def test_get_arena_dir_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", "/tmp/arena")
    from arena_bringup import get_arena_dir

    assert get_arena_dir() == "/tmp/arena"


def test_get_arena_ws_dir_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARENA_WS_DIR", raising=False)
    from arena_bringup import get_arena_ws_dir

    with pytest.raises(AssertionError, match="ARENA_WS_DIR"):
        get_arena_ws_dir()


def test_get_arena_dir_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARENA_DIR", raising=False)
    from arena_bringup import get_arena_dir

    with pytest.raises(AssertionError, match="ARENA_DIR"):
        get_arena_dir()


def test_get_arena_ws_dir_returns_exact_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_WS_DIR", "/some/path with spaces/ws")
    from arena_bringup import get_arena_ws_dir

    assert get_arena_ws_dir() == "/some/path with spaces/ws"


def test_get_arena_dir_returns_exact_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", "/some/other/path")
    from arena_bringup import get_arena_dir

    assert get_arena_dir() == "/some/other/path"


def test_get_arena_ws_dir_empty_string_is_not_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_WS_DIR", "")
    from arena_bringup import get_arena_ws_dir

    assert get_arena_ws_dir() == ""


def test_get_arena_dir_empty_string_is_not_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", "")
    from arena_bringup import get_arena_dir

    assert get_arena_dir() == ""
