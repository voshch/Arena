from __future__ import annotations

import pytest

import arena_bringup

GETTERS = [("ARENA_WS_DIR", "get_arena_ws_dir"), ("ARENA_DIR", "get_arena_dir")]


@pytest.mark.parametrize(("variable", "getter"), GETTERS)
@pytest.mark.parametrize("value", ["/tmp/arena", "/some/path with spaces/ws", ""])
def test_getter_returns_the_exact_value(monkeypatch: pytest.MonkeyPatch, variable: str, getter: str, value: str) -> None:
    monkeypatch.setenv(variable, value)
    assert getattr(arena_bringup, getter)() == value


@pytest.mark.parametrize(("variable", "getter"), GETTERS)
def test_getter_raises_naming_the_missing_variable(monkeypatch: pytest.MonkeyPatch, variable: str, getter: str) -> None:
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(AssertionError, match=variable):
        getattr(arena_bringup, getter)()
