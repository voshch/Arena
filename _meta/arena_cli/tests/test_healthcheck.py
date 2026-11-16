"""Tests for the install checks, the step rows, the simulator selection and the argument parsing of arena healthcheck."""

import io
import threading
from pathlib import Path

import pytest

from arena_cli.common import CLIError
from arena_cli.healthcheck import GREEN, RED, RESET, SPINNER, YELLOW, Phase, _parse, boot, build, row, shell, sims, step, venv, verdict

ARENA = Path(__file__).resolve().parents[3]


def test_sims_takes_every_installed_simulator_in_launch_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", str(ARENA))
    assert sims(["mujoco", "planners", "gazebo"], ["robot:=jackal"]) == (["gazebo", "mujoco"], ["robot:=jackal"])


def test_sims_launches_only_a_named_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", str(ARENA))
    assert sims(["gazebo", "mujoco"], ["sim:=flatsim", "headless:=true"]) == (["flatsim"], ["headless:=true"])


def test_sims_is_empty_without_an_installed_simulator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_DIR", str(ARENA))
    assert sims(["planners", "robots"], []) == ([], [])


def test_verdict_names_the_runner_exit_codes() -> None:
    assert verdict(0) == "ok"
    assert verdict(3) == "wedged or weak"
    assert verdict(4) == "failed (exit 4)"


def test_shell_passes_inside_the_container() -> None:
    assert shell({"ARENA_CONTAINER": "1"}) == (True, "ok")


def test_shell_outside_the_container_names_source_arena() -> None:
    ok, detail = shell({})
    assert not ok
    assert "source arena" in detail


def test_boot_passes_once_the_marker_exists(tmp_path: Path) -> None:
    marker = tmp_path / ".built"
    marker.write_text("")
    assert boot(marker) == (True, "ok")


def test_boot_fails_while_first_boot_runs(tmp_path: Path) -> None:
    ok, detail = boot(tmp_path / ".built")
    assert not ok
    assert "Initial setup complete" in detail


def test_venv_passes_when_provisioned(tmp_path: Path) -> None:
    (tmp_path / ".installed").write_text("")
    assert venv(tmp_path) == (True, "ok")


def test_venv_unprovisioned_names_arena_repair(tmp_path: Path) -> None:
    ok, detail = venv(tmp_path)
    assert not ok
    assert "arena repair" in detail


def test_build_passes_with_nothing_stale() -> None:
    assert build([]) == (True, "ok, nothing stale")


def test_build_lists_stale_packages_and_names_arena_build() -> None:
    assert build(["a", "b"]) == (False, "2 stale package(s): a b, run 'arena build'")


def test_build_truncates_a_long_stale_list() -> None:
    ok, detail = build([f"p{i}" for i in range(11)])
    assert not ok
    assert detail == "11 stale package(s): p0 p1 p2 p3 p4 p5 p6 p7 and 3 more, run 'arena build'"


def test_parse_splits_the_launch_flag_from_launch_arguments() -> None:
    assert _parse([]) == (True, [])
    assert _parse(["--no-launch", "sim:=gazebo"]) == (False, ["sim:=gazebo"])


def test_parse_rejects_unknown_arguments_naming_the_accepted_ones() -> None:
    with pytest.raises(CLIError, match="--no-launch or KEY:=VALUE"):
        _parse(["--fast"])


class Terminal(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.drawn = threading.Event()

    def isatty(self) -> bool:
        return True

    def write(self, text: str) -> int:
        self.drawn.set()
        return super().write(text)


def test_step_off_a_terminal_prints_only_the_plain_final_row() -> None:
    out = io.StringIO()
    assert step("venv", lambda: (True, "ok"), "checking", out)
    assert out.getvalue() == row("venv", "OK") + "\n"
    assert not step("build", lambda: (False, "2 stale package(s)"), "", out)
    assert out.getvalue().splitlines()[-1] == "build           [FAIL] 2 stale package(s)"


def test_step_on_a_terminal_spins_then_replaces_the_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    out = Terminal()

    def work() -> tuple[bool, str]:
        assert out.drawn.wait(5)
        return False, "wedged or weak"

    assert not step("launch gazebo", work, "preflight", out)
    drawn = out.getvalue()
    assert row("launch gazebo", SPINNER[0], "preflight", YELLOW) in drawn
    assert drawn.endswith("\r\033[K" + row("launch gazebo", "FAIL", "wedged or weak", RED) + "\n")


def test_step_on_a_terminal_colors_ok_green(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm")
    out = Terminal()
    assert step("shell", lambda: (True, "ok"), "", out)
    assert out.getvalue().endswith(f"[{GREEN}OK{RESET}]\n")


def test_step_honors_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm")
    out = Terminal()
    step("shell", lambda: (True, "ok"), "", out)
    assert "\033[3" not in out.getvalue()
    assert out.getvalue().endswith(row("shell", "OK") + "\n")


def test_phase_follows_the_benchmark_log_from_bringup_to_shutdown(tmp_path: Path) -> None:
    log = tmp_path / "gazebo.log"
    now = [100.0]
    phase = Phase(log, lambda: now[0])
    assert phase() == "starting (0s)"
    log.write_bytes(b"benchmark: prepared run_id=x steps=1\n[16:41:53] [INFO] [runner]: Starting step default/preflight (env=0, 2 episodes)\n")
    now[0] = 131.7
    assert phase() == "starting (31s)"
    with log.open("ab") as f:
        f.write(b"[env 0] episode record: id=1 outcome=0 (QUEUED) info=''\n[env 0] episode record: id=1 outcome=1 (RUNNING) info=''\n")
    assert phase() == "episode 1/2 (31s)"
    with log.open("ab") as f:
        f.write(b"[env 0] episode record: id=1 outcome=2 (SUCCESS) info=''\n[env 0] episode record: id=2 outcome=1 (RUN")
    assert phase() == "episode 1/2 (31s)"
    with log.open("ab") as f:
        f.write(b"NING) info=''\n")
    assert phase() == "episode 2/2 (31s)"
    with log.open("ab") as f:
        f.write(b"benchmark: shutting down arena runtime...\n")
    assert phase() == "shutting down (31s)"


def test_step_on_a_terminal_redraws_a_live_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm")
    out = Terminal()
    assert step("launch mujoco", lambda: (out.drawn.wait(5), (True, "ok"))[1], lambda: "episode 1/2 (7s)", out)
    assert row("launch mujoco", SPINNER[0], "episode 1/2 (7s)") in out.getvalue()
