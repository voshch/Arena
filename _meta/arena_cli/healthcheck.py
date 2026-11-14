"""arena healthcheck: whether this install, every installed feature and a launch in every installed simulator work."""

import contextlib
import functools
import io
import itertools
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TextIO

from arena_cli import common, evaluation, features, settings
from arena_cli.build import stale_packages, workspace
from arena_cli.common import CLIError, make_verb
from arena_cli.complete import Flags, LaunchArgs, Union

BUILT = Path("/.built")
SHOWN_STALE = 8
LOG_TAIL = 15
WEAK_OR_WEDGED = 3
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
FRAME_S = 0.1
LABEL_WIDTH = 15
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"

Step = features.Health


def row(label: str, tag: str, text: str = "", color: str = "") -> str:
    shown = f"{color}{tag}{RESET}" if color else tag
    return " ".join(part for part in (f"{label:<{LABEL_WIDTH}}", f"[{shown}]", text) if part)


class Phase:
    """The phase of a running preflight, read incrementally from its benchmark log."""

    TOTAL = re.compile(rb"Starting step .* \(env=\d+, (\d+) episodes\)")
    EPISODE = re.compile(rb"episode record: id=(\d+) outcome=1 \(RUNNING\)")
    SHUTDOWN = b"benchmark: shutting down arena runtime"

    def __init__(self, log: Path, clock: Callable[[], float] = time.monotonic) -> None:
        self.log = log
        self.clock = clock
        self.start = clock()
        self.offset = 0
        self.total = 0
        self.label = "starting"

    def __call__(self) -> str:
        with contextlib.suppress(OSError), self.log.open("rb") as f:
            f.seek(self.offset)
            chunk = f.read()
            complete = chunk.rfind(b"\n") + 1
            self.offset += complete
            for line in chunk[:complete].splitlines():
                self._feed(line)
        return f"{self.label} ({int(self.clock() - self.start)}s)"

    def _feed(self, line: bytes) -> None:
        if total := self.TOTAL.search(line):
            self.total = int(total.group(1))
        elif episode := self.EPISODE.search(line):
            self.label = f"episode {int(episode.group(1))}/{self.total}"
        elif self.SHUTDOWN in line:
            self.label = "shutting down"


def step(label: str, work: Callable[[], Step], text: str | Callable[[], str] = "", stream: TextIO | None = None) -> bool:
    """Run `work` under a spinner row on a terminal and leave `[OK]` or `[FAIL] reason` in its place."""
    out = stream or sys.stdout
    live = out.isatty() and os.environ.get("TERM") != "dumb"
    color = live and not os.environ.get("NO_COLOR")
    done = threading.Event()

    def spin() -> None:
        for frame in itertools.cycle(SPINNER):
            out.write("\r\033[K" + row(label, frame, text() if callable(text) else text, YELLOW if color else ""))
            out.flush()
            if done.wait(FRAME_S):
                return

    spinner = threading.Thread(target=spin, daemon=True)
    if live:
        spinner.start()
    try:
        ok, detail = work()
    finally:
        done.set()
        if live:
            spinner.join()
            out.write("\r\033[K")
    print(row(label, "OK", "", GREEN if color else "") if ok else row(label, "FAIL", detail, RED if color else ""), file=out, flush=True)
    return ok


def shell(env: dict[str, str]) -> Step:
    if env.get("ARENA_CONTAINER"):
        return True, "ok"
    return False, "not in the arena container, enter it with 'source arena'"


def boot(marker: Path) -> Step:
    if marker.is_file():
        return True, "ok"
    return False, "first boot is still running, wait for 'Initial setup complete' in the container log"


def venv(venv_dir: Path) -> Step:
    if (venv_dir / ".installed").is_file():
        return True, "ok"
    return False, f"{venv_dir} is not provisioned, run 'arena repair'"


def build(stale: list[str]) -> Step:
    if not stale:
        return True, "ok, nothing stale"
    shown = " ".join(stale[:SHOWN_STALE])
    more = f" and {len(stale) - SHOWN_STALE} more" if len(stale) > SHOWN_STALE else ""
    return False, f"{len(stale)} stale package(s): {shown}{more}, run 'arena build'"


def sims(installed: list[str], tokens: list[str]) -> tuple[list[str], list[str]]:
    """The simulators to launch, a sim:= token alone or every installed one, and the tokens without it."""
    chosen = [token.split(":=", 1)[1] for token in tokens if token.startswith("sim:=")]
    rest = [token for token in tokens if not token.startswith("sim:=")]
    return chosen or settings._defaults().available_sims(installed), rest


def verdict(rc: int) -> str:
    if rc == 0:
        return "ok"
    if rc == WEAK_OR_WEDGED:
        return "wedged or weak"
    return f"failed (exit {rc})"


def _parse(argv: list[str]) -> tuple[bool, list[str]]:
    launch, tokens = True, []
    for arg in argv:
        if arg == "--no-launch":
            launch = False
        elif ":=" in arg:
            tokens.append(arg)
        else:
            raise CLIError(f"unknown argument '{arg}', expected --no-launch or KEY:=VALUE launch arguments")
    return launch, tokens


def _launch(sim: str, tokens: list[str]) -> int:
    from arena_cli.features import planners

    log = Path(common._env("ARENA_WS_DIR")) / "log" / "healthcheck" / f"{sim}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = evaluation.benchmark_argv(["--suite", json.dumps(planners.preflight_suite()), "--contest", json.dumps([{"name": "default"}]), *planners.PREFLIGHT_VERDICT, f"sim:={sim}", *tokens])
    rc = 0

    def preflight() -> Step:
        nonlocal rc
        with log.open("w") as out:
            rc = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT, check=False).returncode
        return rc == 0, f"{verdict(rc)}, log {log}"

    if not step(f"launch {sim}", preflight, Phase(log)):
        print("\n".join(log.read_text(errors="replace").splitlines()[-LOG_TAIL:]))
    return rc


def _quiet(health: Callable[[], Step]) -> Step:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return health()


def run(argv: list[str]) -> None:
    """Check that this install, every installed feature and a launch in every installed simulator work.

    `arena healthcheck [--no-launch] [KEY:=VALUE ...]`
    Install checks, in order and stopping at the first that fails: shell (inside the arena container),
    boot (first boot finished), venv (the python venv is provisioned), build (no package is stale for `arena build`).
    Then one check per installed feature (its submodules, repos, containers or definitions), each naming the command that fixes it.
    Then, when the install checks pass, two episodes of map_empty's `preflight` scenario in every installed simulator
    through the benchmark runner, verdict wedged/weak/ok, each run logged under log/healthcheck/.
    `--no-launch`: skip the launches. `sim:=<sim>` launches that simulator alone, other KEY:=VALUE tokens go to every launch.
    Exit 0 when everything passes, 1 on a failed check, 3 on a wedged or weak launch.
    """
    launch, tokens = _parse(argv)
    checks: list[tuple[str, Callable[[], Step]]] = [
        ("shell", lambda: shell(dict(os.environ))),
        ("boot", lambda: boot(BUILT)),
        ("venv", lambda: venv(Path(common._env("ARENA_VENV_DIR")))),
        ("build", lambda: build([package for package, _ in stale_packages(workspace())])),
    ]
    for label, check in checks:
        if not step(label, check):
            sys.exit(1)
    installed = common._reg_list()
    healthy = True
    for name in installed:
        module = features.load(name)
        if module is None:
            continue
        healthy = step(name, functools.partial(_quiet, module.health)) and healthy
    codes = [0 if healthy else 1]
    targets, tokens = sims(installed, tokens)
    if launch and not targets:
        codes.append(int(not step("launch", lambda: (False, "no simulator installed, install one with 'arena feature gazebo install'"))))
    elif launch:
        codes += [_launch(sim, tokens) for sim in targets]
    sys.exit(WEAK_OR_WEDGED if WEAK_OR_WEDGED in codes else max(codes))


VERB = make_verb(
    "healthcheck",
    run,
    passthrough=True,
    complete=Union(Flags({"--no-launch": "skip the preflight launches"}), LaunchArgs("arena_bringup", "arena_runtime.launch.py")),
)
