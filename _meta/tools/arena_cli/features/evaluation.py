"""evaluation feature: recording, metrics, benchmarking."""

import os
import sys

import common
from common import CLIError, Verb, make_verb

SCRIPT_SHA256 = "cd5e8276c1a82d186e77767a0081594de5b60b885cdf0ddecae7b5d824655d6f"

NAME = "evaluation"

DESCRIPTION = "arena_evaluation for recording, metrics, and benchmarking."


def _require() -> None:
    if not common._reg_has(NAME):
        raise CLIError(f"{NAME} is not installed; run 'arena feature {NAME} install' first.")


def _update() -> int:
    """Pull the arena_evaluation submodule and rebuild."""
    import shlex
    import subprocess

    arena_dir = common._env("ARENA_DIR")
    arena_ws_dir = common._env("ARENA_WS_DIR")

    rc = subprocess.run(
        ["git", "submodule", "update", "--init", "--rebase", "--depth", "1", "arena_evaluation"],
        cwd=arena_dir,
        check=False,
    ).returncode
    if rc:
        return rc

    src = common._env("SOURCE_FILE")
    cmd = f"source {shlex.quote(src)} > /dev/null 2>&1 && arena build"
    return subprocess.run(
        [os.environ.get("SHELL", "/bin/bash"), "-c", cmd],
        cwd=arena_ws_dir,
        check=False,
    ).returncode


def install(argv: list[str]) -> None:
    """register feature, pull arena_evaluation submodule and rebuild"""
    if argv:
        raise CLIError("unexpected arguments")
    from features import default_install

    sys.exit(default_install(NAME, _update))


def update(argv: list[str]) -> None:
    """pull arena_evaluation submodule and rebuild"""
    if argv:
        raise CLIError("unexpected arguments")
    if not common._reg_has(NAME):
        raise CLIError(f"{NAME} is not installed, run 'arena feature {NAME} install' first")
    sys.exit(_update())


def uninstall(argv: list[str]) -> None:
    """deinit arena_evaluation and remove from registry"""
    if argv:
        raise CLIError("unexpected arguments")
    import subprocess

    arena_dir = common._env("ARENA_DIR")
    subprocess.run(["git", "submodule", "deinit", "-f", "arena_evaluation"], cwd=arena_dir, check=False)
    common._reg_remove(NAME)


def benchmark(argv: list[str]) -> None:
    """run a benchmark suite (ros2 run arena_evaluation benchmark)"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "benchmark", *argv)


def list_(argv: list[str]) -> None:
    """list available evaluations"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "list", *argv)


def status(argv: list[str]) -> None:
    """show status of a running or completed evaluation"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "status", *argv)


def tail(argv: list[str]) -> None:
    """stream live output of a running evaluation"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "tail", *argv)


def extract(argv: list[str]) -> None:
    """extract MCAP topics into the Parquet cache"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "extract", *argv)


def run(argv: list[str]) -> None:
    """process recording and generate HTML report"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "run", *argv)


def process(argv: list[str]) -> None:
    """process recording to generate metrics.parquet"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "process", *argv)


def report(argv: list[str]) -> None:
    """generate HTML report from processed metrics"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "report", *argv)


def plot(argv: list[str]) -> None:
    """generate static PNG plots from processed metrics"""
    _require()
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "plot", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("install", install),
        make_verb("update", update),
        make_verb("uninstall", uninstall),
        make_verb("benchmark", benchmark, passthrough=True),
        make_verb("list", list_, passthrough=True),
        make_verb("status", status, passthrough=True),
        make_verb("tail", tail, passthrough=True),
        make_verb("extract", extract, passthrough=True),
        make_verb("run", run, passthrough=True),
        make_verb("process", process, passthrough=True),
        make_verb("report", report, passthrough=True),
        make_verb("plot", plot, passthrough=True),
    ]
}
