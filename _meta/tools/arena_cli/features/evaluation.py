"""evaluation feature: recording, metrics, benchmarking."""

import common
from common import Verb, make_verb

from features import lifecycle_verbs

NAME = "evaluation"

DESCRIPTION = "arena_evaluation for recording, metrics, and benchmarking."


def _update() -> int:
    """Pull the arena_evaluation submodule and rebuild."""
    import subprocess

    rc = subprocess.run(
        ["git", "submodule", "update", "--init", "--rebase", "--depth", "1", "arena_evaluation"],
        cwd=common._env("ARENA_DIR"),
        check=False,
    ).returncode
    if rc:
        return rc

    return common._resourced("arena build")


def benchmark(argv: list[str]) -> None:
    """run a benchmark suite (ros2 run arena_evaluation benchmark)"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "benchmark", *argv)


def list_(argv: list[str]) -> None:
    """list available evaluations"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "list", *argv)


def status(argv: list[str]) -> None:
    """show status of a running or completed evaluation"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "status", *argv)


def tail(argv: list[str]) -> None:
    """stream live output of a running evaluation"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "tail", *argv)


def ps_(argv: list[str]) -> None:
    """list running arena processes (benchmark runner, sim, nodes)"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "ps", *argv)


def console(argv: list[str]) -> None:
    """tail a benchmark run's console log (MCP-launched runs only)"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation_cli", "console", *argv)


def acoustic(argv: list[str]) -> None:
    """acoustic field visualization (list, animate, snapshot)"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "acoustic", *argv)


def extract(argv: list[str]) -> None:
    """extract MCAP topics into the Parquet cache"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "extract", *argv)


def run(argv: list[str]) -> None:
    """process recording and generate HTML report"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "run", *argv)


def process(argv: list[str]) -> None:
    """process recording to generate metrics.parquet"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "process", *argv)


def report(argv: list[str]) -> None:
    """generate HTML report from processed metrics"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "report", *argv)


def plot(argv: list[str]) -> None:
    """generate static PNG plots from processed metrics"""
    common._reg_require(NAME)
    common._exec("ros2", "run", "arena_evaluation", "evaluation", "plot", *argv)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        *lifecycle_verbs(NAME, _update, deinit="arena_evaluation"),
        make_verb("benchmark", benchmark, passthrough=True),
        make_verb("list", list_, passthrough=True),
        make_verb("status", status, passthrough=True),
        make_verb("tail", tail, passthrough=True),
        make_verb("ps", ps_, passthrough=True),
        make_verb("console", console, passthrough=True),
        make_verb("extract", extract, passthrough=True),
        make_verb("run", run, passthrough=True),
        make_verb("process", process, passthrough=True),
        make_verb("report", report, passthrough=True),
        make_verb("plot", plot, passthrough=True),
        make_verb("acoustic", acoustic, passthrough=True),
    ]
}
