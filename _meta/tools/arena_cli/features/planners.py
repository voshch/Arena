"""planners feature: per-planner submodule management."""

import os
import sys

import common
from common import Verb, make_verb
import complete
from complete import Flags, Static, Union

import features

_NAME = "planners"

DESCRIPTION = (
    "Per-planner submodule management.\n\n"
    "\b\n"
    "  add <name...>        clone planner's submodules (alias: install)\n"
    "  rm <name...|--all>   deinit planner's submodules (alias: uninstall <name...>)\n"
    "  ls                   list planners, [x] ready, [ ] pending\n"
    "  check [--all]        verify planner submodules are initialized\n"
    "  update               refresh initialized planner submodules\n"
    "  uninstall            deinit all planner submodules\n"
    "  test [--preflight] <name...|--all>  lockstep soak, or a short start-and-drive check, via the benchmark runner (needs the evaluation feature)"
)


# one short crowded stage per contestant with the scheduler stepping the sim, so a
# planner that cannot keep its beat shows up as a stall in the run report
_SOAK_SUITE = {
    "launch": {"lockstep": True, "lockstep.paused": False, "headless": True},
    "references": False,
    "stages": [
        {
            "name": "soak",
            "map": "map_empty",
            "robot": "auto",
            "episodes": 2,
            "tm_robots": "random",
            "tm_obstacles": "random",
            "config": {
                "random": {
                    "dynamic": {"min": 4, "max": 6, "models": ["arenian"]},
                    "static": {"min": 3, "max": 6, "models": ["shelf"]},
                    "interactive": {"min": 0, "max": 0},
                }
            },
            "timeout": "60s",
        }
    ],
}


_PREFLIGHT_STAGE = {
    "name": "preflight",
    "map": "map_empty",
    "robot": "auto",
    "episodes": 2,
    "tm_robots": "scenario",
    "tm_obstacles": "scenario",
    "config": {"scenario": {"file": "preflight"}},
    "timeout": "20s",
}


def _payload() -> str:
    return os.path.join(features.assets_dir(_NAME), f"{_NAME}.py")


def _deps_build() -> int:
    return common._resourced("arena deps && arena build --executor sequential")


def _payload_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_planners_payload", _payload())
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _names() -> list[str]:
    from pathlib import Path

    return sorted(_payload_module().planner_submodules(Path(common._env("ARENA_DIR"))))


def _ready_names() -> list[str]:
    """Registered planners whose submodules are all initialized."""
    from pathlib import Path

    mod = _payload_module()
    arena = Path(common._env("ARENA_DIR"))
    status = mod.submodule_status(arena)
    return sorted(name for name, paths in mod.planner_submodules(arena).items() if all(status.get(p) == "init" for p in paths))


def _contestants(names: list[str]) -> list[dict]:
    """Bridge planners run under the drl adapter, anything else is a nav2 local planner."""
    from pathlib import Path

    kinds = _payload_module().planner_kinds(Path(common._env("ARENA_DIR")))
    out = []
    for name in names:
        if kinds.get(name, "nav2") == "bridge":
            out.append({"name": name, "mobile": {"driver": "drl", "planner": name}})
        else:
            out.append({"name": name, "mobile": {"driver": "nav2", "local_planner": name}})
    return out


_ALL = Flags({"--all": "every planner"})
_PREFLIGHT = Flags({"--preflight": "short start-and-drive check instead of the soak"})
_SELECT = Union(Static(_names), _ALL)


def _forward(verb: str, args: list[str]) -> int:
    import subprocess

    os.environ.setdefault("GIT_SSH_COMMAND", common._git_ssh_command())
    rc = subprocess.run(["python3", _payload(), verb, *args], check=False).returncode
    if verb in ("add", "rm", "update", "uninstall"):
        complete.invalidate(common._env("ARENA_WS_DIR"))
    return rc


def add(argv: list[str]) -> None:
    rc = _forward("add", argv)
    sys.exit(_deps_build() or rc)


def update(argv: list[str]) -> None:
    common._reg_require(_NAME)
    rc = _forward("update", argv)
    sys.exit(_deps_build() or rc)


def rm(argv: list[str]) -> None:
    sys.exit(_forward("rm", argv))


def ls(argv: list[str]) -> None:
    sys.exit(_forward("ls", argv))


def check(argv: list[str]) -> None:
    sys.exit(_forward("check", argv))


def install(argv: list[str]) -> None:
    rc = _forward("add", argv)
    sys.exit(_deps_build() or rc)


def test(argv: list[str]) -> None:
    """Lockstep soak of planners via the benchmark runner.

    `arena planners test [--preflight] <name...|--all> [sim:=gazebo] [KEY:=VALUE ...]`
    runs one short crowded stage per planner with the scheduler stepping
    the sim (inline suite + contest, `--lockstep-verdict`), then prints a
    per-planner stall/rtf/beat table. Exit 3 when a planner stalls for 5 s
    or never beats the gate, exit 4 when the runner itself hung and was
    killed by its deadman. `--preflight` is the start-and-drive check: two
    episodes of map_empty's `preflight` scenario (a clear 5 m straight run,
    two scripted arenians patrolling elsewhere in the hall) on a 20 s sim
    budget, a 90 s env startup deadline, a 120 s spawn ceiling and no
    retries. Verdict per planner: wedged (a cell
    errored or ran short, `--strict`), weak (an episode closed less than
    half its start distance without reaching the goal, `--efficacy 0.5`)
    or ok, exit 3 on any wedged or weak. Lockstep rows informational.
    `--all` soaks every initialized planner, the `[x]` rows of
    `arena planners check`. Bridge planners run under the drl driver,
    anything else as a nav2 local planner. Other tokens forward verbatim
    to the benchmark runner. Needs the evaluation feature.
    """
    import json

    common._reg_require("evaluation")
    preflight = "--preflight" in argv
    names = [a for a in argv if ":=" not in a and not a.startswith("-")]
    rest = [a for a in argv if (":=" in a or a.startswith("-")) and a not in ("--all", "--preflight")]
    if "--all" in argv:
        if names:
            raise common.CLIError("planners test: --all is mutually exclusive with planner names")
        names = _ready_names()
        if not names:
            raise common.CLIError("planners test: no initialized planners, see 'arena planners check'")
    elif not names:
        raise common.CLIError("planners test: specify planner name(s) or --all, see 'arena planners ls'")
    contest = json.dumps(_contestants(names))
    suite = _SOAK_SUITE
    verdict = ["--lockstep-verdict"]
    if preflight:
        suite = {**_SOAK_SUITE, "launch": {**_SOAK_SUITE["launch"], "env.bootstrap_timeout": 90, "robot.mobile.deadline": 30}, "stages": [_PREFLIGHT_STAGE]}
        verdict = ["--retries", "0", "--strict", "--efficacy", "0.5", "--spawn-budget", "120"]
    common._exec("ros2", "run", "arena_evaluation", "benchmark", "--suite", json.dumps(suite), "--contest", contest, *verdict, *rest)


def uninstall(argv: list[str]) -> None:
    if argv:
        sys.exit(_forward("rm", argv))
    _forward("uninstall", [])
    common._reg_remove(_NAME)
    sys.exit(0)


COMMANDS: dict[str, Verb] = {
    v.name: v
    for v in [
        make_verb("add", add, passthrough=True, help_text="clone planner's submodules (alias: install)", complete=_SELECT),
        make_verb("update", update, passthrough=True, help_text="Update the feature to the latest state."),
        make_verb("rm", rm, passthrough=True, help_text="deinit planner's submodules (alias: uninstall <name...>)", complete=Union(_SELECT, Flags({"-f": "deinit shared paths too"}))),
        make_verb("ls", ls, passthrough=True, help_text="list planners, [x] ready, [ ] pending"),
        make_verb("check", check, passthrough=True, help_text="verify planner submodules are initialized", complete=Union(_ALL, Flags({"-q": "quiet"}))),
        make_verb("install", install, passthrough=True, help_text="clone planner's submodules (alias for add)", complete=_SELECT),
        make_verb("uninstall", uninstall, passthrough=True, help_text="Uninstall and unregister the feature.", complete=Static(_names)),
        make_verb("test", test, passthrough=True, complete=Union(_SELECT, _PREFLIGHT)),
    ]
}
