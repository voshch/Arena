"""arena CLI, invoked by the shell shim in _meta/tools/source."""

import os
import sys

import complete as _complete
import features as _features
import human as _human_mod
import robot as _robot_mod
import viz as _viz_mod
from common import (
    CLIError,
    Verb,
    _env,
    _exec,
    _reg_add,
    _reg_has,
    _reg_list,
    _reg_pull,
    _reg_remove,
    _run,
    make_verb,
)
from complete import Files, Flags, Kv, LaunchArgs, Manifest, Nothing, Packages, Static, Sub, Union

MAIN_HELP = """Arena workspace CLI.

Sim launching, fleet control, builds, and feature management for the
arena_ws workspace. Most verbs forward KEY:=VALUE tokens verbatim to
the underlying launch file or tool."""

SECTIONS = {
    "Simulation": ["runtime", "env", "viz", "cleanup", "launch", "train", "demo", "lockstep"],
    "Attach": ["human", "robot", "cam"],
    "Workspace": ["build", "rebuild", "test", "deps", "update", "preload", "uninstall"],
    "Features": ["feature", "registry"],
    "Shell": ["deactivate", "resource", "repair"],
}

_VERBS: dict[str, Verb] = {}


def _register(v: Verb) -> None:
    _VERBS[v.name] = v


def verb(name: str, *, hidden: bool = False, passthrough: bool = False, help_text: str | None = None, complete: _complete.Spec | None = None):
    def deco(fn):
        _register(make_verb(name, fn, hidden=hidden, passthrough=passthrough, help_text=help_text, complete=complete))
        return fn

    return deco


def _wants_help(argv: list[str], passthrough: bool) -> bool:
    for a in argv:
        if a in ("-h", "--help"):
            return True
        if passthrough and not a.startswith("-"):
            return False
    return False


def _listing(rows: list[tuple[str, str]]) -> str:
    if not rows:
        return ""
    width = max(len(name) for name, _ in rows)
    return "\n".join(f"  {name.ljust(width)}  {short}" if short else f"  {name}" for name, short in rows)


def _indent(text: str) -> str:
    return "\n".join(f"  {line}" if line else "" for line in text.splitlines())


def _root_help() -> str:
    out = [
        "Usage: arena [OPTIONS] COMMAND [ARGS]...",
        "",
        _indent(MAIN_HELP),
        "",
        "Options:",
        "  -h, --help  Show this message and exit.",
    ]
    listed: set[str] = set()
    sections = dict(SECTIONS)
    sections["Other"] = [n for n in _VERBS if not any(n in names for names in SECTIONS.values())]
    for title, names in sections.items():
        rows = []
        for name in names:
            v = _VERBS.get(name)
            if v is None or v.hidden or name in listed:
                continue
            listed.add(name)
            rows.append((name, v.short))
        if rows:
            out += ["", f"{title}:", _listing(rows)]
    return "\n".join(out)


def _verb_help(v: Verb) -> str:
    out = [f"Usage: arena {v.name} [ARGS]..."]
    if v.help:
        out += ["", _indent(v.help)]
    return "\n".join(out)


def main(prog_name: str = "arena") -> None:
    argv = sys.argv[1:]
    try:
        if not argv or argv[0] in ("-h", "--help"):
            print(_root_help())
            sys.exit(0)
        name, rest = argv[0], argv[1:]
        v = _VERBS.get(name)
        if v is None:
            print(f"Error: No such command '{name}'. Run 'arena --help' for the command list.", file=sys.stderr)
            sys.exit(2)
        if _wants_help(rest, v.passthrough):
            print(_verb_help(v))
            sys.exit(0)
        os.chdir(_env("ARENA_WS_DIR"))
        sys.exit(v.run(rest) or 0)
    except CLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


def _select_args(args: list[str], above: bool = False) -> list[str]:
    argv = list(args)
    if argv and not argv[0].startswith("--"):
        argv = ["--packages-above" if above else "--packages-select", *argv]
    return argv


def _supervisor(*argv: str) -> None:
    _exec("python3", "-m", "arena_bringup.supervisor", *argv)


_LAUNCH_FILES = {"task.config": Files(), "task.scenario": Files(), "task.params": Files(), "record.dir": Files()}
RUNTIME_ARGS = LaunchArgs("arena_bringup", "arena_runtime.launch.py", _LAUNCH_FILES)
ENV_ARGS = LaunchArgs("task_generator", "task_generator.launch.py", _LAUNCH_FILES)
SUPERVISOR_KNOBS = Kv(
    {
        "env.n": None,
        "viz": Static(["true", "false"]),
        "headless": Static(["true", "false"]),
        "human.steering": Static(["auto", "true", "false"]),
        **{f"viz.{k}": v for k, v in _viz_mod.VIZ_KV.keys.items()},
    }
)
LAUNCH_SPEC = Union(SUPERVISOR_KNOBS, RUNTIME_ARGS, ENV_ARGS)


@verb("launch", passthrough=True, complete=LAUNCH_SPEC)
def launch(args: list[str]) -> None:
    """Start a full simulation (runtime, envs, viz).

    Attaches additively if a runtime is already up, spawning env.n more
    envs against it (errors on sim:= mismatch). Otherwise starts
    arena_runtime.launch.py, spawns N envs, and attaches rviz unless
    headless:=true.
    """
    _supervisor(*args)


DEMO_DEFAULTS = ("task.robots:=demo", "world:=demo", "sim:=isaac", "viz.view:=robot3p")
_DEMO_ALIASES = {"tm_robots": "task.robots"}


@verb("demo", passthrough=True, complete=LAUNCH_SPEC, help_text=f"Launch a demo.\n\nSame as `arena launch` with defaults {' '.join(DEMO_DEFAULTS)}, any KEY:=VALUE you pass overrides the corresponding default.")
def demo(args: list[str]) -> None:
    given = {_DEMO_ALIASES.get(k, k) for k in (a.split(":=", 1)[0] for a in args if ":=" in a)}
    merged = [d for d in DEMO_DEFAULTS if d.split(":=", 1)[0] not in given]
    _supervisor(*merged, *args)


@verb("runtime", passthrough=True, complete=RUNTIME_ARGS)
def runtime(args: list[str]) -> None:
    """Start the runtime only (sim + arena_node, no envs).

    Fails if another /arena node is already up. Attach envs afterwards
    with `arena env`.
    """
    _exec("ros2", "launch", "arena_bringup", "arena_runtime.launch.py", *args)


@verb("env", passthrough=True, complete=ENV_ARGS)
def env_(args: list[str]) -> None:
    """Attach one task-generator env to a running runtime.

    Waits forever (10s warning cadence) for /arena/register_env if the
    runtime is not up yet.
    """
    _exec("ros2", "launch", "task_generator", "task_generator.launch.py", *args)


_register(_viz_mod.VERB)
_register(_human_mod.VERB)
_register(_robot_mod.VERB)


@verb("cam", passthrough=True)
def cam(args: list[str]) -> None:
    """Control the simulator viewport camera."""
    _exec("ros2", "run", "arena_cam", "cam", *args)


_LOCKSTEP_PRESETS = {
    "engine": {"name": "engine", "topic": "/arena/{env}/task_generator_node/agent_states", "type": "arena_humansim_msgs/msg/AgentStates", "period_s": 0.05, "hard": True},
    "peds": {"name": "peds", "topic": "/arena/{env}/arena_peds", "type": "arena_people_msgs/msg/Pedestrians", "period_s": 0.15, "hard": True},
}


def _lockstep_parse_channel(spec: str) -> dict:
    if spec in _LOCKSTEP_PRESETS:
        return _LOCKSTEP_PRESETS[spec]
    fields = spec.split("|")
    if len(fields) != 5:
        raise CLIError(f"bad channel spec '{spec}', want name|topic|type|period_s|role")
    name, topic, type_, period_s, role = fields
    try:
        period = float(period_s)
    except ValueError:
        raise CLIError(f"bad channel spec '{spec}', period_s must be a number") from None
    if period <= 0:
        raise CLIError(f"bad channel spec '{spec}', period_s must be > 0")
    if role not in ("hard", "soft"):
        raise CLIError(f"bad channel spec '{spec}', role must be hard or soft")
    return {"name": name, "topic": topic, "type": type_, "period_s": period, "hard": role == "hard"}


def _lockstep_yq(s: str) -> str:
    """YAML single-quoted scalar, safe for a bash double-quoted string."""
    return "'" + s.replace("'", "''") + "'"


def _lockstep_channel_yaml(c: dict) -> str:
    return (
        "{name: " + _lockstep_yq(c["name"])
        + ", topic: " + _lockstep_yq(c["topic"])
        + ", type: " + _lockstep_yq(c["type"])
        + f", period_s: {c['period_s']}, hard: {'true' if c['hard'] else 'false'}}}"
    )


_LOCKSTEP_SPEC = Sub(
    {
        "run": Union(Kv({"rtf": None, "for": None}), Static(["ungated"])),
        "on": Union(Kv({"rtf": None}), Static(["ungated"])),
        "off": Nothing(),
        "pause": Nothing(),
        "resume": Nothing(),
        "status": Nothing(),
        "gate": Static({k: v["topic"] for k, v in _LOCKSTEP_PRESETS.items()}),
    }
)


@verb("lockstep", complete=_LOCKSTEP_SPEC)
def lockstep(args: list[str]) -> None:
    """Run or inspect the lockstep scheduler.

    `arena lockstep run [rtf:=10] [for:=60] [ungated]` runs the sim in
    lockstep until Ctrl-C or `for:=` seconds elapse, then pauses (sim stays
    frozen at the tick boundary; `off` returns to free-run). `on`/`off`
    start/stop without blocking, `pause`/`resume` freeze and continue a
    running lockstep, `status` echoes the scheduler's latched status (config,
    measured RTF, gate state).
    `rtf:=` paces to a target real-time factor (0 = flat out). `ungated`
    ignores registered gate channels and free-steps.
    `arena lockstep gate [engine|peds|CHANNEL ...]` registers the channels
    the scheduler gates on, under caller "cli" (global, all envs). A CHANNEL
    is a preset name (engine, peds) or a raw 'name|topic|type|period_s|role'
    spec; {env} expands per env. No args clears the cli registration.
    """
    if not args or args[0] not in ("run", "on", "off", "pause", "resume", "status", "gate"):
        raise CLIError("lockstep needs one of: run, on, off, pause, resume, status, gate")
    action, rest = args[0], args[1:]
    unreachable = "lockstep unreachable"

    def srv_guard(name: str) -> str:
        return f'ros2 service list 2>/dev/null | grep -qx /arena/sim_lifecycle/lockstep/{name} || {{ echo "{unreachable}"; exit 1; }}'

    if action == "status":
        _exec(
            "bash",
            "-c",
            f'ros2 topic list 2>/dev/null | grep -qx /arena/state/lockstep || {{ echo "{unreachable}"; exit 1; }};'
            " ros2 topic echo /arena/state/lockstep --once"
            " --qos-durability transient_local --qos-reliability reliable",
        )
    stop = "ros2 service call /arena/sim_lifecycle/lockstep/stop std_srvs/srv/Trigger >/dev/null 2>&1"
    if action == "off":
        _exec("bash", "-c", f'{srv_guard("stop")}; {stop} && echo "lockstep off"')
    if action in ("pause", "resume"):
        _exec(
            "bash",
            "-c",
            f"{srv_guard(action)};"
            f" R=$(ros2 service call /arena/sim_lifecycle/lockstep/{action} std_srvs/srv/Trigger 2>&1);"
            f' grep -q "success=True" <<< "$R" && echo "lockstep {action}d" || {{ echo "lockstep not running"; exit 1; }}',
        )
    if action == "gate":
        channels = [_lockstep_parse_channel(a) for a in rest]
        reg = "registration: {caller: 'cli', env: ''" + f", channels: [{', '.join(_lockstep_channel_yaml(c) for c in channels)}]}}"
        register = (
            f"{srv_guard('register')};"
            " R=$(ros2 service call /arena/sim_lifecycle/lockstep/register"
            f' arena_runtime_msgs/srv/LockstepRegister "{reg}" 2>&1);'
            ' grep -q "success=True" <<< "$R" || { grep -oE "error_msg=.*" <<< "$R"; exit 1; }'
        )
        msg = "cli gates: " + ", ".join(c["name"] for c in channels) if channels else "cli gates cleared"
        _exec("bash", "-c", f'{register} && echo "{msg}"')

    rtf = duration = None
    ungated = False
    for a in rest:
        if a == "ungated":
            ungated = True
        elif a.startswith("rtf:="):
            rtf = float(a.removeprefix("rtf:="))
            if rtf <= 0:
                raise CLIError("rtf:= must be > 0")
        elif a.startswith("for:="):
            if action != "run":
                raise CLIError("for:= only applies to lockstep run")
            duration = float(a.removeprefix("for:="))
        else:
            raise CLIError(f"lockstep {action}: unrecognized argument '{a}'")
    start_req = f"{{target_rtf: {rtf if rtf is not None else 0.0}, ungated: {'true' if ungated else 'false'}}}"
    start = (
        f"{srv_guard('start')};"
        " R=$(ros2 service call /arena/sim_lifecycle/lockstep/start"
        f" arena_runtime_msgs/srv/LockstepStart '{start_req}' 2>&1);"
        ' grep -q "success=True" <<< "$R" || { grep -oE "error_msg=.*" <<< "$R"; exit 1; }'
    )
    if action == "on":
        _exec("bash", "-c", f'{start} && echo "lockstep on"')
    pause_cmd = "ros2 service call /arena/sim_lifecycle/lockstep/pause std_srvs/srv/Trigger >/dev/null 2>&1"
    trap_pause = f"ros2 service list 2>/dev/null | grep -qx /arena/sim_lifecycle/lockstep/pause && {pause_cmd}; echo lockstep paused"
    wait = f"sleep {duration}" if duration is not None else "sleep infinity"
    _exec(
        "bash",
        "-c",
        f'{start} && echo "lockstep running (ctrl-c to pause)";'
        f' trap "{trap_pause}" EXIT; trap "exit 0" INT TERM; {wait}',
    )


@verb("preload", passthrough=True, complete=Union(Manifest("world"), Flags({"--no-scenarios": "skip scenario assets"})))
def preload(args: list[str]) -> None:
    """Preload a world's assets ahead of launch.

    `arena preload <world_name> [--no-scenarios]`.
    """
    if not args:
        raise CLIError("missing argument WORLD")
    _exec("ros2", "run", "arena_simulation_setup", "preload_world", *args)


@verb("cleanup")
def cleanup(args: list[str]) -> None:
    """Tear down one env by id via /arena/cleanup_env."""
    if len(args) != 1 or not args[0].isdigit():
        raise CLIError("cleanup takes one non-negative integer ENV_ID")
    _exec("ros2", "service", "call", "/arena/cleanup_env", "arena_runtime_msgs/srv/CleanupEnv", f"{{env_id: {args[0]}}}")


@verb("build", passthrough=True, complete=Packages())
def build(args: list[str]) -> None:
    """Build the workspace (or selected packages) with colcon.

    Bare package names are shorthand for --packages-above. The shell
    shim re-sources the environment afterwards.
    """
    from build import build_main

    sys.exit(build_main(_select_args(args, above=True)))


@verb("rebuild", passthrough=True, complete=Packages())
def rebuild(args: list[str]) -> None:
    """Clean and rebuild selected packages.

    Accepts bare package names or colcon selection flags, e.g.
    `arena rebuild foo bar` or `arena rebuild --packages-select-regex 'arena_.*'`.
    Bare names expand to --packages-above like `arena build`.
    """
    import shutil
    import subprocess

    if not args:
        raise CLIError("rebuild needs a package selection")
    argv = _select_args(args, above=True)
    listing = subprocess.run(
        ["colcon", "list", "--names-only", "--base-paths", os.path.join(_env("ARENA_WS_DIR"), "src"), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode:
        raise CLIError("colcon list rejected the arguments, aborting before clean")
    pkgs = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    if not pkgs:
        raise CLIError("no packages matched")
    print(f"arena rebuild: resolved {len(pkgs)} package(s): {' '.join(pkgs)}")
    for pkg in pkgs:
        for tree in (os.path.join("build", pkg), os.path.join("install", pkg)):
            if os.path.isdir(tree):
                print(f"  rm -rf {tree}")
                shutil.rmtree(tree)
    print("arena rebuild: clean done, invoking build")
    from build import build_main

    sys.exit(build_main(argv))


TEST_DEFAULT_SELECT = ("--packages-select-regex", "^arena_", "^task_generator$")


@verb("test", passthrough=True, complete=Packages(), help_text=f"Run colcon test and print a summary.\n\nDefaults to `{' '.join(TEST_DEFAULT_SELECT)}` unless a selection flag is given. Bare package names are shorthand for --packages-select.")
def test(args: list[str]) -> None:
    import re
    import subprocess

    argv = _select_args(args)
    if not any(re.match(r"^--packages-(select|select-regex|up-to|above|ignore)", a) for a in argv):
        argv = [*TEST_DEFAULT_SELECT, *argv]
    listing = subprocess.run(
        ["colcon", "list", "--names-only", "--base-paths", os.path.join(_env("ARENA_WS_DIR"), "src"), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    pkgs = [line.strip() for line in listing.stdout.splitlines() if line.strip()] if listing.returncode == 0 else []
    test_rc = _run("colcon", "test", "--event-handlers", "console_direct+", *argv)
    from testsum import summarize

    summary = [os.path.join(_env("ARENA_WS_DIR"), "build")]
    if pkgs:
        summary += ["--packages", *pkgs]
    summary_rc = summarize(summary)
    sys.exit(test_rc or summary_rc)


@verb("deps", passthrough=True)
def deps(args: list[str]) -> None:
    """Install ROS dependencies for the workspace via rosdep."""
    excludes = os.environ.get("ROSDEP_EXCLUDES", "libignition-gazebo6-dev gazebo_dev gazebo_ros gazebo_plugins gazebo_ros2_control flir_ptu_description")
    _exec("rosdep", "install", "--ignore-src", "-r", "-y", "--rosdistro", _env("ARENA_ROS_DISTRO"), "--from-paths", "src", "--skip-keys", excludes, *args)


@verb("update", passthrough=True)
def update(args: list[str]) -> None:
    """Pull the Arena repos and refresh the python env."""
    import subprocess

    from pull import pull_main

    rc = pull_main(list(args))
    probe = subprocess.run(["python", "-c", "import pip.__main__"], capture_output=True, check=False)
    if probe.returncode:
        subprocess.run(["python", "-m", "ensurepip", "--upgrade"], stdout=subprocess.DEVNULL, check=False)
    sys.exit(rc)


FEATURE_HELP = """Manage optional features.

install, update, and uninstall are common verbs, see each feature's
help page for its full verb list."""


def _feature_names() -> list[str]:
    return [name for name in sorted(_features.available()) if _features.load(name) is not None]


def _feature_desc(mod) -> str:
    return mod.DESCRIPTION.replace("\b", "").strip()


def _feature_short(name: str) -> str:
    mod = _features.load(name)
    return _feature_desc(mod).splitlines()[0] if mod is not None else ""


def _feature_group_help() -> str:
    out = ["Usage: arena feature COMMAND [ARGS]...", "", _indent(FEATURE_HELP)]
    rows = [(name, _feature_short(name)) for name in _feature_names()]
    if rows:
        out += ["", "Commands:", _listing(rows)]
    return "\n".join(out)


def _feature_module_help(name: str, mod) -> str:
    out = [f"Usage: arena feature {name} COMMAND [ARGS]...", "", _indent(_feature_desc(mod))]
    rows = [(v.name, v.short) for v in mod.COMMANDS.values() if not v.hidden]
    if rows:
        out += ["", "Commands:", _listing(rows)]
    return "\n".join(out)


def _feature_cmd(args: list[str]) -> int:
    if not args or args[0] in ("-h", "--help"):
        print(_feature_group_help())
        return 0
    name, sub = args[0], args[1:]
    mod = _features.load(name)
    if mod is None:
        raise CLIError(f"No such feature '{name}'.")
    if not sub or sub[0] in ("-h", "--help"):
        print(_feature_module_help(name, mod))
        return 0
    v = mod.COMMANDS.get(sub[0])
    if v is None:
        raise CLIError(f"No such command '{sub[0]}' for feature '{name}'.")
    if _wants_help(sub[1:], v.passthrough):
        print(_verb_help(v))
        return 0
    return v.run(sub[1:]) or 0


def _feature_subs() -> dict[str, Sub]:
    return {name: Sub(_features.load(name).COMMANDS) for name in _feature_names()}


FEATURE_SPEC = Sub(_feature_subs)

_register(make_verb("feature", _feature_cmd, help_text=FEATURE_HELP, complete=FEATURE_SPEC))
_register(make_verb("ft", _feature_cmd, hidden=True, help_text="Alias for feature.", complete=FEATURE_SPEC))


@verb("shellinit", hidden=True)
def shellinit(args: list[str]) -> None:
    """Print shell init code for installed features, eval'd by `source arena`."""
    for name in _reg_list():
        mod = _features.load(name)
        if mod is None:
            continue
        v = mod.COMMANDS.get("source")
        if v is None:
            continue
        try:
            v.run([])
        except Exception as e:
            print(f"Failed to source {name}, skipping: {e}", file=sys.stderr)


@verb("train", passthrough=True, complete=LaunchArgs("arena_training", "training.launch.py", {"train_config": Files()}))
def train(args: list[str]) -> None:
    """Run DRL training (requires the training feature).

    `arena train train_config:=<yaml> [launch args]`.
    """
    sys.exit(_feature_cmd(["training", "launch", *args]))


@verb("evaluation", hidden=True, passthrough=True, complete=Sub(lambda: _features.load("evaluation").COMMANDS))
def evaluation(args: list[str]) -> None:
    """Alias for feature evaluation."""
    sys.exit(_feature_cmd(["evaluation", *args]))


REGISTRY_VERBS = ("has", "require", "add", "remove", "list", "pull")


@verb("registry", complete=Sub({a: Nothing() if a == "list" else Static(_features.available) for a in REGISTRY_VERBS}))
def registry(args: list[str]) -> None:
    """Query or mutate the installed-features registry."""
    if not args or args[0] not in REGISTRY_VERBS:
        raise CLIError(f"registry needs one of: {', '.join(REGISTRY_VERBS)}")
    action, name = args[0], args[1] if len(args) > 1 else None
    if action == "list":
        for n in _reg_list():
            print(n)
        return
    if not name:
        raise CLIError(f"'registry {action}' needs a feature name")
    if action == "has":
        sys.exit(0 if _reg_has(name) else 1)
    elif action == "require":
        if not _reg_has(name):
            print(f"{name} is not installed; run 'arena feature {name} install' first.", file=sys.stderr)
            sys.exit(1)
    elif action == "add":
        _reg_add(name)
    elif action == "remove":
        _reg_remove(name)
    elif action == "pull":
        _reg_pull(name)


@verb("complete", hidden=True, passthrough=True)
def complete_(args: list[str]) -> None:
    """Print completion candidates for a command line (`arena` and every word up to the cursor). `--refresh` regenerates the manifest cache."""
    if args == ["--refresh"]:
        sys.exit(_complete.refresh(_env("ARENA_WS_DIR")))
    print("\n".join(_complete.complete(args[1:], _VERBS, SECTIONS, _env("ARENA_WS_DIR"))))


@verb("uninstall")
def uninstall(args: list[str]) -> None:
    """Remove the container, image, and workspace trees (host only)."""
    raise CLIError("host-level verb, run it from the host workspace: source arena uninstall")


@verb("deactivate")
def deactivate(args: list[str]) -> None:
    """Leave the arena environment."""
    raise CLIError("shell-level verb, handled by the arena shell function")


@verb("repair")
def repair(args: list[str]) -> None:
    """Repair the python venv (in-container only)."""
    raise CLIError("shell-level verb, handled by the arena shell function")


@verb("resource")
def resource(args: list[str]) -> None:
    """Re-source the arena environment."""
    raise CLIError("shell-level verb, handled by the arena shell function")
