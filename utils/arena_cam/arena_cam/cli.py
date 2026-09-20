"""Unified CLI for Arena viewport-camera scripting.

One command shape for everything:

    arena cam <name> [key=value ...] [--sim] [--viz [ENV_ID]]   # name is verb, shot, or .yaml
    arena cam <name> ... --record [FILE] [--fps 30] [--lockstep] [-f]   # render to a video (ffmpeg) instead
    arena cam drive [--sim] [--viz [ENV_ID]] [--record [FILE] ...]   # fly the camera from the keyboard, R records
    arena cam list                               # catalog of verbs and shots
    arena cam show <name>                        # parameters of a verb or shot

A verb and a shot launch identically; the caller need not know which a name is.
Params are bare `key=value` (coerced: number / x,y,z tuple / bool / string);
launcher options are `--flags` and may sit anywhere on the line. Nested or
list-valued params live in a shot file. `--record` switches from live playback to
deterministic capture: its optional FILE names the output (.mp4 if no suffix),
bare it is `<name>_<YYYYmmdd-HHMMSS>`. `--fps` sets the frame rate and
`--lockstep` rides an active lockstep run as a registered hard channel gated at
1/fps, or steps physics by 1/fps between frames itself when no run is active.

Targets select which viewport cameras the shot drives. With no flag it drives
everything: the sim GUI camera plus every env's rviz camera. `--sim` is sim only,
`--viz` is all rviz cameras, `--viz <env_id>` is one env's, and the flags compose.
Record writes one file per selected camera, tagged `-sim` / `-viz<env>` when there are several.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

import yaml

from arena_cam import Camera, TargetSelection, load_shot
from arena_cam.record import default_name
from arena_cam.registry import PRIMITIVES
from arena_cam.shots import SHOTS

# Sentinel for `--viz` given with no env id (all vizes), distinct from `--viz` absent.
_VIZ_ALL = object()


def _resolve_targets(sim_flag: bool, viz_arg: object) -> TargetSelection:
    """Map the --sim / --viz flags to a selection. No flag drives everything."""
    viz_given = viz_arg is not None
    if not sim_flag and not viz_given:
        return TargetSelection(include_sim=True, viz_all=True, viz_env=None)
    if viz_arg is _VIZ_ALL:
        return TargetSelection(include_sim=sim_flag, viz_all=True, viz_env=None)
    if viz_given:
        try:
            env_id = int(str(viz_arg))
        except (TypeError, ValueError):
            raise SystemExit(f"--viz expects an env id, got {viz_arg!r}") from None
        return TargetSelection(include_sim=sim_flag, viz_all=False, viz_env=env_id)
    return TargetSelection(include_sim=True, viz_all=False, viz_env=None)


def _coerce(token: str) -> object:
    low = token.lower()
    if low in ("true", "false"):
        return low == "true"
    if "," in token:
        return tuple(_coerce(part) for part in token.split(","))
    try:
        return float(token)
    except ValueError:
        return token


def _parse_params(tokens: list[str]) -> dict:
    params: dict = {}
    for token in tokens:
        if "=" not in token:
            raise SystemExit(f"expected key=value, got {token!r}")
        key, _, value = token.partition("=")
        params[key] = _coerce(value)
    return params


def _is_path(name: str) -> bool:
    return name.endswith((".yaml", ".yml")) or "/" in name or os.path.exists(name)


def _print_catalog() -> None:
    print("interactive:")
    print("  drive")
    print("verbs:")
    for verb in sorted(PRIMITIVES):
        print(f"  {verb}")
    print("shots:")
    for shot in sorted(SHOTS):
        print(f"  {shot}")
    if not SHOTS:
        print("  (none installed)")


def _drive(sim_flag: bool, viz_arg: object, record: str | None, fps: float, lockstep: bool, force: bool) -> None:
    """Hand the camera to the rqt panel, forwarding the target and record flags."""
    flags: list[str] = []
    if record is not None:
        flags.append(f"--record={record}")
    flags += ["--fps", f"{fps:g}"]
    if lockstep:
        flags.append("--lockstep")
    if force:
        flags.append("--force")
    if sim_flag:
        flags.append("--sim")
    if viz_arg is _VIZ_ALL:
        flags.append("--viz")
    elif viz_arg is not None:
        flags += ["--viz", str(viz_arg)]
    # force-discover: an rqt plugin cache written before this panel existed hides it.
    argv = ["rqt", "--force-discover", "--standalone", "arena_cam", "--args", *flags]
    os.execvp(argv[0], argv)


def _show(name: str) -> None:
    if name in PRIMITIVES:
        cls = PRIMITIVES[name]
        print(f"{name} (verb)")
        for param in inspect.signature(cls.__init__).parameters.values():
            if param.name == "self":
                continue
            default = "" if param.default is inspect.Parameter.empty else f" = {param.default!r}"
            print(f"  {param.name}{default}")
        if cls.__doc__:
            print(f"  -- {cls.__doc__.strip().splitlines()[0]}")
    elif name in SHOTS:
        print(f"{name} (shot)")
        print(yaml.safe_dump(SHOTS[name], sort_keys=False, default_flow_style=False).rstrip())
    else:
        raise SystemExit(f"unknown name: {name!r} (try 'arena cam list')")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arena cam", description="Arena viewport-camera CLI")
    parser.add_argument("name", nargs="?", help="verb, shot, or .yaml file (or 'list' / 'show' / 'drive')")
    parser.add_argument("params", nargs="*", metavar="key=value", help="verb or shot parameters")
    parser.add_argument("--sim", action="store_true", help="drive the sim GUI camera")
    parser.add_argument(
        "--viz",
        nargs="?",
        const=_VIZ_ALL,
        default=None,
        metavar="ENV_ID",
        help="drive rviz cameras: bare for all, or an env id for one",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="render to a video instead of playing live: FILE under $ARENA_DATA_DIR/recordings (.mp4 if no suffix), bare for <name>_<YYYYmmdd-HHMMSS>",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="record frame rate (default 30)")
    parser.add_argument("--lockstep", action="store_true", help="record in physics lockstep, one 1/fps step per frame")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite an existing record file")
    args = parser.parse_intermixed_args(argv if argv is not None else sys.argv[1:])
    if args.record is None and args.name != "drive" and (args.fps != 30.0 or args.lockstep or args.force):
        parser.error("--fps, --lockstep and -f only apply with --record")
    if args.record and "=" in args.record:
        parser.error(f"--record took {args.record!r} as the file name, use --record=FILE or put it after the key=value params")

    if args.name is None:
        parser.print_usage()
        _print_catalog()
        print("play one with `arena cam <name> [key=value ...]`, film it with `--record [FILE]`, see `arena cam --help`")
        return
    if args.name == "list":
        _print_catalog()
        return
    if args.name == "show":
        if not args.params:
            parser.error("show needs a name: arena cam show <verb|shot>")
        _show(args.params[0])
        return
    if args.name == "drive":
        if args.params:
            parser.error("drive takes no params, only the target flags")
        _drive(args.sim, args.viz, args.record, args.fps, args.lockstep, args.force)
        return
    params = _parse_params(args.params)
    targets = _resolve_targets(args.sim, args.viz)
    is_path = _is_path(args.name)
    try:
        cam = load_shot(args.name, targets) if is_path else Camera(targets).add(args.name, params)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if args.record is None:
        cam.play()
        return
    out = args.record or default_name(os.path.splitext(os.path.basename(args.name))[0] if is_path else args.name)
    try:
        cam.record(out, fps=args.fps, force=args.force, lockstep=args.lockstep)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from e


if __name__ == "__main__":
    main()
