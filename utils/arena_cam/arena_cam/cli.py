"""Unified CLI for Arena viewport-camera scripting.

One command shape for everything:

    arena cam <name> [key=value ...] [--sim] [--viz [ENV_ID]] [--env N]   # name is verb, camera asset, or .yaml
    arena cam <name> ... --lockstep [--fps 30]   # play live, gated frame by frame
    arena cam <name> ... --record [FILE] [--fps 30] [--lockstep] [-f]   # render to a video (ffmpeg) instead
    arena cam serve [--env N] [--sim] [--fps 30] [--setup REQUEST]   # one env's camera server
    arena cam <name> ... --record [FILE] --lockstep --mount FRAME[@X,Y,Z[,PITCH]]   # plus a camera fixed on FRAME
    arena cam drive [--sim] [--viz [ENV_ID]] [--record [FILE] ...]   # fly the camera from the keyboard, R records
    arena cam list                               # catalog of verbs and global camera assets
    arena cam show <name>                        # parameters of a verb or camera asset

A verb and an asset launch identically; the caller need not know which a name is.
Params are bare `key=value` (coerced: number / x,y,z tuple / bool / string);
launcher options are `--flags` and may sit anywhere on the line. Nested or
list-valued params live in a shot file. `--record` switches from live playback to
deterministic capture: its optional FILE names the output (.mp4 if no suffix),
bare it is `<name>_<YYYYmmdd-HHMMSS>`. `--fps` sets the frame rate and
`--lockstep`, recording or not, rides an active lockstep run as a registered hard
channel gated at 1/fps, or steps physics by 1/fps between frames itself when no run
is active. Without it live playback has no sync to the sim.

Targets select which viewport cameras the shot drives. With no flag it drives
everything: the sim GUI camera plus every env's rviz camera. `--sim` is sim only,
`--viz` is all rviz cameras, `--viz <env_id>` is one env's, and the flags compose.
Record writes one file per selected camera, tagged `-sim` / `-viz<env>` when there are several.
`--mount` adds a camera fixed on a TF frame, filmed through the sim camera at each
frame's sim time into a file tagged by the frame, and needs `--lockstep` and the sim camera.
Context tokens in a shot read the env of their surface, the sim camera reads `--env`.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

import yaml

from arena_cam import Camera, TargetSelection, load_shot
from arena_cam.camera import global_cameras, global_spec
from arena_cam.curves import FOV_DEFAULT
from arena_cam.mounts import SPEC_FORM, parse_mount
from arena_cam.record import default_name
from arena_cam.registry import PRIMITIVES

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


def parse_params(tokens: list[str]) -> dict:
    params: dict = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected key=value, got {token!r}")
        key, _, value = token.partition("=")
        params[key] = _coerce(value)
    return params


def is_path(name: str) -> bool:
    return name.endswith((".yaml", ".yml")) or "/" in name or os.path.exists(name)


def _print_catalog() -> None:
    print("interactive:")
    print("  drive")
    print("service:")
    print("  serve")
    print("verbs:")
    for verb in sorted(PRIMITIVES):
        print(f"  {verb}")
    print("shots:")
    shots = global_cameras()
    for shot in shots:
        print(f"  {shot}")
    if not shots:
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
    else:
        spec = global_spec(name)
        if spec is None:
            raise SystemExit(f"unknown name: {name!r} (try 'arena cam list')")
        print(f"{name} (camera asset)")
        print(yaml.safe_dump(spec, sort_keys=False, default_flow_style=False).rstrip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arena cam", description="Arena viewport-camera CLI")
    parser.add_argument("name", nargs="?", help="verb, camera asset, or .yaml file (or 'list' / 'show' / 'drive' / 'serve')")
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
    parser.add_argument("--fps", type=float, default=30.0, help="frame rate of a recording, of a --lockstep take and of serve (default 30)")
    parser.add_argument("--lockstep", action="store_true", help="gate the take in physics lockstep, one 1/fps step per frame, recording or live")
    parser.add_argument("--env", type=int, default=0, help="the env of the sim camera: what its context tokens read, and what serve drives (default 0)")
    parser.add_argument("--setup", default="", metavar="REQUEST", help="serve: camera asset the env's task generator plays at every episode start, by name or index")
    parser.add_argument("-f", "--force", action="store_true", help="overwrite an existing record file")
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        metavar=SPEC_FORM,
        help="also film from a camera fixed on TF frame FRAME at X,Y,Z m (x forward, z up), tilted down by PITCH rad, repeatable, needs --record --lockstep and the sim camera",
    )
    parser.add_argument("--mount-fov", type=float, default=FOV_DEFAULT, help=f"horizontal fov of every mount in rad (default {FOV_DEFAULT})")
    args = parser.parse_intermixed_args(argv if argv is not None else sys.argv[1:])
    if args.record is None and args.name != "drive" and args.force:
        parser.error("-f only applies with --record")
    if args.record is None and args.name not in ("drive", "serve") and args.fps != 30.0 and not args.lockstep:
        parser.error("--fps only applies with --record or --lockstep")
    if args.record and "=" in args.record:
        parser.error(f"--record took {args.record!r} as the file name, use --record=FILE or put it after the key=value params")
    if args.mount and (args.record is None or not args.lockstep):
        parser.error("--mount films in lockstep, add --record and --lockstep")
    if args.mount and args.viz is not None and not args.sim:
        parser.error("--mount films through the sim camera, add --sim")
    try:
        mounts = [parse_mount(spec, args.mount_fov) for spec in args.mount]
    except ValueError as e:
        parser.error(str(e))

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
    if args.name == "serve":
        if args.params or args.viz is not None or args.record is not None:
            parser.error("serve takes only --env, --sim, --fps and --setup")
        from arena_cam.server import serve

        serve(args.env, args.sim, args.fps, args.setup)
        return
    targets = _resolve_targets(args.sim, args.viz)
    shot_file = is_path(args.name)
    try:
        params = parse_params(args.params)
        cam = load_shot(args.name, targets, params) if shot_file else Camera(targets).add(args.name, params)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if args.record is None:
        if not cam.play(lockstep=args.lockstep, fps=args.fps, env=args.env):
            raise SystemExit(1)
        return
    out = args.record or default_name(os.path.splitext(os.path.basename(args.name))[0] if shot_file else args.name)
    try:
        ran = cam.record(out, fps=args.fps, force=args.force, lockstep=args.lockstep, mounts=mounts, env=args.env)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e
    if not ran:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
