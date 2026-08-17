"""Unified CLI for Arena viewport-camera scripting.

One command shape for everything:

    arena cam <name> [key=value ...] [--sim] [--viz [ENV_ID]]   # name is verb, shot, or .yaml
    arena cam <name> ... record=<dir> [fps=30] [lockstep=true] [-f]   # render to a PPM frame sequence instead
    arena cam drive [--sim] [--viz [ENV_ID]]     # fly the camera from the keyboard
    arena cam list                               # catalog of verbs and shots
    arena cam show <name>                        # parameters of a verb or shot

A verb and a shot launch identically; the caller need not know which a name is.
Params are bare `key=value` (coerced: number / x,y,z tuple / bool / string);
launcher options are `--flags`. Nested or list-valued params live in a shot file.
The reserved params `record` (output dir), `fps`, and `lockstep` switch from live
playback to deterministic capture. `lockstep` rides an active lockstep run as a
registered hard channel gated at 1/fps, or steps physics by 1/fps between frames
itself when no run is active.

Targets select which viewport cameras the shot drives. With no flag it drives
everything: the sim GUI camera plus every env's rviz camera. `--sim` is sim only,
`--viz` is all rviz cameras, `--viz <env_id>` is one env's, and the flags compose.
Record needs the selection to resolve to a single camera.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys

import yaml

from arena_cam import Camera, TargetSelection, load_shot
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


def _drive(sim_flag: bool, viz_arg: object) -> None:
    """Hand the camera to the rqt panel, forwarding the target flags."""
    flags: list[str] = []
    if sim_flag:
        flags.append("--sim")
    if viz_arg is _VIZ_ALL:
        flags.append("--viz")
    elif viz_arg is not None:
        flags += ["--viz", str(viz_arg)]
    # force-discover: an rqt plugin cache written before this panel existed hides it.
    argv = ["rqt", "--force-discover", "--standalone", "arena_cam"]
    if flags:
        argv += ["--args", *flags]
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
    parser.add_argument("-f", "--force", action="store_true", help="overwrite a non-empty record dir")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.name is None or args.name == "list":
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
        _drive(args.sim, args.viz)
        return
    params = _parse_params(args.params)
    record = params.pop("record", None)
    fps = float(params.pop("fps", 30.0))
    lockstep = bool(params.pop("lockstep", False))

    targets = _resolve_targets(args.sim, args.viz)
    try:
        cam = load_shot(args.name, targets) if _is_path(args.name) else Camera(targets).add(args.name, params)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if record is not None:
        try:
            cam.record(str(record), fps=fps, force=args.force, lockstep=lockstep)
        except FileExistsError as e:
            raise SystemExit(str(e)) from e
    else:
        cam.play()


if __name__ == "__main__":
    main()
