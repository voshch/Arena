"""Attach the human_steering rqt panel to a running arena env."""

import os
import sys

from common import make_verb
from complete import Flags

MANIFEST_SUFFIX = "/state/viz_manifest"


def _discover_envs() -> list[str]:
    import subprocess

    try:
        out = subprocess.check_output(
            ["ros2", "topic", "list"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [
        line[: -len(MANIFEST_SUFFIX)]
        for line in (s.strip() for s in out.splitlines())
        if line.endswith(MANIFEST_SUFFIX)
    ]


def _matches(ns: str, target: str) -> bool:
    env_ns = os.path.dirname(ns)
    return target in (ns, env_ns, env_ns.lstrip("/")) or os.path.basename(env_ns) == target


def _wait_for_env(target: str | None) -> list[str]:
    import time

    waited = 0
    while True:
        nodes = _discover_envs()
        if nodes and (target is None or any(_matches(n, target) for n in nodes)):
            return nodes
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            who = f"env '{target}'" if target is not None else "an env"
            print(f"arena human: waiting for {who} ({waited}s elapsed)", file=sys.stderr)


def _main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="arena human", description=VERB.help)
    parser.add_argument("--ns", dest="ns")
    parser.add_argument("--unlimited", action="store_true", help="pose sliders run 0..2pi, no joint-limit clamping")
    parser.add_argument("target", nargs="?")
    args = parser.parse_args(argv)

    if args.ns and args.target:
        parser.error("pass either --ns or a positional target, not both")
    target: str | None = args.ns or args.target

    nodes = _wait_for_env(target)

    if target is None:
        if len(nodes) > 1:
            print("arena human: multiple envs running, pass <env_id> or --ns:", file=sys.stderr)
            for n in nodes:
                print(f"  {os.path.dirname(n)}", file=sys.stderr)
            return 1
        chosen = nodes[0]
    else:
        chosen = next(n for n in nodes if _matches(n, target))

    print(f"arena human: attaching to {chosen}", file=sys.stderr)
    argv_cmd = ["rqt", "--standalone", "human_steering", "--args", "--ns", chosen]
    if args.unlimited:
        argv_cmd.append("--unlimited")
    os.execvp(argv_cmd[0], argv_cmd)


def cmd(argv: list[str]) -> None:
    """Attach the human_steering rqt panel to a running arena env.

    \b
    Usage:
      human               attach to the only running env (errors if more than one)
      human <env_id>      attach to the env whose namespace equals or basename-matches <env_id>
      human --ns <ns>     explicit form of the above
      human --unlimited   pose sliders run 0..2pi, no joint-limit clamping

    Interactive control works on any backend serving human/*, driving a ped
    possesses it until the engine reclaims it. human:=dummy is the engine-less
    stage where peds only move when driven.

    Polls `ros2 topic list` for `<ns>/state/viz_manifest` topics to discover envs,
    waiting forever (10s heartbeat) until a match appears.
    """
    sys.exit(_main(list(argv)))


VERB = make_verb("human", cmd, passthrough=True, complete=Flags({"--ns": "explicit env namespace", "--unlimited": "pose sliders run 0..2pi"}, valued=("--ns",)))
