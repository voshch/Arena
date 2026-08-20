"""Attach a viz backend to a running arena env."""

import os
import sys
from typing import TYPE_CHECKING

from common import make_verb
from complete import Flags, Kv, Static, Union

if TYPE_CHECKING:
    from types import FrameType

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
            print(f"arena viz: waiting for {who} ({waited}s elapsed)", file=sys.stderr)


def _viz_commands(ns: str, viz_args: dict[str, str]) -> list[list[str]]:
    from arena_bringup import viz_backends

    try:
        pairs = viz_backends.viz_commands(ns, viz_backends.env_idx_from_ns(ns), viz_args)
    except ValueError as e:
        raise SystemExit(f"arena viz: {e}") from None
    return [cmd for _label, cmd in pairs]


def _attach_all(namespaces: list[str], viz_args: dict[str, str]) -> int:
    import signal
    import subprocess

    procs: list[subprocess.Popen] = []
    for ns in namespaces:
        for c in _viz_commands(ns, viz_args):
            print(f"arena viz: attaching to {ns} ({' '.join(c[2:4])})", file=sys.stderr)
            procs.append(
                subprocess.Popen(
                    c,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
            )

    shutting_down = False

    def shutdown(_signo: "int | None", _frame: "FrameType | None") -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    rc = 0
    for p in procs:
        try:
            r = p.wait()
        except KeyboardInterrupt:
            shutdown(None, None)
            r = p.wait()
        if r != 0 and rc == 0:
            rc = r
    return rc


def _main(argv: list[str]) -> int:
    import argparse

    viz_args: dict[str, str] = {}
    passthrough: list[str] = []
    for tok in argv:
        if ":=" in tok:
            key, _, value = tok.partition(":=")
            if key.startswith("viz."):
                key = key[len("viz."):]
            viz_args[key] = value
            continue
        passthrough.append(tok)

    parser = argparse.ArgumentParser(prog="arena viz", description=VERB.help)
    parser.add_argument("--all", dest="all_envs", action="store_true")
    parser.add_argument("--ns", dest="ns")
    parser.add_argument("target", nargs="?")
    args = parser.parse_args(passthrough)

    if args.all_envs and (args.ns or args.target):
        parser.error("--all takes no target")
    if args.ns and args.target:
        parser.error("pass either --ns or a positional target, not both")
    target: str | None = args.ns or args.target

    nodes = _wait_for_env(target)

    if args.all_envs:
        return _attach_all(nodes, viz_args)

    if target is None:
        if len(nodes) > 1:
            print("arena viz: multiple envs running, pass <env_id> or --all:", file=sys.stderr)
            for n in nodes:
                print(f"  {os.path.dirname(n)}", file=sys.stderr)
            return 1
        chosen = nodes[0]
    else:
        chosen = next(n for n in nodes if _matches(n, target))

    cmds = _viz_commands(chosen, viz_args)
    if len(cmds) == 1:
        print(f"arena viz: attaching to {chosen}", file=sys.stderr)
        os.execvp(cmds[0][0], cmds[0])
    return _attach_all([chosen], viz_args)


def cmd(argv: list[str]) -> None:
    """Attach a viz backend to a running arena env.

    \b
    Usage:
      viz                attach to the only running env (errors if more than one)
      viz <env_id>       attach to the env whose namespace equals or basename-matches <env_id>
      viz --ns <ns>      explicit form of the above
      viz --all          attach to every running env, shut down all on Ctrl-C

    Any <key>:=<value> token is forwarded to the backend's launch file. A `viz.`
    prefix is accepted (so tokens shared with `arena launch` work unchanged) but
    not required.

    \b
    Backend selection:
      backend:=rviz|rerun[,rviz|rerun]   one or more backends, comma-separated
                                         (default: $ARENA_VIZ_BACKEND or 'rviz')

    \b
    Backend-specific args:
      rviz:
        view:=map|robot|robot3p          camera view (default: map)
        robot:=<int>|all                 robot index in fleet, 'all' fans out one window per robot
      rerun:
        web_port:=<int>                  web viewer port (default: 9090, auto-bumped per env)
        grpc_port:=<int>                 gRPC stream port (default: 9876, auto-bumped per env)

    Polls `ros2 topic list` for `<ns>/state/viz_manifest` topics to discover envs,
    waiting forever (10s heartbeat) until a match appears.
    """
    sys.exit(_main(list(argv)))


VIZ_KV = Kv(
    {
        "backend": Static(["rviz", "rerun"]),
        "view": Static(["map", "robot", "robot3p"]),
        "robot": Static(["all"]),
        "web_port": None,
        "grpc_port": None,
    }
)

VERB = make_verb("viz", cmd, passthrough=True, complete=Union(Flags({"--all": "attach to every running env", "--ns": "explicit env namespace"}, valued=("--ns",)), VIZ_KV))
