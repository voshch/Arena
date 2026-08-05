"""Spawn, list, and despawn robots in a running arena fleet at runtime."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from common import make_verb

if TYPE_CHECKING:
    import argparse
    import contextlib

    import rclpy.node

ROBOTS_SUFFIX = "/state/robots"


def discover_envs() -> list[str]:
    import subprocess

    try:
        out = subprocess.check_output(["ros2", "topic", "list"], stderr=subprocess.DEVNULL, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    return [
        line[: -len(ROBOTS_SUFFIX)]
        for line in (s.strip() for s in out.splitlines())
        if line.endswith(ROBOTS_SUFFIX)
    ]


def matches(ns: str, target: str) -> bool:
    env_ns = os.path.dirname(ns)
    return target in (ns, env_ns, env_ns.lstrip("/")) or os.path.basename(env_ns) == target


def wait_for_env(target: str | None) -> list[str]:
    import time

    waited = 0
    while True:
        nodes = discover_envs()
        if nodes and (target is None or any(matches(n, target) for n in nodes)):
            return nodes
        time.sleep(1)
        waited += 1
        if waited % 10 == 0:
            who = f"env '{target}'" if target is not None else "an env"
            print(f"arena robot: waiting for {who} ({waited}s elapsed)", file=sys.stderr)


def resolve_single(target: str | None) -> str:
    nodes = wait_for_env(target)
    if target is None:
        if len(nodes) > 1:
            print("arena robot: multiple envs running, pass --env <id> or --ns <ns>:", file=sys.stderr)
            for n in nodes:
                print(f"  {os.path.dirname(n)}", file=sys.stderr)
            raise SystemExit(1)
        return nodes[0]
    return next(n for n in nodes if matches(n, target))


def ros_node() -> contextlib.AbstractContextManager[rclpy.node.Node]:
    import contextlib

    @contextlib.contextmanager
    def _cm() -> object:
        import rclpy

        own = not rclpy.ok()
        if own:
            rclpy.init(args=[])
        node = rclpy.create_node("arena_robot_cli")
        try:
            yield node
        finally:
            node.destroy_node()
            if own:
                rclpy.shutdown()

    return _cm()


def call_service(node: rclpy.node.Node, srv_type: type, srv_name: str, request: object, timeout: float = 10.0) -> object:
    import rclpy

    client = node.create_client(srv_type, srv_name)
    if not client.wait_for_service(timeout_sec=timeout):
        raise SystemExit(f"arena robot: service {srv_name} unavailable")
    future = client.call_async(request)
    while rclpy.ok() and not future.done():
        rclpy.spin_once(node, timeout_sec=0.1)
    return future.result()


def wait_until_ready(node: rclpy.node.Node, node_ns: str) -> None:
    import time

    import rclpy
    from rcl_interfaces.srv import GetParameters

    client = node.create_client(GetParameters, node_ns + "/get_parameters")
    start = time.monotonic()
    next_beat = 10.0
    try:
        while rclpy.ok():
            if client.service_is_ready():
                req = GetParameters.Request()
                req.names = ["initialized"]
                future = client.call_async(req)
                while rclpy.ok() and not future.done():
                    rclpy.spin_once(node, timeout_sec=0.1)
                resp = future.result()
                if resp is not None and resp.values and resp.values[0].bool_value:
                    return
            rclpy.spin_once(node, timeout_sec=0.2)
            elapsed = time.monotonic() - start
            if elapsed >= next_beat:
                print(f"arena robot: waiting for {node_ns} to initialize ({int(elapsed)}s elapsed)", file=sys.stderr)
                next_beat += 10.0
    finally:
        node.destroy_client(client)


def read_fleet(node: rclpy.node.Node, node_ns: str, timeout: float = 5.0) -> list[object]:
    import time

    import rclpy
    import rclpy.qos
    from task_generator_msgs.msg import RobotFleet

    latest: list = []
    qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(RobotFleet, node_ns + ROBOTS_SUFFIX, lambda m: latest.append(m), qos)
    deadline = time.monotonic() + timeout
    while rclpy.ok() and not latest and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    return list(latest[-1].robots) if latest else []


def wait_for_presence(node: rclpy.node.Node, node_ns: str, names: set[str], present: bool, label: str) -> None:
    import rclpy
    import rclpy.qos
    from task_generator_msgs.msg import RobotFleet

    state: dict[str, set[str] | None] = {"names": None}
    qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(
        RobotFleet,
        node_ns + ROBOTS_SUFFIX,
        lambda m: state.update(names={r.descriptor.name for r in m.robots}),
        qos,
    )
    waited = 0.0
    next_beat = 10.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            live = state["names"]
            if live is not None and all((n in live) == present for n in names):
                return
            waited += 0.1
            if waited >= next_beat:
                print(f"arena robot: waiting for {label} ({int(waited)}s elapsed)", file=sys.stderr)
                next_beat += 10.0
    finally:
        node.destroy_subscription(sub)


def cmd_spawn(model: str, kv: dict[str, str], opts: argparse.Namespace) -> int:
    from diagnostic_msgs.msg import KeyValue
    from task_generator_msgs.srv import SpawnRobot

    count = int(kv.pop("count", "1"))
    name = kv.pop("name", "")
    immediate = kv.pop("immediate", "true").lower() not in ("false", "0", "no")
    if count > 1 and name:
        print("arena robot: name:= cannot be combined with count:= > 1", file=sys.stderr)
        return 1

    node_ns = resolve_single(opts.target)
    spawned: set[str] = set()
    with ros_node() as node:
        wait_until_ready(node, node_ns)
        for _ in range(count):
            req = SpawnRobot.Request()
            req.model = model
            req.name = name
            req.immediate = immediate
            req.args = [KeyValue(key=k, value=v) for k, v in kv.items()]
            resp = call_service(node, SpawnRobot, node_ns + "/runtime/spawn_robot", req)
            if resp is None or not resp.success:
                msg = resp.error_msg if resp else "no response"
                print(f"arena robot: spawn failed: {msg}", file=sys.stderr)
                return 1
            print(resp.name)
            spawned.add(resp.name)
        if opts.nowait:
            return 0
        wait_for_presence(node, node_ns, spawned, present=True, label=f"{', '.join(sorted(spawned))} to spawn")
    return 0


def cmd_rm(name: str, opts: argparse.Namespace) -> int:
    from task_generator_msgs.srv import DespawnRobot

    node_ns = resolve_single(opts.target)
    with ros_node() as node:
        wait_until_ready(node, node_ns)
        req = DespawnRobot.Request()
        req.name = name
        resp = call_service(node, DespawnRobot, node_ns + "/runtime/despawn_robot", req)
        if resp is None or not resp.success:
            msg = resp.error_msg if resp else "no response"
            print(f"arena robot: despawn failed: {msg}", file=sys.stderr)
            return 1
        if opts.nowait:
            return 0
        wait_for_presence(node, node_ns, {name}, present=False, label=f"'{name}' to despawn")
    return 0


def cmd_ls(opts: argparse.Namespace) -> int:
    nodes = wait_for_env(None) if opts.all_envs else [resolve_single(opts.target)]
    with ros_node() as node:
        for ns in nodes:
            print(os.path.dirname(ns))
            robots = read_fleet(node, ns)
            if not robots:
                print("  (empty)")
            for r in robots:
                print(f"  {r.descriptor.name}\t{r.descriptor.model}\t{r.descriptor.ns}")
    return 0


def cmd_caps(model: str) -> int:
    from ament_index_python.packages import get_package_share_path
    from arena_robots.Robot import RobotIdentifier

    try:
        view = RobotIdentifier(model).resolve_sync()
    except FileNotFoundError as e:
        print(f"arena robot: {e}", file=sys.stderr)
        return 1

    print(f"model: {model}")

    caps = sorted(view.caps.available)
    print(f"caps: {', '.join(caps) if caps else '(none)'}")

    spec = view.assembly
    if spec is None:
        print("morphology: not migrated (no assembly.yaml) - lidar=/arm=/etc parametrization unavailable")
    else:
        print("mounts:")
        for name in sorted(spec.mounts):
            m = spec.mounts[name]
            xyz = " ".join(f"{v:g}" for v in m.xyz)
            rpy = " ".join(f"{v:g}" for v in m.rpy)
            accepts = ",".join(m.accepts)
            print(f"  {name}\tparent={m.parent}\txyz=[{xyz}]\trpy=[{rpy}]\taccepts=[{accepts}]")

        print("defaults:")
        for t in sorted(spec.defaults):
            for p in spec.defaults[t]:
                extra = ""
                if p.params:
                    extra += f"\tparams={p.params}"
                if p.overrides:
                    extra += f"\toverrides={p.overrides}"
                print(f"  {t}={p.variant}@{p.mount}{extra}")

        from arena_robots.catalog import ComponentSpec

        types = sorted({t for m in spec.mounts.values() for t in m.accepts})
        components_root = get_package_share_path("arena_robots") / "components"
        print("catalog:")
        for t in types:
            type_dir = components_root / t
            variants: list[str] = []
            if type_dir.is_dir():
                for d in sorted(type_dir.iterdir()):
                    manifest = d / "component.yaml"
                    if not manifest.is_file():
                        continue
                    family = ComponentSpec.from_yaml(manifest).variants
                    variants.extend(family or [d.name])
            print(f"  {t}: {', '.join(sorted(variants)) if variants else '(none)'}")

    print("sensors:")
    for s in view.effective_sensors({}):
        print(f"  {s.name}\t{s.type}\t{s.topic}")
    return 0


def _add_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", dest="env", type=int)
    parser.add_argument("--ns", dest="ns")


def _resolve_target(args: argparse.Namespace) -> str | None:
    if args.env is not None and args.ns:
        raise SystemExit("arena robot: pass either --env or --ns, not both")
    if args.env is not None:
        return f"env_{args.env}"
    return args.ns


HELP = """Spawn, list, and despawn robots in a running arena fleet at runtime.

\b
Usage:
  arena robot <model> [name:=N] [mobile:=M] [arm:=A] [count:=K] [key:=value ...]
                      [--env <id>|--ns <ns>] [--nowait]
  arena robot rm <name> [--env <id>|--ns <ns>] [--nowait]
  arena robot ls [--env <id>|--ns <ns>|--all]
  arena robot caps <model>

`caps` is a build-time, ROS-network-free lookup: it prints a robot model's caps,
morphology (mounts/priority/defaults, if migrated to assembly.yaml), catalog
variants per mounted type, and effective default sensors. Use it to see what a
`robot:=<model>[...]` bracket can legally do before spawning anything.

<key>:=<value> tokens describe the robot and are forwarded to the spawn service:
`name` selects the instance name, `count` spawns that many, and every other key
is passed through to Robot.parse (e.g. mobile:=manual, arm:=none). --flags
control the command, the model is the sole positional for spawn.

Spawns go live immediately (idle, in the staging area) and join the episode at the
next reset. Pass immediate:=false to defer the whole spawn to the next reset instead.
Despawns take effect on the next reset. By default the command waits until the robot
appears in or disappears from the fleet (10s heartbeat), and --nowait returns as soon
as the request is accepted.

Discovers envs by polling `ros2 topic list` for `<ns>/state/robots`, waiting
forever (10s heartbeat) until a match appears.
"""


def _main(argv: list[str]) -> int:
    import argparse

    kv: dict[str, str] = {}
    passthrough: list[str] = []
    for tok in argv:
        if ":=" in tok:
            key, _, value = tok.partition(":=")
            kv[key] = value
        else:
            passthrough.append(tok)

    sub = passthrough[0] if passthrough else None

    if sub == "ls":
        parser = argparse.ArgumentParser(prog="arena robot ls")
        _add_target(parser)
        parser.add_argument("--all", dest="all_envs", action="store_true")
        pargs = parser.parse_args(passthrough[1:])
        pargs.target = None if pargs.all_envs else _resolve_target(pargs)
        return cmd_ls(pargs)

    if sub == "caps":
        parser = argparse.ArgumentParser(prog="arena robot caps")
        parser.add_argument("model")
        pargs = parser.parse_args(passthrough[1:])
        return cmd_caps(pargs.model)

    if sub == "rm":
        parser = argparse.ArgumentParser(prog="arena robot rm")
        parser.add_argument("name")
        _add_target(parser)
        parser.add_argument("--nowait", action="store_true")
        pargs = parser.parse_args(passthrough[1:])
        pargs.target = _resolve_target(pargs)
        return cmd_rm(pargs.name, pargs)

    parser = argparse.ArgumentParser(prog="arena robot", description=HELP)
    parser.add_argument("model")
    _add_target(parser)
    parser.add_argument("--nowait", action="store_true")
    pargs = parser.parse_args(passthrough)
    pargs.target = _resolve_target(pargs)
    return cmd_spawn(pargs.model, kv, pargs)


VERB = make_verb("robot", _main, passthrough=True, help_text=HELP)
