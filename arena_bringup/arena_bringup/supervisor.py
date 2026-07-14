"""Process supervisor for `arena launch`.

Spawns arena_runtime + N task_generator envs (+ rviz per env), discovers
readiness via rclpy graph queries, and propagates signals to the entire
process tree of each child via process-group signaling.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterable

import rclpy
import rclpy.node
from rcl_interfaces.srv import GetParameters

REGISTER_ENV_SERVICE = '/arena/register_env'
VIZ_MANIFEST_SUFFIX = '/state/viz_manifest'

GRACE_INT_S = float(os.environ.get('ARENA_SHUTDOWN_TIMEOUT', '30'))
GRACE_TERM_S = 3.0
GRACE_KILL_S = 5.0


class Supervisor:
    def __init__(self, node: rclpy.node.Node) -> None:
        self._node = node
        self._children: dict[str, subprocess.Popen] = {}
        self._signal_count = 0

    def spawn(self, role: str, cmd: list[str]) -> subprocess.Popen:
        proc = subprocess.Popen(cmd, start_new_session=True)
        self._children[role] = proc
        sys.stderr.write(f'arena launch: spawned {role} (pid={proc.pid})\n')
        return proc

    def runtime(self) -> subprocess.Popen | None:
        return self._children.get('runtime')

    def alive(self) -> list[tuple[str, subprocess.Popen]]:
        return [(r, p) for r, p in self._children.items() if p.poll() is None]

    def signal_handler(self, signum: int, _frame: types.FrameType | None) -> None:
        self._signal_count += 1
        n = self._signal_count
        snap = self.alive()
        if n == 1:
            sig, label = signal.SIGINT, 'SIGINT'
        elif n == 2:
            sig, label = signal.SIGTERM, 'SIGTERM'
        elif n == 3:
            sig, label = signal.SIGKILL, 'SIGKILL'
        else:
            os._exit(130)
        sys.stderr.write(f'arena launch: {label} -> {len(snap)} child(ren) (Ctrl-C again to escalate)\n')
        self._signal_pgs(snap, sig)

    def cleanup(self) -> None:
        snap = self.alive()
        if not snap:
            return
        self._signal_pgs(snap, signal.SIGINT)
        if self._wait(snap, GRACE_INT_S):
            return
        sys.stderr.write('arena launch: graceful timeout, sending SIGTERM\n')
        self._signal_pgs(snap, signal.SIGTERM)
        if self._wait(snap, GRACE_TERM_S):
            return
        sys.stderr.write('arena launch: TERM timeout, sending SIGKILL\n')
        self._signal_pgs(snap, signal.SIGKILL)
        self._wait(snap, GRACE_KILL_S)

    @staticmethod
    def _signal_pgs(children: Iterable[tuple[str, subprocess.Popen]], sig: int) -> None:
        for _, proc in children:
            if proc.poll() is not None:
                continue
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(os.getpgid(proc.pid), sig)

    @staticmethod
    def _wait(children: Iterable[tuple[str, subprocess.Popen]], timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        for _, proc in children:
            remaining = max(0.0, deadline - time.monotonic())
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=remaining)
        return all(p.poll() is not None for _, p in children)

    def has_service(self, service: str) -> bool:
        return any(name == service for name, _ in self._node.get_service_names_and_types())

    def viz_namespaces(self) -> set[str]:
        return {name[: -len(VIZ_MANIFEST_SUFFIX)] for name, _ in self._node.get_topic_names_and_types() if name.endswith(VIZ_MANIFEST_SUFFIX)}

    def get_param_string(self, node_name: str, param: str, timeout_s: float = 5.0) -> str | None:
        cli = self._node.create_client(GetParameters, f'{node_name}/get_parameters')
        try:
            if not cli.wait_for_service(timeout_sec=timeout_s):
                return None
            future = cli.call_async(GetParameters.Request(names=[param]))
            rclpy.spin_until_future_complete(self._node, future, timeout_sec=timeout_s)
            if not future.done():
                return None
            result = future.result()
            if not result or not result.values:
                return None
            return result.values[0].string_value or None
        finally:
            self._node.destroy_client(cli)


def _viz_spawn_commands(ns: str, viz_args: dict[str, str]) -> list[tuple[str, list[str]]]:
    """One rviz per env, fanning out across robots when `viz.robot:=all`."""
    robot = viz_args.get('robot', '0')
    extras = [f'{k}:={v}' for k, v in viz_args.items() if k != 'robot']
    base = ['ros2', 'launch', 'rviz_utils', 'rviz_config.launch.py', f'ns:={ns}', *extras]
    if robot == 'all':
        return [(f'rviz_{ns}_r{i}', [*base, f'robot:={i}']) for i in range(_fleet_size(ns))]
    return [(f'rviz_{ns}', [*base, f'robot:={robot}'])]


def _fleet_size(ns: str) -> int:
    """Probe RobotFleet for `ns`; fall back to 1 if unknown."""
    import rclpy.qos
    from task_generator_msgs.msg import RobotFleet

    seen: list[int] = []
    node = rclpy.create_node('arena_supervisor_fleet_probe')
    qos = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
    sub = node.create_subscription(RobotFleet, f'{ns}/state/robots', lambda m: seen.append(len(m.robots)), qos)
    deadline = time.monotonic() + 5.0
    while not seen and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_subscription(sub)
    node.destroy_node()
    return seen[0] if seen else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Forward every k:=v to both runtime and env; supervisor-only knobs are env_n / rviz / viz.* / vla_server."""
    env_n = 1
    headless = False
    rviz = True
    rviz_set = False
    sim: str | None = None
    vla_server: str | None = None
    runtime_args: list[str] = []
    env_args: list[str] = []
    viz_args: dict[str, str] = {}

    for arg in argv:
        if ':=' not in arg:
            env_args.append(arg)
            continue
        key, _, value = arg.partition(':=')
        if key == 'env_n':
            env_n = int(value)
            continue
        if key == 'rviz':
            rviz_set = True
            rviz = value.lower() in ('true', '1')
            continue
        if key.startswith('viz.'):
            viz_args[key[len('viz.') :]] = value
            continue
        if key == 'vla_server':
            vla_server = value
            continue
        if key == 'sim':
            sim = value
        elif key == 'headless':
            headless = value.lower() in ('true', '1')
        runtime_args.append(arg)
        env_args.append(arg)

    if headless and not rviz_set:
        rviz = False

    return argparse.Namespace(
        env_n=env_n,
        headless=headless,
        rviz=rviz,
        sim=sim,
        vla_server=vla_server,
        runtime_args=runtime_args,
        env_args=env_args,
        viz_args=viz_args,
    )


def wait_until(
    predicate: Callable[[], bool],
    *,
    runtime_proc: subprocess.Popen | None,
    label: str,
    period_s: float = 1.0,
) -> bool:
    waited = 0
    while not predicate():
        if runtime_proc is not None and runtime_proc.poll() is not None:
            sys.stderr.write(f'arena launch: runtime exited before {label} (rc={runtime_proc.returncode})\n')
            return False
        time.sleep(period_s)
        waited += 1
        if waited % 10 == 0:
            sys.stderr.write(f'arena launch: waiting for {label} ({waited}s elapsed)\n')
    return True


def run(args: argparse.Namespace, sup: Supervisor) -> int:
    if args.vla_server is not None:
        import os as _os

        _features_dir = _os.environ.get(
            'ARENA_FEATURES_DIR',
            _os.path.join(_os.environ.get('ARENA_DIR', ''), '_meta', 'docker', 'features'),
        )
        _vla_main = _os.path.join(_features_dir, 'vla_server', 'main')
        sup.spawn('vla_server', ['bash', _vla_main, 'launch', args.vla_server])
        sys.stderr.write(f'arena launch: VLA server spawning (model={args.vla_server}); health endpoint: http://localhost:8000/health\n')

    if sup.has_service(REGISTER_ENV_SERVICE):
        if args.sim is not None:
            existing = sup.get_param_string('/arena', 'sim')
            if existing and existing != args.sim:
                sys.stderr.write(f'arena launch: sim mismatch (running: {existing}, requested: {args.sim}); shut down the runtime first\n')
                return 1
        sys.stderr.write('arena launch: attaching to existing runtime\n')
        runtime = None
    else:
        sup.spawn(
            'runtime',
            [
                'ros2',
                'launch',
                'arena_bringup',
                'arena_runtime.launch.py',
                *args.runtime_args,
            ],
        )
        runtime = sup.runtime()
        if not wait_until(
            lambda: sup.has_service(REGISTER_ENV_SERVICE),
            runtime_proc=runtime,
            label='/arena/register_env',
        ):
            return 1

    existing_ns = sup.viz_namespaces()
    target_count = len(existing_ns) + args.env_n

    for i in range(args.env_n):
        sup.spawn(
            f'env_{i}',
            [
                'ros2',
                'launch',
                'task_generator',
                'task_generator.launch.py',
                *args.env_args,
            ],
        )

    if args.rviz:
        if not wait_until(
            lambda: len(sup.viz_namespaces()) >= target_count,
            runtime_proc=runtime,
            label=f'{target_count} env(s)',
        ):
            return 1
        for ns in sorted(sup.viz_namespaces() - existing_ns):
            for role, cmd in _viz_spawn_commands(ns, args.viz_args):
                sup.spawn(role, cmd)

    if runtime is not None:
        runtime.wait()
        return runtime.returncode
    while sup.alive():
        time.sleep(1.0)
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    rclpy.init(args=[])
    node = rclpy.create_node('arena_supervisor')
    sup = Supervisor(node)

    signal.signal(signal.SIGINT, sup.signal_handler)
    signal.signal(signal.SIGTERM, sup.signal_handler)

    try:
        return run(args, sup)
    finally:
        sup.cleanup()
        with contextlib.suppress(Exception):
            node.destroy_node()
        with contextlib.suppress(Exception):
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
