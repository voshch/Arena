"""Backend dispatch for the arena_viz layer (rviz, rerun, ...).

Shared by `arena_bringup.supervisor` (used during `arena launch`) and
`_meta/tools/arena_cli/viz.py` (used by `arena viz`).
"""

from __future__ import annotations

import os
import time
from typing import Callable

BackendFn = Callable[[str, int, dict[str, str]], list[tuple[str, list[str]]]]


def fleet_size(ns: str) -> int:
    """Probe RobotFleet on `ns/state/robots`; fall back to 1 if unknown."""
    import rclpy
    import rclpy.qos
    from task_generator_msgs.msg import RobotFleet

    own_init = not rclpy.ok()
    seen: list[int] = []
    if own_init:
        rclpy.init(args=[])
    try:
        node = rclpy.create_node('arena_viz_fleet_probe')
        qos = rclpy.qos.QoSProfile(
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sub = node.create_subscription(
            RobotFleet,
            f'{ns}/state/robots',
            lambda m: seen.append(len(m.robots)),
            qos,
        )
        deadline = time.monotonic() + 5.0
        while not seen and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        node.destroy_subscription(sub)
        node.destroy_node()
    finally:
        if own_init:
            rclpy.shutdown()
    return seen[0] if seen else 1


def env_idx_from_ns(ns: str) -> int:
    """Parse the env index from a namespace (`/arena/env_3/task_generator_node` → 3). 0 on failure."""
    for part in reversed(ns.strip('/').split('/')):
        head, _, tail = part.rpartition('_')
        if head and tail.isdigit():
            return int(tail)
    return 0


def _rviz_commands(ns: str, env_idx: int, args: dict[str, str]) -> list[tuple[str, list[str]]]:
    del env_idx
    robot = args.get('robot', '0')
    extras = [f'{k}:={v}' for k, v in args.items() if k != 'robot']
    base = ['ros2', 'launch', 'rviz_utils', 'rviz_config.launch.py', f'ns:={ns}', *extras]
    if robot == 'all':
        return [(f'rviz_{ns}_r{i}', [*base, f'robot:={i}']) for i in range(fleet_size(ns))]
    return [(f'rviz_{ns}', [*base, f'robot:={robot}'])]


def _rerun_commands(ns: str, env_idx: int, args: dict[str, str]) -> list[tuple[str, list[str]]]:
    web = int(args.get('web_port', 9090)) + env_idx
    grpc = int(args.get('grpc_port', 9876)) + env_idx
    extras = [f'{k}:={v}' for k, v in args.items() if k not in ('web_port', 'grpc_port')]
    return [
        (
            f'rerun_{ns}',
            [
                'ros2',
                'launch',
                'rerun_utils',
                'rerun_bridge.launch.py',
                f'ns:={ns}',
                f'web_port:={web}',
                f'grpc_port:={grpc}',
                *extras,
            ],
        )
    ]


BACKENDS: dict[str, BackendFn] = {
    "rviz": _rviz_commands,
    "rerun": _rerun_commands,
}


def resolve_backends(args: dict[str, str]) -> list[str]:
    """Pop `backend` from args; return the list of backend names. Defaults to
    `$ARENA_VIZ_BACKEND` then `rviz`. Raises ValueError on unknown names."""
    raw = args.pop('backend', os.environ.get('ARENA_VIZ_BACKEND', 'rviz'))
    backends = [b.strip() for b in raw.split(',') if b.strip()]
    unknown = [b for b in backends if b not in BACKENDS]
    if unknown:
        raise ValueError(f"unknown backend(s) {unknown!r} (known: {sorted(BACKENDS)})")
    return backends


def viz_commands(ns: str, env_idx: int, viz_args: dict[str, str]) -> list[tuple[str, list[str]]]:
    """Expand viz_args into one or more (label, cmd) tuples per backend.
    `viz_args` is not mutated."""
    args = dict(viz_args)
    backends = resolve_backends(args)
    out: list[tuple[str, list[str]]] = []
    for bk in backends:
        out.extend(BACKENDS[bk](ns, env_idx, args))
    return out
