"""A second, generated task runtime runs a reset cycle through the seam, with task_generator not importable at all."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.msg")

RUNTIME = """
import json

from arena_runtime.env.task_layer import ResetPlan, TaskModeOverrides, validate_modes as _validate_selection

log = []
NAMESPACES = {"patrol": {"loop": "runtime.loop", "sweep": "runtime.sweep"}, "terrain": {"flat": "runtime.flat"}, "extras": {"trace": "runtime.trace"}}


class Params:
    def __init__(self):
        self.modes = {"patrol": ["loop"], "terrain": ["flat"], "extras": ["trace"]}

    def resolve(self, modes):
        return {**self.modes, **modes}

    def select(self, modes):
        log.append("select " + json.dumps(modes, sort_keys=True))
        self.modes = {**self.modes, **modes}


class Task:
    @property
    def done(self):
        async def finished():
            return True

        return finished()

    async def reset_robots(self, record):
        log.append(f"reset_robots {record.seed}")
        return ResetPlan()

    async def reset_obstacles(self, plan, record):
        log.append(f"reset_obstacles {record.seed}")

    async def tick(self, ended, assigned):
        pass

    def after_reset(self):
        log.append("after_reset")

    async def teardown(self):
        log.append("teardown")


def declare_task_params(node):
    return Params()


def declare_mode_schema(node, axis, mode):
    log.append(f"declare_mode_schema {axis} {mode}")
    node.declare_parameter(f"{NAMESPACES[axis][mode]}.speed", 1.0)


async def create_task(*, node, ctx, params):
    log.append("create_task")
    return Task()


def validate_modes(modes):
    return _validate_selection(modes, {axis: list(names) for axis, names in NAMESPACES.items()}, ("extras",))


def mode_namespaces():
    return NAMESPACES


def fleet_only_robots():
    return []


async def next_episode(*, previous, queued):
    log.append(f"next_episode {previous.episode_id} {previous.outcome_state}")
    if previous.episode_id == 0:
        return TaskModeOverrides(modes={"patrol": ["sweep"]}, seed=41)
    return TaskModeOverrides(modes={"patrol": ["loop"], "extras": []}, seed=42)
"""

DRIVER = """
import asyncio
import importlib.abc
import json
import sys

import rclpy
import rclpy.parameter
import rclpy.qos


class NoTaskGenerator(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "task_generator":
            raise ModuleNotFoundError(name)


sys.meta_path.insert(0, NoTaskGenerator())

import arena_runtime_msgs.srv
from arena_runtime_msgs.msg import EpisodeRecord

from arena_runtime.env.episode_node import EpisodeNode
from arena_runtime.env.task_layer import decode_modes, decode_namespaces
from arena_runtime.testing.env_host import arena_on_dummy


def summary(msg):
    return [msg.episode_id, msg.world, msg.seed, decode_modes(msg.modes)]


async def main():
    out = {}
    async with arena_on_dummy() as rig:
        overrides = [
            rclpy.parameter.Parameter("env_id", value=rig.env_id),
            rclpy.parameter.Parameter("world", value="map_empty"),
            rclpy.parameter.Parameter("sim", value="dummy"),
            rclpy.parameter.Parameter("human", value="dummy"),
            rclpy.parameter.Parameter("acoustics", value="none"),
            rclpy.parameter.Parameter("task_layer", value="generated_runtime"),
            rclpy.parameter.Parameter("task.episode.auto_reset", value=False),
        ]
        env = EpisodeNode(namespace=f"/{rig.namespace}", parameter_overrides=overrides)
        rig.executor.add_node(env)
        try:
            await asyncio.wait_for(env.setup(), 60.0)
            episodes, queue = [], []
            latched = rclpy.qos.QoSProfile(depth=50, reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
            rig.probe.create_subscription(EpisodeRecord, f"{env.get_fully_qualified_name()}/state/episode", episodes.append, latched)
            rig.probe.create_subscription(EpisodeRecord, f"{env.get_fully_qualified_name()}/state/queue", queue.append, latched)
            out["task_layer_param"] = env.rosparam[str].get("task_layer", "")
            queried = await env._cb_query_task_modes(arena_runtime_msgs.srv.QueryTaskModes.Request(), arena_runtime_msgs.srv.QueryTaskModes.Response())
            out["queried"] = decode_modes(queried.modes)
            out["namespaces"] = decode_namespaces(queried.namespaces)
            env.trigger_activate()
            results = [await asyncio.wait_for(env._run_episode(), 60.0)]
            await asyncio.sleep(1.0)
            out["preview"] = summary(queue[-1])
            results.append(await asyncio.wait_for(env._run_episode(), 60.0))
            await asyncio.sleep(1.0)
            out["states"] = [result.state for result in results]
            out["success"] = results[0].SUCCESS
            running = list({msg.episode_id: msg for msg in episodes if msg.outcome_state == EpisodeRecord.RUNNING}.values())
            out["built"] = [summary(msg) for msg in running]
            out["record_layers"] = [msg.task_layer for msg in running]
            out["params"] = [sorted(param.name for param in msg.params) for msg in running]
            await env.teardown()
        finally:
            rig.executor.remove_node(env)
            env.destroy_node()
    import generated_runtime

    out["log"] = generated_runtime.log
    out["task_generator_loaded"] = sorted(name for name in sys.modules if name.split(".")[0] == "task_generator")
    print("RESULT " + json.dumps(out))


rclpy.init()
asyncio.run(main())
"""


def test_a_generated_runtime_runs_two_episodes_through_the_seam_without_task_generator(tmp_path: Path) -> None:
    from arena_runtime.testing.task_layer import register_task_layer

    register_task_layer(tmp_path, "generated_runtime", RUNTIME)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), os.environ.get("PYTHONPATH", "")])}
    env.pop("ARENA_TASK_LAYER", None)
    done = subprocess.run([sys.executable, "-c", DRIVER], env=env, capture_output=True, text=True, timeout=180, check=False)
    lines = [line for line in done.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines and "already running" in done.stderr:
        pytest.skip("another arena runtime holds the domain")
    assert lines, done.stderr[-2000:]
    out = json.loads(lines[0].removeprefix("RESULT "))
    success = out["success"]

    assert out["task_generator_loaded"] == []
    assert out["task_layer_param"] == "generated_runtime"
    assert out["queried"] == {"patrol": ["loop", "sweep"], "terrain": ["flat"], "extras": ["trace"]}
    assert out["namespaces"] == {"patrol": {"loop": "runtime.loop", "sweep": "runtime.sweep"}, "terrain": {"flat": "runtime.flat"}, "extras": {"trace": "runtime.trace"}}
    assert out["record_layers"] == ["generated_runtime", "generated_runtime"]
    assert out["states"] == [success, success]

    first, second = out["built"]
    assert first == [1, "map_empty", 41, {"patrol": ["sweep"], "terrain": ["flat"], "extras": ["trace"]}]
    assert second == [2, "map_empty", 42, {"patrol": ["loop"], "terrain": ["flat"], "extras": []}]
    assert out["preview"] == second
    assert out["params"] == [["runtime.flat.speed", "runtime.sweep.speed", "runtime.trace.speed"], ["runtime.flat.speed", "runtime.loop.speed"]]

    assert out["log"] == [
        "declare_mode_schema patrol loop",
        "declare_mode_schema patrol sweep",
        "declare_mode_schema terrain flat",
        "declare_mode_schema extras trace",
        "create_task",
        "next_episode 0 0",
        'select {"extras": ["trace"], "patrol": ["sweep"], "terrain": ["flat"]}',
        "reset_robots 41",
        "reset_obstacles 41",
        "after_reset",
        f"next_episode 1 {success}",
        'select {"extras": [], "patrol": ["loop"], "terrain": ["flat"]}',
        "reset_robots 42",
        "reset_obstacles 42",
        "after_reset",
        f"next_episode 2 {success}",
        "teardown",
    ]
