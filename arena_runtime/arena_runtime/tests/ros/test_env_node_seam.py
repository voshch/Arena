"""EnvNode's episode loop hooks, driven by a minimal subclass against an in-process ArenaRuntimeNode on the dummy sim."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_runtime_msgs.msg")

TIMEOUT = 60.0
ENTITY = "seam_entity"
DISPLAY = "Seam"


@dataclass
class Record:
    order: list[str] = field(default_factory=list)
    at_task: dict = field(default_factory=dict)
    at_after_setup: dict = field(default_factory=dict)
    at_episode_teardown: dict = field(default_factory=dict)
    ticks: int = 0
    ended: list[dict] = field(default_factory=list)
    tick_loop_task: object = None
    entities: list[str] = field(default_factory=list)
    displays: list[str] = field(default_factory=list)
    late_watchdog: str = ""
    watchdog_ticks: int = 0
    watchdog_ticks_at_teardown: int = -1
    failure: str = ""


def _env_class(record: Record) -> type:
    from arena_runtime_msgs.msg import AdapterDisplay, SemanticEntityState

    from arena_runtime.env.node import EnvNode

    class Env(EnvNode):
        async def _set_up_task(self) -> None:
            record.order.append("set_up_task")
            record.at_task = {"robots": self._robots_manager is not None, "world": self._world_manager.loaded_world, "initialized": self._initialized.value}

        def _on_watchdog(self) -> None:
            record.watchdog_ticks += 1

        def _after_setup(self) -> None:
            record.order.append("after_setup")
            record.at_after_setup = {"initialized": self._initialized.value}

        async def _on_tick(self, ended: dict) -> None:
            record.ticks += 1
            record.ended.append(dict(ended))
            record.tick_loop_task = asyncio.current_task()

        def _extra_semantic_states(self) -> list[SemanticEntityState]:
            return [SemanticEntityState(entity=ENTITY, kind="seam")]

        def _interaction_displays(self) -> list[AdapterDisplay]:
            return [AdapterDisplay(name=DISPLAY, topic="/seam", topic_type="visualization_msgs/MarkerArray")]

        async def _teardown_episode(self) -> None:
            record.order.append("teardown_episode")
            record.at_episode_teardown = {"tick_loop_done": self._tick_loop_task.done()}

    return Env


async def _until(predicate: object, what: str) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.05)


async def _scenario(record: Record) -> None:
    import rclpy.parameter
    import rclpy.qos
    from arena_runtime_msgs.msg import AdapterVizManifest, SemanticSnapshot

    from arena_runtime.testing.env_host import arena_on_dummy

    async with arena_on_dummy(on_taken=pytest.skip) as rig:
        overrides = [
            rclpy.parameter.Parameter("env_id", value=rig.env_id),
            rclpy.parameter.Parameter("world", value="map_empty"),
            rclpy.parameter.Parameter("sim", value="dummy"),
            rclpy.parameter.Parameter("human", value="dummy"),
            rclpy.parameter.Parameter("acoustics", value="none"),
        ]
        env = _env_class(record)("seam_env", namespace=f"/{rig.namespace}", parameter_overrides=overrides, automatically_declare_parameters_from_overrides=True)
        rig.executor.add_node(env)
        try:
            latched = rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL)
            rig.probe.create_subscription(
                SemanticSnapshot,
                env.service_namespace("state", "semantics"),
                lambda msg: record.entities.extend(state.entity for state in msg.entities),
                latched,
            )
            rig.probe.create_subscription(
                AdapterVizManifest,
                env.service_namespace("state", "viz_manifest"),
                lambda msg: record.displays.extend(display.name for display in msg.env_displays),
                latched,
            )

            await asyncio.wait_for(env.setup(), TIMEOUT)
            await _until(lambda: record.ticks > 0, "the first tick")
            env.on_task_submitted(None)
            await _until(lambda: ENTITY in record.entities, "the extra entity in a snapshot")
            await _until(lambda: DISPLAY in record.displays, "the interaction display in the manifest")

            assert record.tick_loop_task is env._tick_loop_task
            await _until(lambda: record.watchdog_ticks > 0, "a watchdog tick")
            await env.teardown()
            record.watchdog_ticks_at_teardown = record.watchdog_ticks
            await asyncio.sleep(1.5)
        finally:
            watchdog = env._arena_watchdog_timer
            rig.executor.remove_node(env)
            env.destroy_node()
            try:
                watchdog.callback()
            except Exception as e:
                record.late_watchdog = repr(e)


@pytest.fixture(scope="module")
def scenario(rclpy_context: object) -> Record:
    result = Record()
    try:
        asyncio.run(_scenario(result))
    except Exception:
        result.failure = traceback.format_exc()
    return result


@pytest.fixture()
def record(scenario: Record) -> Record:
    if scenario.failure:
        pytest.skip(f"the live scenario failed, see test_the_live_scenario_runs: {scenario.failure.strip().splitlines()[-1]}")
    return scenario


def test_the_live_scenario_runs(scenario: Record) -> None:
    if scenario.failure:
        pytest.fail(scenario.failure, pytrace=False)


def test_the_env_layer_imports_without_the_task_layer() -> None:
    code = "import sys, arena_runtime.env.episode_node, arena_runtime.env.episode_mcp, arena_runtime.arena_runtime_node\nassert not [m for m in sys.modules if m.split('.')[0] == 'task_generator']"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr


def test_setup_builds_the_task_on_the_managers_before_the_env_is_initialized(record: Record) -> None:
    assert record.order[:2] == ["set_up_task", "after_setup"]
    assert record.at_task == {"robots": True, "world": "map_empty", "initialized": False}
    assert record.at_after_setup == {"initialized": True}


def test_on_tick_runs_inside_the_tick_loop(record: Record) -> None:
    assert record.ticks > 0


def test_on_tick_is_handed_the_phases_that_ended_which_is_none_without_robots(record: Record) -> None:
    assert record.ended
    assert all(ended == {} for ended in record.ended)


def test_extra_semantic_states_reach_the_snapshot(record: Record) -> None:
    assert ENTITY in record.entities


def test_interaction_displays_reach_the_viz_manifest(record: Record) -> None:
    assert DISPLAY in record.displays


def test_teardown_stops_the_tick_loop_before_the_episode(record: Record) -> None:
    assert record.at_episode_teardown == {"tick_loop_done": True}
    assert record.order[2:] == ["teardown_episode"]


def test_a_watchdog_tick_after_the_node_is_destroyed_leaves_it_alone(record: Record) -> None:
    assert record.late_watchdog == ""


def test_the_arena_watchdog_drives_the_watchdog_hook(record: Record) -> None:
    assert record.watchdog_ticks > 0


def test_the_watchdog_hook_stops_at_teardown(record: Record) -> None:
    assert record.watchdog_ticks == record.watchdog_ticks_at_teardown
