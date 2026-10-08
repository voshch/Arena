from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid

import pytest


@pytest.fixture(autouse=True)
def _ros_gate(requires_auditory):
    pytest.importorskip("rclpy")
    pytest.importorskip("task_generator_msgs.msg")


def _emission(**overrides: object):
    from task_generator.simulators.acoustics import SoundEmission

    fields: dict[str, object] = {
        "entity": "env_0/radio",
        "name": "radio",
        "group_id": "env_0/radio",
        "kind": "music",
        "asset_id": "radio_loop",
        "variant_id": "radio_loop_01",
        "model": "wav",
        "tags": ("radio", "music"),
        "loop": True,
        "reference_distance_m": 1.0,
        "level_db": 62.0,
        "seed": 7,
        "frame_id": "map",
        "position": (1.0, 2.0, 1.2),
        "yaw": 0.5,
        "active": True,
        "program_start_ns": 0,
    }
    fields.update(overrides)
    return SoundEmission(**fields)


def _with_backend(body):
    import rclpy
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from task_generator.simulators.acoustics.arena import ArenaAcousticsSimulator

    class _Node(ServiceNamespace, AsyncNode):
        pass

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main():
        node = _Node(f"acoustics_arena_{suffix}", namespace=str(ns))
        stop = threading.Event()

        def _spin() -> None:
            while not stop.is_set():
                rclpy.spin_once(node, timeout_sec=0.02)

        thread = threading.Thread(target=_spin, daemon=True)
        thread.start()
        try:
            return await body(node, ArenaAcousticsSimulator(node=node, namespace=ns))
        finally:
            stop.set()
            thread.join(timeout=1.0)
            node.destroy_node()

    return asyncio.run(main())


def test_source_msg_maps_wav_variants_to_wav_loop() -> None:
    from builtin_interfaces.msg import Time
    from task_generator.simulators.acoustics.arena.arena import ArenaAcousticsSimulator

    msg = ArenaAcousticsSimulator.source_msg(_emission(), Time(sec=3))
    assert (msg.source.id, msg.source.kind, msg.source.asset_id, msg.source.variant_id, msg.source.model) == ("environment:env_0/radio", "music", "radio_loop", "radio_loop_01", "wav_loop")
    assert (msg.source.agent_kind, msg.source.agent_name, list(msg.source.tags)) == ("environment", "radio", ["radio", "music"])
    assert msg.source.level_db == pytest.approx(62.0)
    assert (msg.source.position.x, msg.source.position.y, msg.source.position.z) == pytest.approx((1.0, 2.0, 1.2))
    assert msg.source.active
    assert (msg.header.frame_id, msg.header.stamp.sec) == ("map", 3)


def test_source_msg_keeps_procedural_models() -> None:
    from builtin_interfaces.msg import Time
    from task_generator.simulators.acoustics.arena.arena import ArenaAcousticsSimulator

    assert ArenaAcousticsSimulator.source_msg(_emission(model="drivetrain"), Time()).source.model == "drivetrain"


def test_plugins_and_recorded_topics_come_from_the_auditory_api(rclpy_context) -> None:
    from arena_auditory.api import recorded_topics, rviz_plugins

    async def body(node, backend):
        return node.get_fully_qualified_name(), backend.plugins(), backend.recorded_topics()

    fqn, plugins, topics = _with_backend(body)
    assert [(p.role, p.class_name, p.name, json.loads(p.properties_json)) for p in plugins] == [(role, cls, name, dict(props)) for role, cls, name, props in rviz_plugins(fqn)]
    assert [(t.key, t.topic, t.msg_type, t.robot_scoped, t.throttled, t.qos_transient_local, t.reliable, t.depth, t.recorded) for t in topics] == [tuple(row) for row in recorded_topics()]


def test_a_removed_sounding_source_repeats_inactive(rclpy_context) -> None:
    from arena_auditory_msgs.msg import ContinuousAudioSourceState
    from arena_rclpy_mixins.qos import best_effort
    from task_generator.simulators.acoustics.arena.arena import CONTINUOUS_QOS_DEPTH

    from arena_auditory.api import CONTINUOUS_AUDIO_SOURCES, INACTIVE_REPEATS

    async def body(node, backend):
        received: list[tuple[str, bool]] = []
        sub = node.create_subscription(
            ContinuousAudioSourceState,
            str(node.service_namespace(CONTINUOUS_AUDIO_SOURCES)),
            lambda msg: received.append((msg.source.id, msg.source.active)),
            best_effort(CONTINUOUS_QOS_DEPTH),
        )
        deadline = time.monotonic() + 5.0
        while backend._source_publisher.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "subscription never matched"
            await asyncio.sleep(0.02)
        backend.emit_sounds([_emission(), _emission(entity="env_0/alarm", active=False)])
        for _ in range(INACTIVE_REPEATS + 2):
            await asyncio.sleep(0.05)
            backend.emit_sounds([])
        deadline = time.monotonic() + 5.0
        while len(received) < 2 + INACTIVE_REPEATS:
            assert time.monotonic() < deadline, f"received so far: {received}"
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        node.destroy_subscription(sub)
        return received

    received = _with_backend(body)
    assert received[:2] == [("environment:env_0/radio", True), ("environment:env_0/alarm", False)]
    assert received[2:] == [("environment:env_0/radio", False)] * INACTIVE_REPEATS


def test_a_source_cleared_at_reset_does_not_retire(rclpy_context) -> None:
    from arena_auditory_msgs.msg import ContinuousAudioSourceState
    from arena_rclpy_mixins.qos import best_effort
    from task_generator.simulators.acoustics.arena.arena import CONTINUOUS_QOS_DEPTH

    from arena_auditory.api import CONTINUOUS_AUDIO_SOURCES, INACTIVE_REPEATS

    async def body(node, backend):
        received: list[tuple[str, bool]] = []
        sub = node.create_subscription(
            ContinuousAudioSourceState,
            str(node.service_namespace(CONTINUOUS_AUDIO_SOURCES)),
            lambda msg: received.append((msg.source.id, msg.source.active)),
            best_effort(CONTINUOUS_QOS_DEPTH),
        )
        deadline = time.monotonic() + 5.0
        while backend._source_publisher.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "subscription never matched"
            await asyncio.sleep(0.02)
        backend.emit_sounds([_emission()])
        backend.clear_sounds()
        for _ in range(INACTIVE_REPEATS + 2):
            await asyncio.sleep(0.05)
            backend.emit_sounds([])
        await asyncio.sleep(0.5)
        node.destroy_subscription(sub)
        return received

    assert _with_backend(body) == [("environment:env_0/radio", True)]
