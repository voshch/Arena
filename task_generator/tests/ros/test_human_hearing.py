from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("arena_people_msgs.msg")
    pytest.importorskip("task_generator_msgs.msg")
    pytest.importorskip("arena_runtime_msgs.srv")


def _spin_in_background(rclpy, node, stop: threading.Event) -> threading.Thread:
    def _spin() -> None:
        while not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.02)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    return thread


def _heard_state(listener_id: str, sound_type: str, audible: bool):
    from task_generator_msgs.msg import ContinuousHeardSoundState

    msg = ContinuousHeardSoundState()
    msg.listener_id = listener_id
    msg.sound_type = sound_type
    msg.active = True
    msg.audible = audible
    return msg


def test_heard_sounds_drive_notify_stimulus_edge_triggered(rclpy_context):
    import rclpy
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from task_generator.auditory.qos_profiles import continuous_audio_qos
    from task_generator.manager.realizer import Realizer
    from task_generator.simulators.human import TOPIC_CONTINUOUS_HEARD_SOUNDS
    from task_generator.simulators.human.noop import NoopHumanSimulator
    from task_generator_msgs.msg import ContinuousHeardSoundState

    class _Node(ServiceNamespace, AsyncNode):
        pass

    class _Recording(NoopHumanSimulator):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.calls: list[tuple[int, str, float]] = []

        async def notify_stimulus(self, agent_id: int, stimulus: str, intensity: float) -> None:
            self.calls.append((agent_id, stimulus, intensity))

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main() -> list[tuple[int, str, float]]:
        node = _Node(f"human_hearing_{suffix}", namespace=str(ns))
        stop = threading.Event()
        thread = _spin_in_background(rclpy, node, stop)
        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=""))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            human = _Recording(node=node, namespace=ns, simulator=simulator, realizer=realizer)
            topic = str(node.service_namespace(TOPIC_CONTINUOUS_HEARD_SOUNDS))
            publisher = node.create_publisher(ContinuousHeardSoundState, topic, continuous_audio_qos())

            deadline = time.monotonic() + 5.0
            while publisher.get_subscription_count() == 0:
                assert time.monotonic() < deadline, "subscription never matched"
                await asyncio.sleep(0.02)

            for audible in (True, True, False):
                publisher.publish(_heard_state("agent:3", "alarm", audible))
                await asyncio.sleep(0.1)

            deadline = time.monotonic() + 5.0
            while len(human.calls) < 2:
                assert time.monotonic() < deadline, f"calls so far: {human.calls}"
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
            return list(human.calls)
        finally:
            stop.set()
            thread.join(timeout=1.0)
            node.destroy_node()

    assert asyncio.run(main()) == [(3, "alarm", 1.0), (3, "alarm", 0.0)]
