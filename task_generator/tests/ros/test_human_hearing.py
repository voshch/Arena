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


def _heard_state(listener_id: str, sound_type: str, audible: bool, *, source_id: str = "environment:alarm_0", active: bool = True):
    from arena_auditory_msgs.msg import ContinuousHeardSoundState

    msg = ContinuousHeardSoundState()
    msg.reception.listener_id = listener_id
    msg.source.id = source_id
    msg.source.kind = sound_type
    msg.source.active = active
    msg.reception.audible = audible
    return msg


def _stimuli(backend: str, drive, expected: int) -> list[tuple[int, str, float]]:
    import rclpy
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.sim.dummy_simulator import DummySimulator
    from task_generator.manager.realizer import Realizer
    from task_generator.simulators.human.noop import NoopHumanSimulator

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
            if backend == "arena":
                from task_generator.simulators.acoustics.arena import ArenaAcousticsSimulator

                acoustics = ArenaAcousticsSimulator(node=node, namespace=ns)
            else:
                from task_generator.simulators.acoustics.noop import NoopAcousticsSimulator

                acoustics = NoopAcousticsSimulator(node=node, namespace=ns)
            human = _Recording(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=acoustics)
            await drive(node, human)

            deadline = time.monotonic() + 5.0
            while len(human.calls) < expected:
                assert time.monotonic() < deadline, f"calls so far: {human.calls}"
                await asyncio.sleep(0.02)
            await asyncio.sleep(0.2)
            return list(human.calls)
        finally:
            stop.set()
            thread.join(timeout=1.0)
            node.destroy_node()

    return asyncio.run(main())


def _published(messages: list):
    async def drive(node, _human) -> None:
        from arena_auditory_msgs.msg import ContinuousHeardSoundState
        from arena_rclpy_mixins.qos import best_effort
        from task_generator.simulators.acoustics.arena.arena import CONTINUOUS_QOS_DEPTH

        from arena_auditory.api import CONTINUOUS_HEARD_SOUNDS

        publisher = node.create_publisher(ContinuousHeardSoundState, str(node.service_namespace(CONTINUOUS_HEARD_SOUNDS)), best_effort(CONTINUOUS_QOS_DEPTH))
        deadline = time.monotonic() + 5.0
        while publisher.get_subscription_count() == 0:
            assert time.monotonic() < deadline, "subscription never matched"
            await asyncio.sleep(0.02)
        for msg in messages:
            publisher.publish(msg)
            await asyncio.sleep(0.1)

    return drive


def _heard(calls: list[tuple[int, str, str, bool]]):
    async def drive(_node, human) -> None:
        from task_generator.simulators.acoustics import PedestrianHearing

        hearing = PedestrianHearing(human)
        for call in calls:
            await hearing.on_heard(*call)

    return drive


def test_noop_acoustics_gives_pedestrians_no_hearing(rclpy_context):
    async def drive(_node, human) -> None:
        assert human._hearing is None

    assert _stimuli("noop", drive, 0) == []


def test_pedestrian_hearing_holds_a_kind_until_every_source_falls_silent(rclpy_context):
    calls = [
        (3, "alarm", "environment:alarm_a", True),
        (3, "alarm", "environment:alarm_b", True),
        (3, "alarm", "environment:alarm_a", False),
        (4, "alarm", "environment:alarm_a", False),
        (3, "alarm", "environment:alarm_b", False),
    ]
    assert _stimuli("noop", _heard(calls), 2) == [(3, "alarm", 1.0), (3, "alarm", 0.0)]


@pytest.mark.usefixtures("requires_auditory")
def test_heard_sounds_drive_notify_stimulus_edge_triggered(rclpy_context):
    messages = [_heard_state("agent:3", "alarm", audible) for audible in (True, True, False)]
    assert _stimuli("arena", _published(messages), 2) == [(3, "alarm", 1.0), (3, "alarm", 0.0)]


@pytest.mark.usefixtures("requires_auditory")
def test_inactive_sources_are_not_heard(rclpy_context):
    messages = [
        _heard_state("agent:3", "alarm", True, active=False),
        _heard_state("agent:3", "alarm", True, active=True),
        _heard_state("agent:3", "alarm", True, active=False),
    ]
    assert _stimuli("arena", _published(messages), 2) == [(3, "alarm", 1.0), (3, "alarm", 0.0)]


@pytest.mark.usefixtures("requires_auditory")
def test_two_sources_of_one_type_hold_the_stimulus_until_both_fall_silent(rclpy_context):
    messages = [
        _heard_state("agent:3", "alarm", audible, source_id=source)
        for source, audible in (
            ("environment:alarm_a", True),
            ("environment:alarm_b", True),
            ("environment:alarm_a", False),
            ("environment:alarm_a", True),
            ("environment:alarm_b", False),
            ("environment:alarm_a", False),
        )
    ]
    assert _stimuli("arena", _published(messages), 2) == [(3, "alarm", 1.0), (3, "alarm", 0.0)]


@pytest.mark.usefixtures("requires_auditory")
def test_non_agent_listeners_are_ignored(rclpy_context):
    messages = [
        _heard_state("robot:rob", "alarm", True),
        _heard_state("agent:x", "alarm", True),
        _heard_state("agent:3", "alarm", True),
    ]
    assert _stimuli("arena", _published(messages), 1) == [(3, "alarm", 1.0)]
