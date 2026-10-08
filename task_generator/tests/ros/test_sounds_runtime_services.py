from __future__ import annotations

import asyncio
import uuid

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    pytest.importorskip("rclpy")
    pytest.importorskip("task_generator_msgs.msg")


def _during_reset(call):
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from task_generator.simulators.acoustics.noop import NoopAcousticsSimulator
    from task_generator.tasks.modules.sounds.impl import Mod_Sounds

    class _Node(ServiceNamespace, AsyncNode):
        pass

    suffix = f"t_{uuid.uuid4().hex[:8]}"
    ns = Namespace(f"/test/{suffix}")

    async def main():
        node = _Node(f"sounds_services_{suffix}", namespace=str(ns))
        try:
            node._reset_lock = asyncio.Lock()
            node._acoustics_simulator = NoopAcousticsSimulator(node=node, namespace=ns)
            module = Mod_Sounds(node=node, ctx=None, namespace=ns, task=None)
            async with node._reset_lock:
                pending = asyncio.ensure_future(call(module))
                await asyncio.sleep(0.05)
                waited = not pending.done()
            return waited, await pending
        finally:
            node.destroy_node()

    return asyncio.run(main())


def test_remove_sound_during_a_reset_waits_for_the_reset_to_finish(rclpy_context) -> None:
    from task_generator_msgs.srv import RemoveSound

    waited, response = _during_reset(lambda module: module._remove_sound(RemoveSound.Request(entity="radio"), RemoveSound.Response()))
    assert waited
    assert not response.success
    assert response.error_msg == "unknown or non-removable sound 'radio'"


def test_spawn_sound_during_a_reset_waits_for_the_reset_to_finish(rclpy_context) -> None:
    from task_generator_msgs.srv import SpawnSound

    waited, response = _during_reset(lambda module: module._spawn_sound(SpawnSound.Request(kind="no_such_kind"), SpawnSound.Response()))
    assert waited
    assert not response.success
