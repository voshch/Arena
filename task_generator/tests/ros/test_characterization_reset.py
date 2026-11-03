"""TM_Characterization.reset places the robots and starts the sweep."""

from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_people_msgs.msg")


def test_reset_plans_the_start_and_publishes_the_schedule(rclpy_context: object) -> None:
    import rclpy
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.env.task_api import WorldOccupancy
    from arena_runtime.shared import Position
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    from task_generator.modes import Modes
    from arena_runtime.env.task_layer import declare_mode_schemas

    from task_generator.tasks import layer
    from task_generator.tasks.registry import ROBOTS_MODES

    world = SimpleNamespace(map=SimpleNamespace(occupancy=SimpleNamespace(grid=np.full((4, 4), WorldOccupancy.EMPTY)), tf_grid2pos=lambda cell: Position(*cell)))
    key = Modes.TM_Robots.CHARACTERIZATION
    suffix = uuid.uuid4().hex[:8]
    received: list[str] = []

    async def main() -> object:
        node = ArenaMixinNode(f"characterization_{suffix}")
        listener = rclpy.create_node(f"characterization_listener_{suffix}")
        try:
            declare_mode_schemas(layer, node)
            manager = SimpleNamespace(name="r0", namespace=f"/characterization_{suffix}/r0", model_name="jackal")
            ctx = SimpleNamespace(robots={"r0": manager}, world_manager=world, conf=SimpleNamespace(RNG=EpisodeRng()))
            mode = ROBOTS_MODES.get(key)(ctx=ctx, namespace=ROBOTS_MODES.meta(key).namespace, node=node)
            plan = await mode.reset()
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            listener.create_subscription(String, f"{manager.namespace}/characterization_schedule", lambda msg: received.append(msg.data), qos)
            for _ in range(100):
                rclpy.spin_once(listener, timeout_sec=0.05)
                if received:
                    break
            await mode.teardown()
            return plan
        finally:
            listener.destroy_node()
            node.destroy_node()

    plan = asyncio.run(main())

    assert list(plan.robots) == ["r0"]
    assert (plan.robots["r0"].start_pose.position.x, plan.robots["r0"].start_pose.position.y) == (2.0, 2.0)
    assert plan.robots["r0"].task is None
    assert len(received) == 1
    record = json.loads(received[0])
    assert record["robot"] == "r0"
    assert record["phases"]
