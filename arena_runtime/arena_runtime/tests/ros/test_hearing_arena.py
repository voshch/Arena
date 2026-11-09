"""The arena hearing backend records the hearing topics arena_hearing names for the env's front end."""

from __future__ import annotations

import asyncio
import uuid

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("arena_hearing")


def test_recorded_topics_are_arena_hearings_rows_for_the_front_end(rclpy_context) -> None:
    import rclpy
    from arena_hearing.constants import recorded_topics
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace

    from arena_runtime.hearing.arena.arena import ArenaHearing

    suffix = f"t_{uuid.uuid4().hex[:8]}"

    async def main():
        node = ArenaMixinNode(f"hearing_arena_{suffix}", namespace=f"/test/{suffix}", parameter_overrides=[rclpy.Parameter("robot.hearing", value="call")], automatically_declare_parameters_from_overrides=True)
        try:
            return ArenaHearing(node=node, namespace=Namespace(f"/test/{suffix}")).recorded_topics()
        finally:
            node.destroy_node()

    topics = asyncio.run(main())
    assert topics
    assert [(t.key, t.topic, t.msg_type, t.robot_scoped, t.throttled, t.qos_transient_local, t.reliable, t.depth, t.recorded) for t in topics] == [tuple(row) for row in recorded_topics(("call",))]
