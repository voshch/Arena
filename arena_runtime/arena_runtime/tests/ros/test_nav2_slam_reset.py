from __future__ import annotations

import asyncio
import threading
import typing
import uuid
from types import SimpleNamespace

import pytest

pytest.importorskip("rclpy")
pytest.importorskip("slam_toolbox")

import lifecycle_msgs.msg
import lifecycle_msgs.srv
from nav2_msgs.srv import ClearEntireCostmap
from slam_toolbox.srv import Reset

from arena_runtime.env.adapters.mobile.nav2 import Nav2Adapter

ACTIVE = lifecycle_msgs.msg.State.PRIMARY_STATE_ACTIVE
INACTIVE = lifecycle_msgs.msg.State.PRIMARY_STATE_INACTIVE


class _Stack:
    """In-process slam_toolbox and global costmap services of one robot, recording every reset and clear."""

    def __init__(self, node: typing.Any, namespace: str, slam_state: int, costmap_up: bool = True) -> None:
        self.slam_state = slam_state
        self.resets: list[bool] = []
        self.clears = 0
        node.create_service(lifecycle_msgs.srv.GetState, f"{namespace}/slam_toolbox/get_state", self._slam_get_state)
        node.create_service(Reset, f"{namespace}/slam_toolbox/reset", self._reset)
        if costmap_up:
            node.create_service(lifecycle_msgs.srv.GetState, f"{namespace}/global_costmap/global_costmap/get_state", self._costmap_get_state)
            node.create_service(ClearEntireCostmap, f"{namespace}/global_costmap/clear_entirely_global_costmap", self._clear)

    def _slam_get_state(self, request: lifecycle_msgs.srv.GetState.Request, response: lifecycle_msgs.srv.GetState.Response) -> lifecycle_msgs.srv.GetState.Response:
        response.current_state.id = self.slam_state
        return response

    @staticmethod
    def _costmap_get_state(request: lifecycle_msgs.srv.GetState.Request, response: lifecycle_msgs.srv.GetState.Response) -> lifecycle_msgs.srv.GetState.Response:
        response.current_state.id = ACTIVE
        return response

    def _reset(self, request: Reset.Request, response: Reset.Response) -> Reset.Response:
        self.resets.append(request.pause_new_measurements)
        response.result = Reset.Response.RESULT_SUCCESS
        return response

    def _clear(self, request: ClearEntireCostmap.Request, response: ClearEntireCostmap.Response) -> ClearEntireCostmap.Response:
        self.clears += 1
        return response


def _adapter(mapping: str) -> Nav2Adapter:
    adapter = Nav2Adapter.__new__(Nav2Adapter)
    adapter._bringup_kwargs = {"mapping": mapping}
    adapter._service_clients = {}
    return adapter


def _reset_map(mapping: str, slam_state: int = ACTIVE, costmap_up: bool = True) -> tuple[bool, _Stack]:
    import rclpy.executors
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.shared import Namespace

    async def main() -> tuple[bool, _Stack]:
        suffix = uuid.uuid4().hex[:8]
        namespace = Namespace(f"/slam_reset_{suffix}/jackal")
        node = ArenaMixinNode(f"slam_reset_{suffix}")
        stack = _Stack(node, namespace, slam_state, costmap_up)
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(node)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            done = await _adapter(mapping).reset_map(SimpleNamespace(node=node, namespace=namespace))
        finally:
            executor.shutdown()
            thread.join(timeout=2.0)
            node.destroy_node()
        return done, stack

    return asyncio.run(main())


def test_a_slam_robot_starts_each_episode_with_an_empty_map_and_global_costmap() -> None:
    done, stack = _reset_map("slam")
    assert done
    assert stack.resets == [False]
    assert stack.clears == 1


def test_a_static_map_robot_keeps_its_map() -> None:
    done, stack = _reset_map("static")
    assert not done
    assert stack.resets == []
    assert stack.clears == 0


def test_a_slam_toolbox_that_is_not_active_yet_is_left_alone() -> None:
    done, stack = _reset_map("slam", slam_state=INACTIVE)
    assert not done
    assert stack.resets == []
    assert stack.clears == 0


def test_a_global_costmap_still_coming_up_skips_the_clear_without_failing_the_reset() -> None:
    done, stack = _reset_map("slam", costmap_up=False)
    assert not done
    assert stack.resets == [False]
    assert stack.clears == 0
