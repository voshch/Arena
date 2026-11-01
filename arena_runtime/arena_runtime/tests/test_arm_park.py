from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from arena_rclpy_mixins.Time import Time
from arena_robots.caps import ArmSpec

from arena_runtime.env.adapters.arm import park_arms, park_positions


def _arm(raw: dict) -> ArmSpec:
    return ArmSpec(path=Path("caps/arm.yaml"), raw=raw, name="top")


class TestParkPositions:
    def test_stow_pose_ordered_by_chain(self):
        arm = _arm(
            {
                "chain": ["a", "b", "c"],
                "controller": "top_controller",
                "named_poses": {"stow": {"joints": {"b": -1.57, "a": 0.5}}},
            }
        )
        assert park_positions(arm) == [0.5, -1.57, 0.0]

    def test_zeros_without_stow(self):
        arm = _arm({"chain": ["a", "b"], "controller": "top_controller"})
        assert park_positions(arm) == [0.0, 0.0]


class _Publisher:
    def __init__(self) -> None:
        self.sent: list = []

    def get_subscription_count(self) -> int:
        return 1

    def publish(self, msg) -> None:
        self.sent.append(msg)


class _Node:
    def __init__(self, now: Time) -> None:
        self.sim_time = now
        self.waited: list[Time] = []
        self.publishers: dict[str, _Publisher] = {}

    def create_publisher(self, msg_type, topic: str, depth: int) -> _Publisher:
        self.publishers[topic] = _Publisher()
        return self.publishers[topic]

    async def await_sim_time(self, target: Time, freeze_timeout: float | None = None) -> bool:
        self.waited.append(target)
        return True


class _Robot:
    name = "jackal"

    def __init__(self, arms: dict[str, ArmSpec], node: _Node) -> None:
        self.node = node
        self.robot = SimpleNamespace(resolved_request={}, frames=None)
        self.robot_view = SimpleNamespace(effective_caps=lambda request, frames: SimpleNamespace(arm=arms))

    def namespace(self, *parts: str) -> str:
        return "/env/jackal/" + "/".join(parts)


class TestParkArms:
    def test_waits_in_sim_time_until_the_parked_arms_settled(self):
        node = _Node(Time.from_float(10.0))
        arm = _arm({"chain": ["a", "b"], "controller": "top_controller", "named_poses": {"stow": {"joints": {"a": -1.57}}}})
        asyncio.run(park_arms(_Robot({"top": arm}, node)))  # pyright: ignore[reportArgumentType]
        [traj] = node.publishers["/env/jackal/top_controller/joint_trajectory"].sent
        assert list(traj.points[0].positions) == [-1.57, 0.0]
        assert [t.to_seconds() for t in node.waited] == [14.0]

    def test_a_robot_without_arms_does_not_wait(self):
        node = _Node(Time.from_float(10.0))
        assert asyncio.run(park_arms(_Robot({}, node))) == []  # pyright: ignore[reportArgumentType]
        assert node.waited == []
