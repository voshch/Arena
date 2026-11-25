from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from arena_rclpy_mixins.Time import Time
from arena_robots.caps import ArmSpec
from arena_runtime.env.adapters.arm import park_arms, park_positions, park_seconds


def _arm(raw: dict) -> ArmSpec:
    return ArmSpec(path=Path("caps.yaml"), raw=raw, name="top")


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


def _limited_arm(joints: dict[str, dict[str, float]], chain: list[str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(name="top", chain=chain or list(joints), controller="top_controller", named_poses={"stow": {}}, joint_limits=joints, raw={})


def _tiago_park_seconds(caps: dict | None = None) -> float:
    from arena_robots import urdf_facts
    from arena_robots.Robot import RobotIdentifier

    effective = RobotIdentifier("tiago").resolve_sync().effective_caps({}, caps=caps or {})
    return park_seconds(effective.arm["arm"], urdf_facts.joint_limits(effective.rendered_urdf()))


class TestParkSeconds:
    def test_unoverridden_limits_park_in_the_default_time(self):
        pytest.importorskip("xacro")
        assert _tiago_park_seconds() == 3

    def test_halving_every_joint_velocity_doubles_the_park_time(self):
        pytest.importorskip("xacro")
        assert _tiago_park_seconds({"arm.arm.joint_limits.*.max_velocity": {"scale": 0.5}}) == pytest.approx(6.0)

    def test_the_most_slowed_joint_sets_the_factor(self):
        pytest.importorskip("xacro")
        assert _tiago_park_seconds({"arm.arm.joint_limits.arm_1_joint.max_velocity": {"scale": 0.25}, "arm.arm.joint_limits.arm_2_joint.max_velocity": {"scale": 0.5}}) == pytest.approx(12.0)

    def test_arms_without_a_joint_limits_file_park_in_the_default_time(self):
        assert park_seconds(_limited_arm({"a": {"max_velocity": 0.1}}), {}) == 3  # pyright: ignore[reportArgumentType]


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

    def __init__(self, arms: dict[str, ArmSpec], node: _Node, caps: object = None) -> None:
        self.node = node
        self.caps = caps if caps is not None else SimpleNamespace(arm=arms, rendered_urdf=lambda: '<robot name="jackal"/>')

    def namespace(self, *parts: str) -> str:
        return "/env/jackal/" + "/".join(parts)


class TestParkArms:
    def test_waits_in_sim_time_until_the_parked_arms_settled(self):
        node = _Node(Time.from_float(10.0))
        arm = _limited_arm({})
        arm.chain = ["a", "b"]
        arm.named_poses = {"stow": {"a": -1.57}}
        asyncio.run(park_arms(_Robot({"top": arm}, node)))  # pyright: ignore[reportArgumentType]
        [traj] = node.publishers["/env/jackal/top_controller/joint_trajectory"].sent
        assert list(traj.points[0].positions) == [-1.57, 0.0]
        assert [t.to_seconds() for t in node.waited] == [14.0]

    def test_an_override_slowing_the_arm_stretches_the_trajectory_time_and_the_settle_wait(self):
        pytest.importorskip("xacro")
        from arena_robots.Robot import RobotIdentifier

        node = _Node(Time.from_float(10.0))
        caps = RobotIdentifier("tiago").resolve_sync().effective_caps({}, caps={"arm.arm.joint_limits.*.max_velocity": {"scale": 0.4}})
        arm = caps.arm["arm"]
        asyncio.run(park_arms(_Robot({}, node, caps)))  # pyright: ignore[reportArgumentType]
        [traj] = node.publishers[f"/env/jackal/{arm.controller}/joint_trajectory"].sent
        assert traj.points[0].time_from_start.sec == 7
        assert traj.points[0].time_from_start.nanosec == pytest.approx(5e8, abs=1)
        assert [t.to_seconds() for t in node.waited] == [pytest.approx(10.0 + 7.5 + 1.0)]

    def test_a_robot_without_arms_does_not_wait(self):
        node = _Node(Time.from_float(10.0))
        assert asyncio.run(park_arms(_Robot({}, node))) == []  # pyright: ignore[reportArgumentType]
        assert node.waited == []
