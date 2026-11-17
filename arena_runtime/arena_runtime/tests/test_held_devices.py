"""Held assistive devices: the agent type's `device:` block parsed and posed from the rendered gait phase."""

from __future__ import annotations

import asyncio
import math
import uuid
from pathlib import Path

import pytest
from arena_people_msgs.msg import HeldDevice
from arena_runtime.humans.devices import DeviceSpec, Track, footprint, shift
from arena_runtime.humans.profile import agent_type_device_section

CANE = {"kind": "cane", "model": "Common/Object/WhiteCane", "grip": {"x": 0.3, "y": -0.1, "z": 0.9}, "tip": {"x": {"mean": 1.0, "harmonics": [[0.0, 0.0]]}, "y": {"mean": 0.0, "harmonics": [[0.3, 0.0]]}, "z": 0.0}, "radius": 0.0125}
FRAME = {"kind": "walking_frame", "origin": {"x": 0.35, "y": 0.0, "z": {"mean": 0.02, "harmonics": [[0.02, 0.0]]}}, "size": {"length": 0.4, "width": 0.5, "height": 0.67, "ahead": 0.1, "left": 0.0}}


def test_a_track_is_a_constant_or_harmonics_of_the_phase() -> None:
    assert Track.parse(0.36, "x").at(1.0) == 0.36
    track = Track.parse({"mean": 0.1, "harmonics": [[0.2, 0.5], [0.05, -1.0]]}, "x")
    assert track.at(0.7) == pytest.approx(0.1 + 0.2 * math.sin(0.7 + 0.5) + 0.05 * math.sin(1.4 - 1.0))


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "umbrella"},
        {**CANE, "radius": -0.01},
        {k: v for k, v in CANE.items() if k != "tip"},
        {**CANE, "origin": FRAME["origin"]},
        {**FRAME, "size": {"length": 0.4, "width": 0.0, "height": 0.6}},
        {**FRAME, "size": {"length": 0.4, "width": 0.5}},
        {**FRAME, "origin": {"x": 0.3, "y": 0.0}},
        {**FRAME, "origin": {"x": True, "y": 0.0, "z": 0.0}},
    ],
)
def test_a_malformed_device_block_is_rejected(raw: dict) -> None:
    with pytest.raises(ValueError):
        DeviceSpec.parse(raw)


def test_a_cane_runs_from_the_fist_to_the_tip_in_the_world() -> None:
    dev = DeviceSpec.parse(CANE).pose(2.0, 3.0, 0.5, math.pi / 2, w=math.pi / 2)  # heading +y, tip swept 0.3 m to the left
    assert (dev.kind, dev.model, dev.radius) == ("cane", "Common/Object/WhiteCane", 0.0125)
    assert (dev.start.x, dev.start.y, dev.start.z) == pytest.approx((2.1, 3.3, 1.4))
    assert (dev.end.x, dev.end.y, dev.end.z) == pytest.approx((1.7, 4.0, 0.5))
    assert (dev.centre.x, dev.centre.y) == pytest.approx((dev.end.x, dev.end.y))
    assert dev.yaw == pytest.approx(math.atan2(0.7, -0.4))


def test_a_box_sits_ahead_of_the_holder_on_the_floor_and_lifts() -> None:
    dev = DeviceSpec.parse(FRAME).pose(1.0, 1.0, 0.0, math.pi, w=math.pi / 2)  # heading -x
    assert (dev.start.x, dev.start.y, dev.start.z) == pytest.approx((0.65, 1.0, 0.04))
    assert (dev.end.x, dev.end.y, dev.end.z) == pytest.approx((0.65, 1.0, 0.04))
    assert (dev.centre.x, dev.centre.y, dev.centre.z) == pytest.approx((0.55, 1.0, 0.0))
    assert dev.yaw == pytest.approx(math.pi)
    assert (dev.size.x, dev.size.y, dev.size.z) == pytest.approx((0.4, 0.5, 0.67))


def test_the_footprint_of_a_cane_and_of_a_box() -> None:
    import shapely

    cane = footprint(DeviceSpec.parse(CANE).pose(0.0, 0.0, 0.0, 0.0, w=0.0))
    assert cane.intersects(shapely.Point(1.0, 0.0).buffer(0.01))
    assert not cane.intersects(shapely.Point(1.0, 0.05).buffer(0.01))
    box = footprint(DeviceSpec.parse(FRAME).pose(0.0, 0.0, 0.0, math.pi / 2, w=0.0))
    assert box.area == pytest.approx(0.4 * 0.5)
    assert (box.centroid.x, box.centroid.y) == pytest.approx((0.0, 0.45))
    assert footprint(HeldDevice()) is None
    assert footprint(DeviceSpec.parse(CANE).pose(math.nan, 0.0, 0.0, 0.0, w=0.0)) is None
    assert footprint(DeviceSpec.parse(FRAME).pose(0.0, math.inf, 0.0, 0.0, w=0.0)) is None


def test_a_device_moves_with_its_pedestrian_into_another_frame() -> None:
    dev = DeviceSpec.parse(CANE).pose(0.0, 0.0, 0.0, 0.0, w=0.0)
    shift(dev, 5.0, -1.0, 3.0)
    assert (dev.start.x, dev.start.y, dev.start.z) == pytest.approx((5.3, -1.1, 3.9))
    assert (dev.end.x, dev.centre.x) == pytest.approx((6.0, 6.0))
    none = HeldDevice()
    shift(none, 5.0, 0.0, 0.0)
    assert none.start.x == 0.0


def test_the_device_block_comes_whole_from_the_nearest_type_along_extends(tmp_path: Path) -> None:
    import yaml

    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "adult.yaml").write_text("name: adult\n")
    (builtin / "blind.yaml").write_text(yaml.safe_dump({"name": "blind", "extends": "adult", "device": CANE}))
    (builtin / "loop_a.yaml").write_text("name: loop_a\nextends: loop_b\n")
    (builtin / "loop_b.yaml").write_text("name: loop_b\nextends: loop_a\n")
    unaware = tmp_path / "unaware.yaml"
    unaware.write_text("name: unaware\nextends: blind\nperception: {robot_awareness: 0.0}\n")
    framed = tmp_path / "framed.yaml"
    framed.write_text(yaml.safe_dump({"name": "framed", "extends": "blind", "device": FRAME}))

    assert agent_type_device_section(str(unaware), builtin) == CANE
    assert agent_type_device_section(str(framed), builtin) == FRAME
    assert agent_type_device_section("adult", builtin) is None
    assert agent_type_device_section("ghost", builtin) is None
    with pytest.raises(ValueError, match="cycle"):
        agent_type_device_section("loop_a", builtin)


def test_the_published_roster_carries_each_holders_device(rclpy_context, tmp_path: Path) -> None:
    pytest.importorskip("arena_runtime.sim.dummy_simulator")
    import yaml
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.acoustics.noop import NoopAcousticsSimulator
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.sim.dummy_simulator import DummySimulator

    blind = tmp_path / "blind.yaml"
    blind.write_text(yaml.safe_dump({"name": "blind", "device": CANE}))

    class _Node(ServiceNamespace, AsyncNode):
        pass

    published: list[Pedestrians] = []

    async def main() -> None:
        suffix = f"t_{uuid.uuid4().hex[:8]}"
        ns = Namespace(f"/test/{suffix}")
        node = _Node(f"held_devices_{suffix}", namespace=str(ns))
        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=""))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            human = NoopHumanSimulator(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=NoopAcousticsSimulator(node=node, namespace=ns))
            human._arena_peds_publisher.publish = published.append  # type: ignore[method-assign]
            msg = Pedestrians()
            for name, agent_type, x in (("walker", "", 0.0), ("blind", str(blind), 4.0)):
                ped = Pedestrian(name=name, id=len(msg.pedestrians) + 1, agent_type=agent_type, animation_state=Pedestrian.WALKING)
                ped.pose.position.x, ped.pose.orientation.w = x, 1.0
                ped.twist.linear.x = 1.0
                msg.pedestrians.append(ped)
            human.publish_arena_peds(msg)
            human.publish_arena_peds(msg)  # a broken block is not retried every tick
        finally:
            node.destroy_node()

    asyncio.run(main())
    walker, holder = published[-1].pedestrians
    assert walker.device.kind == ""
    assert holder.device.kind == "cane"
    assert (holder.device.start.x, holder.device.start.y, holder.device.start.z) == pytest.approx((4.3, -0.1, 0.9))
    swept = 0.3 * math.sin(holder.gait_phase)  # the default profile does not warp the phase
    assert (holder.device.end.x, holder.device.end.y) == pytest.approx((5.0, swept))
