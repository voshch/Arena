"""Held assistive devices: the agent type's `device:` block posed on the rendered skeleton and projected on the ground."""

from __future__ import annotations

import asyncio
import math
import uuid
from pathlib import Path

import numpy as np
import pytest
from arena_people_msgs.msg import Pedestrian
from arena_runtime.humans.devices import DeviceSpec, box_footprint, bundle_of, place, shapes, shift
from arena_runtime.humans.profile import agent_type_device_section

CANE = {"kind": "cane", "bone": "cane"}
FRAME = {"kind": "walking_frame", "model": "Common/Object/WalkerFitted"}
HUMANSIM_TYPES = Path(__file__).resolve().parents[3] / "humansim" / "arena_humansim" / "config" / "agent_types"
CANE_BUNDLE = "Common/Human/casual_female_caucasian_middleage_whitecane"


@pytest.mark.parametrize("raw", [{"kind": "umbrella", "bone": "x"}, {"kind": "cane"}, {**CANE, "model": "Common/Object/WhiteCane"}, {**CANE, "radius": 0.01}, "cane"])
def test_a_malformed_device_block_is_rejected(raw: object) -> None:
    with pytest.raises(ValueError):
        DeviceSpec.parse(raw)


def test_a_device_names_a_prop_bone_or_an_object_held_in_both_hands() -> None:
    assert DeviceSpec.parse(CANE) == DeviceSpec("cane", bone="cane")
    assert DeviceSpec.parse(FRAME) == DeviceSpec("walking_frame", model="Common/Object/WalkerFitted")


def test_the_bundle_of_an_actor_sdf() -> None:
    assert bundle_of("/a/Common/Human/x/rig/x.sdf") == Path("/a/Common/Human/x")
    assert bundle_of("/a/Common/Human/x/x.sdf") == Path("/a/Common/Human/x")


@pytest.mark.parametrize(
    ("face", "expected"),
    [("-y", (0.6, 0.4, -0.05, 0.1)), ("+x", (0.4, 0.6, 0.1, 0.05)), ("+y", (0.6, 0.4, 0.05, -0.1)), ("-x", (0.4, 0.6, -0.1, -0.05))],
)
def test_a_box_turns_the_models_front_onto_the_heading(face: str, expected: tuple[float, ...]) -> None:
    assert box_footprint([[-0.1, 0.3], [-0.25, 0.35], [0.0, 0.9]], face) == pytest.approx(expected)


def _ped(x: float, y: float, radius: float = 0.0) -> Pedestrian:
    ped = Pedestrian(name="p", radius=radius)
    ped.pose.position.x, ped.pose.position.y, ped.pose.orientation.w = x, y, 1.0
    return ped


def test_an_outline_is_placed_at_the_pedestrian_and_moves_with_it() -> None:
    ped = _ped(2.0, 3.0)
    place(ped, "cane", np.array([(0.3, -0.1), (1.0, 0.0), (0.3, 0.0)]), math.pi / 2)
    assert ped.device.kind == "cane"
    assert [(p.x, p.y) for p in ped.device.footprint] == pytest.approx([(2.1, 3.3), (2.0, 4.0), (2.0, 3.3)])
    shift(ped.device, 1.0, -1.0)
    assert (ped.device.footprint[1].x, ped.device.footprint[1].y) == pytest.approx((3.0, 3.0))


def test_a_pedestrian_is_its_own_body_disc_and_its_device() -> None:
    ped = _ped(0.0, 0.0, radius=0.45)
    place(ped, "rollator", np.array([(0.2, -0.3), (0.9, -0.3), (0.9, 0.3), (0.2, 0.3)]), 0.0)
    body, device = shapes(ped, 0.3)
    assert body.area == pytest.approx(math.pi * 0.45**2, rel=0.01)
    assert device.area == pytest.approx(0.7 * 0.6, rel=1e-5)
    assert shapes(_ped(0.0, 0.0), 0.3)[0].area == pytest.approx(math.pi * 0.3**2, rel=0.01)
    assert shapes(_ped(0.0, 0.0), 0.3)[1] is None


def test_a_lost_pedestrian_has_no_shapes() -> None:
    ped = _ped(math.nan, 0.0)
    place(ped, "cane", np.array([(0.3, 0.0), (1.0, 0.1), (1.0, -0.1)]), 0.0)
    assert shapes(ped, 0.3) == (None, None)


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


@pytest.mark.skipif(not HUMANSIM_TYPES.is_dir(), reason="the humansim submodule is not checked out")
@pytest.mark.parametrize(("name", "kind"), [("blind_white_cane", "cane"), ("blind_rolling_cane", "cane"), ("elder_rollator", "rollator"), ("patient_walking_frame", "walking_frame"), ("patient_iv_pole", "iv_stand")])
def test_the_shipped_device_blocks_parse(name: str, kind: str) -> None:
    section = agent_type_device_section(name, HUMANSIM_TYPES)
    assert section is not None
    assert DeviceSpec.parse(section).kind == kind


SCENARIO_TYPES = sorted((Path(__file__).resolve().parents[3] / "arena_simulation_setup" / "worlds" / "map_empty" / "scenarios").glob("device_contact_*/*_unaware.yaml"))


@pytest.mark.skipif(not HUMANSIM_TYPES.is_dir(), reason="the humansim submodule is not checked out")
@pytest.mark.parametrize("path", SCENARIO_TYPES, ids=lambda p: p.stem)
def test_the_device_contact_people_load_in_humansim_with_their_device(path: Path) -> None:
    from arena_runtime.humans.arena_humansim.agent import agent_type_def

    assert agent_type_def(str(path)) is not None
    assert agent_type_device_section(str(path), HUMANSIM_TYPES) is not None


def test_a_cane_follows_the_arm_of_the_drawn_character() -> None:
    from arena_runtime.humans.devices import HeldBody
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    try:
        bundle = Path(asyncio.run(HumanIdentifier.parse(CANE_BUNDLE).resolve_path()))
    except FileNotFoundError:
        pytest.skip(f"{CANE_BUNDLE} is not available here")
    body = HeldBody(bundle, DeviceSpec.parse(CANE))
    still = body.footprint({}, {})
    raised = body.footprint({"r_p_shoulder": -0.6}, {})
    assert len(still) >= 3
    assert not np.allclose(still.mean(axis=0), raised.mean(axis=0), atol=0.05)  # the cane rides the arm
    assert len(body.footprint({"r_p_shoulder": math.nan}, {})) == 0
    with pytest.raises(ValueError, match="no prop bone"):
        HeldBody(bundle, DeviceSpec("cane", bone="umbrella"))


def test_the_published_roster_carries_each_holders_device(rclpy_context, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("arena_runtime.sim.dummy_simulator")
    import rclpy
    import tf2_ros
    import yaml
    from arena_people_msgs.msg import Pedestrians
    from arena_rclpy_mixins.Async import AsyncNode
    from arena_rclpy_mixins.ServiceNamespace import ServiceNamespace
    from arena_rclpy_mixins.shared import Namespace
    from arena_runtime.acoustics.noop import NoopAcousticsSimulator
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.humans import TrackedRobot, devices
    from arena_runtime.humans.noop import NoopHumanSimulator
    from arena_runtime.shared import Pose, Position
    from arena_runtime.sim.dummy_simulator import DummySimulator

    blind = tmp_path / "blind.yaml"
    blind.write_text(yaml.safe_dump({"name": "blind", "device": CANE}))
    gripless = tmp_path / "gripless.yaml"
    gripless.write_text(yaml.safe_dump({"name": "gripless", "device": FRAME}))

    class _Body:
        def __init__(self, bundle: Path, spec: DeviceSpec) -> None:
            self.kind = spec.kind

        def footprint(self, angles: dict, hold: dict) -> np.ndarray:
            return np.array([(0.3, -0.1), (1.0, 0.0), (0.3, 0.0)]) if angles else np.empty((0, 2))

    monkeypatch.setattr(devices, "HeldBody", _Body)

    def roster(*entries: tuple[str, str, float]) -> Pedestrians:
        msg = Pedestrians()
        for name, agent_type, x in entries:
            ped = Pedestrian(name=name, id=len(msg.pedestrians) + 1, agent_type=agent_type, animation_state=Pedestrian.WALKING)
            ped.pose.position.x, ped.pose.orientation.w = x, 1.0
            ped.twist.linear.x = 1.0
            msg.pedestrians.append(ped)
        return msg

    everyone = (("walker", "", 0.0), ("blind", str(blind), 4.0), ("far", str(blind), 9.0), ("gripless", str(gripless), 4.5))
    published: list[Pedestrians] = []

    class _Node(ServiceNamespace, AsyncNode):
        pass

    async def main() -> None:
        suffix = f"t_{uuid.uuid4().hex[:8]}"
        ns = Namespace(f"/test/{suffix}")
        node = _Node(f"held_devices_{suffix}", namespace=str(ns))
        node.tf_buffer = tf2_ros.Buffer()

        def publish(msg: Pedestrians) -> Pedestrians | None:
            seen = len(published)
            human.publish_arena_peds(msg)
            for _ in range(50):
                rclpy.spin_once(node, timeout_sec=0.02)
                if len(published) > seen:
                    return published[-1]
            return None

        try:
            realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix=""))
            simulator = DummySimulator(node=node, namespace=ns, realizer=realizer)
            human = NoopHumanSimulator(node=node, namespace=ns, simulator=simulator, realizer=realizer, acoustics=NoopAcousticsSimulator(node=node, namespace=ns))
            node.create_subscription(Pedestrians, str(ns("arena_peds")), published.append, 10)
            assert any(publish(roster()) is not None for _ in range(20))
            for name in ("blind", "far", "gripless"):
                human._ped_model_uris[name] = "/bundles/caner/rig/caner.sdf"
            human._robots["robot"] = TrackedRobot(robot=None, tf_frame=None, pose=Pose(Position(x=6.0, y=0.0)), radius=0.3)  # type: ignore[arg-type]
            loading = publish(roster(*everyone))
            assert loading is not None
            assert loading.pedestrians[1].device.kind == ""
            for _ in range(100):
                latest = publish(roster(*everyone))
                if latest is not None and latest.pedestrians[1].device.kind:
                    break
        finally:
            node.destroy_node()

    asyncio.run(main())
    walker, holder, far, gripless_ped = published[-1].pedestrians
    assert gripless_ped.device.kind == ""
    assert walker.device.kind == ""
    assert far.device.kind == ""
    assert holder.device.kind == "cane"
    np.testing.assert_allclose([(p.x, p.y) for p in holder.device.footprint], [(4.3, -0.1), (5.0, 0.0), (4.3, 0.0)], atol=1e-6)
