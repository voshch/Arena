"""MuJoCo requests carry abstract poses into the stacked simulator frame and keep each robot's map -> odom abstract."""

from __future__ import annotations

import asyncio
import typing
import uuid

import pytest

_JACKAL = "env_0_Robots_jackal_base_link"
_UPSTAIRS = (1.0 + 17.0 + 3.0, 2.0 + 4.0)
_DOWNSTAIRS = (1.0 + 3.0, 2.0 + 4.0)


def _realizer(stacked: bool):
    from arena_runtime.env.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(x=1.0, y=2.0, prefix="world"))
    realizer.reset_level_origins({"0": (0.0, 0.0), "1": (17.0, 0.0)})
    if stacked:
        realizer.reset_placement({"0": (0.0, 0.0, 0.0), "1": (9.2, 0.0, 3.0)}, {"0": (0.0, 0.0, 14.0, 10.0), "1": (0.0, 0.0, 14.0, 10.0)})
    return realizer


@pytest.fixture
def on_mujoco_sim():
    module = pytest.importorskip("arena_runtime.sim.mujoco_simulator")

    def run(body: typing.Callable[..., typing.Awaitable[None]], *, stacked: bool = True) -> None:
        from arena_rclpy_mixins import ArenaMixinNode
        from arena_rclpy_mixins.shared import Namespace

        async def main() -> None:
            suffix = uuid.uuid4().hex[:8]
            node = ArenaMixinNode(f"mujoco_placement_{suffix}", namespace=f"/mujoco_placement_{suffix}")
            sim = module.MujocoSimulator(node=node, namespace=Namespace(f"/mujoco_placement_{suffix}"), realizer=_realizer(stacked))
            try:
                await body(sim)
            finally:
                await sim.stop_mechanisms()
                node.destroy_node()

        asyncio.run(main())

    return run


def _peds(*positions: tuple[float, float, float]):
    from arena_people_msgs.msg import Pedestrian, Pedestrians
    from geometry_msgs.msg import Point, Pose

    return Pedestrians(pedestrians=[Pedestrian(name=f"ped_{index}", id=index, pose=Pose(position=Point(x=x, y=y, z=z))) for index, (x, y, z) in enumerate(positions)])


def _xyz(point) -> tuple[float, float, float]:
    return point.x, point.y, point.z


def test_pedestrians_upstairs_move_by_their_level_shift_and_the_stream_stays_abstract(on_mujoco_sim):
    async def body(sim) -> None:
        stream = _peds((*_UPSTAIRS, 0.2), (*_DOWNSTAIRS, 0.0))
        placed = sim._placed_peds(stream)
        assert [_xyz(ped.pose.position) for ped in placed.pedestrians] == pytest.approx([(_UPSTAIRS[0] - 7.8, _UPSTAIRS[1], 3.2), (*_DOWNSTAIRS, 0.0)])
        assert [(ped.name, ped.id) for ped in placed.pedestrians] == [("ped_0", 0), ("ped_1", 1)]
        assert _xyz(stream.pedestrians[0].pose.position) == (*_UPSTAIRS, 0.2)

    on_mujoco_sim(body)


def test_pedestrians_of_a_spread_world_go_out_as_they_came(on_mujoco_sim):
    async def body(sim) -> None:
        stream = _peds((*_UPSTAIRS, 0.0))
        assert sim._placed_peds(stream) is stream

    on_mujoco_sim(body, stacked=False)


def test_light_fixtures_and_points_take_the_shift_of_the_level_holding_them(on_mujoco_sim):
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.utils.geometry import Pose, Position
    from geometry_msgs.msg import Point

    async def body(sim) -> None:
        lamp = Light(name="lamp", fixture="bulb", pose=Pose(position=Position(*_UPSTAIRS, 2.5)), lumens=800.0)
        assert _xyz(sim._light_msg(lamp).positions[0]) == pytest.approx((_UPSTAIRS[0] - 7.8, _UPSTAIRS[1], 5.5))
        assert _xyz(sim._placed_point(Point(x=_DOWNSTAIRS[0], y=_DOWNSTAIRS[1], z=2.5))) == (*_DOWNSTAIRS, 2.5)

    on_mujoco_sim(body)


def test_a_robot_upstairs_asks_once_for_the_inverse_shift_and_one_downstairs_never(on_mujoco_sim):
    from arena_runtime.shared import Pose, Position

    async def body(sim) -> None:
        sim._track_level("jackal", Pose(position=Position(*_UPSTAIRS, 0.0)))
        assert sim._map_offset_request("jackal") is None
        sim._robot_prims["jackal"] = _JACKAL
        request = sim._map_offset_request("jackal")
        assert (request.env_id, request.name) == (0, _JACKAL)
        assert _xyz(request.translation) == pytest.approx((7.8, 0.0, -3.0))
        assert sim._map_offset_request("jackal") is None

        sim._robot_prims["ranger"] = "env_0_Robots_ranger_base_link"
        sim._track_level("ranger", Pose(position=Position(*_DOWNSTAIRS, 0.0)))
        assert sim._map_offset_request("ranger") is None

    on_mujoco_sim(body)
