from __future__ import annotations

import abc
import traceback
import typing

from arena_bringup.constants import SimSimulator
from arena_rclpy_mixins import ArenaMixinNode
from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.tree import IdentifierProtocol

from arena_runtime._node import NodeInterface
from arena_runtime.env.realizer import Realizer
from arena_runtime.shared import Pose, Position

from ._interface import MechanismITF, ObstacleITF, PedestrianITF, RobotITF, SimLifecycle, ViewportITF, WorldITF
from ._interface import SimUnavailable as SimUnavailable


class SimKwargs(typing.TypedDict):
    node: ArenaMixinNode
    namespace: Namespace
    realizer: Realizer
    env_id: typing.NotRequired[int]


class BaseSim(NodeInterface[ArenaMixinNode], ObstacleITF, PedestrianITF, RobotITF, WorldITF, MechanismITF, ViewportITF, abc.ABC):
    _namespace: Namespace
    _realizer: Realizer
    _env_id: int
    _robot_levels: dict[str, str]

    def __init__(self, **kwargs: typing.Unpack[SimKwargs]) -> None:
        super().__init__(node=kwargs['node'])
        self._namespace = kwargs['namespace']
        self._realizer = kwargs['realizer']
        self._env_id = kwargs.get('env_id', 0)
        self._robot_levels = {}

    async def before_reset_episode(self) -> bool:
        """Episode-boundary hook before the task resets, for sim work that must not ride
        the mid-episode pause path (physics state reset, contact caches). Default no-op."""
        return True

    async def after_reset_episode(self) -> bool:
        """Episode-boundary hook after the task resets. Default no-op."""
        return True

    async def step(self, n: int = 1) -> bool:
        """Advance the simulation by ``n`` ticks. Default no-op."""
        del n
        return True

    async def shutdown(self) -> None:
        """Per-env teardown, e.g. delete entities this env spawned. Default no-op."""

    # Placement of the abstract frame into the simulator frame, the identity unless the levels are stacked
    def _shift(self, x: float, y: float) -> tuple[float, float, float]:
        """Simulator minus abstract coordinates at an abstract point, by the level holding it."""
        placement = self._realizer.placement()
        return placement.shift(placement.level_at(x, y))

    def _placed(self, pose: Pose) -> Pose:
        """An abstract pose in the simulator frame."""
        dx, dy, dz = self._shift(pose.position.x, pose.position.y)
        return Pose(position=Position(pose.position.x + dx, pose.position.y + dy, pose.position.z + dz), orientation=pose.orientation)

    def _offset(self, x: float, y: float) -> Pose:
        """Spawn pose that moves a model authored in abstract coordinates around (x, y) into the simulator frame."""
        dx, dy, dz = self._shift(x, y)
        return Pose(position=Position(dx, dy, dz))

    def _robot_shift(self, sim_path: str) -> tuple[float, float, float]:
        """Simulator minus abstract coordinates on the level the robot stands on."""
        return self._realizer.placement().shift(self._robot_levels.get(sim_path, ''))

    def _track_level(self, sim_path: str, pose: Pose) -> None:
        """Record the level of the abstract pose a robot was spawned or teleported to and republish its map -> odom."""
        self._robot_levels[sim_path] = self._realizer.placement().level_at(pose.position.x, pose.position.y)
        self._publish_odom(sim_path)

    async def robot_changed_level(self, sim_path: str, level: str) -> None:
        if self._robot_levels.get(sim_path) == level:
            return
        self._robot_levels[sim_path] = level
        self._publish_odom(sim_path)

    def _publish_odom(self, sim_path: str) -> None:
        """Make the robot's map -> odom the inverse of `_robot_shift`. No-op on a simulator that does not stack."""

    # Utils
    async def safe_resolve(self, identifier: IdentifierProtocol) -> object:
        try:
            return await identifier.resolve()
        except Exception as e:
            self._logger.warning(f"Failed to resolve {identifier}")
            self._logger.debug(traceback.format_exc())
            return None


SimulatorRegistry = Registry[SimSimulator, BaseSim]()
LifecycleRegistry = Registry[SimSimulator, SimLifecycle]()


@SimulatorRegistry.register(SimSimulator.DUMMY)
async def lazy_dummy(**kwargs: typing.Unpack[SimKwargs]) -> BaseSim:
    from .dummy_simulator import DummySimulator

    return DummySimulator(**kwargs)


@LifecycleRegistry.register(SimSimulator.DUMMY)
async def lazy_dummy_lifecycle(physics_dt: float = 0.0333, **kwargs: object) -> SimLifecycle:
    del kwargs
    from .dummy_simulator import DummyHost

    return DummyHost(physics_dt)


def _flatsim_missing(e: ModuleNotFoundError) -> typing.NoReturn:
    """Re-raise a failed adapter import, naming the feature when its packages are absent."""
    if e.name is not None and e.name.split('.')[0] in ('flatsim', 'flatsim_msgs'):
        raise SimUnavailable('sim:=flatsim needs the flatsim package, install it with `arena feature flatsim install`') from e
    raise e


@SimulatorRegistry.register(SimSimulator.FLATSIM)
async def lazy_flatsim(**kwargs: typing.Unpack[SimKwargs]) -> BaseSim:
    try:
        from .flatsim_simulator import FlatsimSimulator
    except ModuleNotFoundError as e:
        _flatsim_missing(e)

    return await FlatsimSimulator.create(**kwargs)


@LifecycleRegistry.register(SimSimulator.FLATSIM)
async def lazy_flatsim_lifecycle(node: object, physics_dt: float = 0.0333, **kwargs: object) -> SimLifecycle:
    from arena_rclpy_mixins import ArenaMixinNode

    try:
        from .flatsim_simulator.flatsim_simulator import FlatsimHost
    except ModuleNotFoundError as e:
        _flatsim_missing(e)

    del kwargs
    assert isinstance(node, ArenaMixinNode)
    return FlatsimHost(physics_dt, node)


# @SimulatorRegistry.register(SimSimulator.FLATLAND)
# async def lazy_flatland(**kwargs):
#     from .flatland_simulator import FlatlandSimulator
#     return FlatlandSimulator(**kwargs)


@SimulatorRegistry.register(SimSimulator.GAZEBO)
async def lazy_gazebo(**kwargs: typing.Unpack[SimKwargs]) -> BaseSim:
    from .gazebo_simulator import GazeboSimulator

    return await GazeboSimulator.create(**kwargs)


@LifecycleRegistry.register(SimSimulator.GAZEBO)
async def lazy_gazebo_lifecycle(node: object, physics_dt: float = 0.0333, **kwargs: object) -> SimLifecycle:
    import asyncio

    from arena_rclpy_mixins import ArenaMixinNode
    from ros_gz_interfaces.srv import ControlWorld, DeleteEntity

    from .gazebo_simulator.gazebo_simulator import GazeboHost

    del kwargs
    assert isinstance(node, ArenaMixinNode)
    semaphore = asyncio.Semaphore(5)
    return GazeboHost(
        node=node,
        semaphore=semaphore,
        service_control_world=node.create_client_wrapper(ControlWorld, "/world/default/control"),
        service_delete_entity=node.create_client_wrapper(DeleteEntity, "/world/default/remove"),
        logger=node.get_logger().get_child("GazeboHost"),
        physics_dt=physics_dt,
    )


# @SimulatorRegistry.register(SimSimulator.UNITY)
# async def lazy_unity(**kwargs):
#     from .unity_simulator import UnitySimulator
#     return UnitySimulator(**kwargs)


@SimulatorRegistry.register(SimSimulator.ISAAC)
async def lazy_isaac(**kwargs: typing.Unpack[SimKwargs]) -> BaseSim:
    from .isaac_simulator import IsaacSimulator

    return await IsaacSimulator.create(**kwargs)


@LifecycleRegistry.register(SimSimulator.ISAAC)
async def lazy_isaac_lifecycle(node: object, **kwargs: object) -> SimLifecycle:
    from arena_rclpy_mixins import ArenaMixinNode

    from .isaac_simulator import IsaacHost

    del kwargs
    assert isinstance(node, ArenaMixinNode)
    return IsaacHost(node=node)


@SimulatorRegistry.register(SimSimulator.MUJOCO)
async def lazy_mujoco(**kwargs: typing.Unpack[SimKwargs]) -> BaseSim:
    from .mujoco_simulator import MujocoSimulator

    return await MujocoSimulator.create(**kwargs)


@LifecycleRegistry.register(SimSimulator.MUJOCO)
async def lazy_mujoco_lifecycle(node: object, **kwargs: object) -> SimLifecycle:
    from arena_rclpy_mixins import ArenaMixinNode

    from .mujoco_simulator import MujocoHost

    del kwargs
    assert isinstance(node, ArenaMixinNode)
    return MujocoHost(node=node)
