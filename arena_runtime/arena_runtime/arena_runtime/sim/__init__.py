from __future__ import annotations

import abc
import traceback

from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.tree import IdentifierProtocol
from task_generator.manager.realizer import Realizer

from arena_runtime._node import NodeInterface
from arena_runtime.constants import SimSimulator

from ._interface import MechanismITF, ObstacleITF, PedestrianITF, RobotITF, SimLifecycle, ViewportITF, WorldITF


class BaseSim(NodeInterface, ObstacleITF, PedestrianITF, RobotITF, WorldITF, MechanismITF, ViewportITF, abc.ABC):
    _namespace: Namespace
    _realizer: Realizer
    _env_id: int

    def __init__(self, *args: object, namespace: Namespace, realizer: Realizer, env_id: int = 0, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._namespace = namespace
        self._realizer = realizer
        self._env_id = env_id

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
async def lazy_dummy(**kwargs: object) -> BaseSim:
    from .dummy_simulator import DummySimulator

    return DummySimulator(**kwargs)


@LifecycleRegistry.register(SimSimulator.DUMMY)
async def lazy_dummy_lifecycle(**kwargs: object) -> SimLifecycle:
    del kwargs
    from .dummy_simulator import DummyHost

    return DummyHost()


# @SimulatorRegistry.register(SimSimulator.FLATLAND)
# async def lazy_flatland(**kwargs):
#     from .flatland_simulator import FlatlandSimulator
#     return FlatlandSimulator(**kwargs)


@SimulatorRegistry.register(SimSimulator.GAZEBO)
async def lazy_gazebo(**kwargs: object) -> BaseSim:
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
async def lazy_isaac(**kwargs: object) -> BaseSim:
    from .isaac_simulator import IsaacSimulator

    return await IsaacSimulator.create(**kwargs)


@LifecycleRegistry.register(SimSimulator.ISAAC)
async def lazy_isaac_lifecycle(node: object, **kwargs: object) -> SimLifecycle:
    from arena_rclpy_mixins import ArenaMixinNode

    from .isaac_simulator import IsaacHost

    del kwargs
    assert isinstance(node, ArenaMixinNode)
    return IsaacHost(node=node)
