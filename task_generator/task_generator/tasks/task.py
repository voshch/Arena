import asyncio
import typing
from collections.abc import Sequence

import rclpy
import rclpy.publisher
import std_msgs.msg as std_msgs
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_runtime._node import NodeInterface

from task_generator.constants import Constants
from task_generator.manager.environment_manager import EnvironmentManager
from task_generator.manager.robot_manager import RobotsManager
from task_generator.manager.world_manager.world_manager_ros import WorldManager
from task_generator.shared import Pose
from task_generator.tasks.registry import (
    _REGISTRY_NAMESPACE,
    MODULE_MODES,
    OBSTACLES_MODES,
    ROBOTS_MODES,
    walk_schemas,
)
from task_generator.tasks.robots.adapters import ResetContext
from task_generator.tasks.robots.composite import (
    TM_Composite,
    _scoped_ctx,
    get_extra_tm_loader,
)
from task_generator.tasks.robots.fleet_manager import FleetManager, TaskModeSpec
from task_generator.tasks.robots.request import TaskKind, TaskRequest

from . import TaskContext
from .obstacles import TM_Obstacles
from .robots import TM_Robots

# import training.srv as training_srvs


class Task(NodeInterface):
    """Task class that comibnes task modes."""

    TOPIC_RESET_START = "reset_start"
    TOPIC_RESET_END = "reset_end"
    PARAM_RESETTING = "resetting"

    @classmethod
    def declare_parameters(cls, node: ROSParamServer):
        node.ROSParam[bool](cls.PARAM_RESETTING, True)
        walk_schemas(node)

    __reset_start: rclpy.publisher.Publisher
    __reset_end: rclpy.publisher.Publisher

    PARAM_TM_ROBOTS = "tm_robots"
    PARAM_TM_OBSTACLES = "tm_obstacles"

    __param_tm_robots: Constants.TaskMode.TM_Robots
    __param_tm_obstacles: Constants.TaskMode.TM_Obstacles

    __tm_robots: TM_Robots | None = None
    __tm_obstacles: TM_Obstacles | None = None

    _force_reset: bool
    _abort_reason: str | None

    @classmethod
    async def create(
        cls,
        *,
        environment_manager: EnvironmentManager,
        robots_manager: RobotsManager,
        world_manager: WorldManager,
        modules: Sequence[Constants.TaskMode.TM_Module] = (),
        **kwargs: object,
    ) -> "Task":
        self = cls(
            environment_manager=environment_manager,
            robots_manager=robots_manager,
            world_manager=world_manager,
            modules=modules,
            **kwargs,
        )
        await self.set_tm_robots(self.node.conf.TaskMode.TM_ROBOTS.value)
        await self.set_tm_obstacles(self.node.conf.TaskMode.TM_OBSTACLES.value)
        await self.robots_manager.set_up()
        return self

    _ctx: TaskContext

    @property
    def environment_manager(self) -> EnvironmentManager:
        return self._ctx.environment_manager

    @property
    def robots_manager(self) -> RobotsManager:
        return self._ctx.robots_manager

    @property
    def world_manager(self) -> WorldManager:
        return self._ctx.world_manager

    @property
    def robots(self) -> dict:
        return self._ctx.robots

    def __init__(
        self,
        *args: object,
        environment_manager: EnvironmentManager,
        robots_manager: RobotsManager,
        world_manager: WorldManager,
        modules: Sequence[Constants.TaskMode.TM_Module] = (),
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._force_reset = False
        self._abort_reason = None

        self._ctx = TaskContext(
            environment_manager=environment_manager,
            robots_manager=robots_manager,
            world_manager=world_manager,
            abort_episode=self.abort_episode,
        )

        robots_manager.bind_abort(self.abort_episode)

        self.__reset_start = self.node.create_publisher(std_msgs.Empty, 'reset_start', 1)
        self.__reset_end = self.node.create_publisher(std_msgs.Empty, 'reset_end', 1)

        self._logger.debug('initing modules')
        self.__modules = []
        for module in modules:
            cls = MODULE_MODES.get(module)
            meta = MODULE_MODES.meta(module)
            self.__modules.append(cls(ctx=self._ctx, namespace=meta.namespace, task=self, node=self.node))

    async def _tear_down_tm_robots(self) -> None:
        if self.__tm_robots is not None:
            await self.__tm_robots.teardown()
            self.__tm_robots = None

    async def set_tm_robots(self, tm_robots: Constants.TaskMode.TM_Robots):
        assert tm_robots in ROBOTS_MODES, f"TaskMode '{tm_robots}' for robots is not registered!"
        cls = ROBOTS_MODES.get(tm_robots)
        meta = ROBOTS_MODES.meta(tm_robots)
        new_mode = cls(ctx=self._ctx, namespace=meta.namespace, node=self.node)
        await self._tear_down_tm_robots()
        self.__tm_robots = new_mode
        self.__param_tm_robots = tm_robots

    async def set_tm_robots_composite(
        self,
        specs: typing.Sequence[TaskModeSpec],
    ) -> None:
        """Bind a multi-TM composite task mode via FleetManager allocation."""
        if not specs:
            raise ValueError("task_modes list is empty")

        allocation = FleetManager.match(
            list(specs),
            self._ctx.robots.values(),
        )

        sub_modes: list[TM_Robots] = []
        for spec, robots in allocation.items():
            # Resolve the TM class via the standard TaskMode enum;
            # fall back to the composite module's extra registry for TM_Null.
            try:
                enum_key = Constants.TaskMode.TM_Robots(spec.kind)
            except ValueError:
                enum_key = None
            if enum_key is not None and enum_key in ROBOTS_MODES:
                tm_cls = ROBOTS_MODES.get(enum_key)
                ns = ROBOTS_MODES.meta(enum_key).namespace
            else:
                extra = get_extra_tm_loader(spec.kind)
                if extra is None:
                    raise KeyError(f"task_mode kind {spec.kind!r} is not registered")
                tm_cls = extra()
                ns = _REGISTRY_NAMESPACE(spec.kind)

            scoped = _scoped_ctx(self._ctx, (r.name for r in robots))
            sub_modes.append(tm_cls(ctx=scoped, namespace=ns, node=self.node))

        await self._tear_down_tm_robots()
        self.__tm_robots = TM_Composite(
            ctx=self._ctx,
            namespace=_REGISTRY_NAMESPACE("composite"),
            node=self.node,
            sub_modes=sub_modes,
        )
        # No single enum value applies; sentinel prevents the
        # new_tm_robots != __param_tm_robots comparison in _reset_episode
        # from retriggering a rebind.
        self.__param_tm_robots = None  # type: ignore[assignment]

    async def set_tm_obstacles(self, tm_obstacles: Constants.TaskMode.TM_Obstacles):
        assert tm_obstacles in OBSTACLES_MODES, f"TaskMode '{tm_obstacles}' for obstacles is not registered!"
        cls = OBSTACLES_MODES.get(tm_obstacles)
        meta = OBSTACLES_MODES.meta(tm_obstacles)
        new_mode = cls(ctx=self._ctx, namespace=meta.namespace, node=self.node)
        if self.__tm_obstacles is not None:
            await self.__tm_obstacles.teardown()
        self.__tm_obstacles = new_mode
        self.__param_tm_obstacles = tm_obstacles

    async def teardown(self) -> None:
        """Release both task modes; called when the task itself goes down."""
        await self._tear_down_tm_robots()
        if self.__tm_obstacles is not None:
            await self.__tm_obstacles.teardown()
            self.__tm_obstacles = None

    async def _reset_episode(self, **kwargs: object) -> None:
        try:
            self.__reset_start.publish(std_msgs.Empty())

            self.node.conf.General.RNG.reseed(int(kwargs["seed"]))

            await self.robots_manager.set_up()
            await self.robots_manager.launch_pending()

            await self.environment_manager.before_reset_episode()

            try:
                self.node._apply_staged_params()

                if (new_tm_robots := self.node.conf.TaskMode.TM_ROBOTS.value) != self.__param_tm_robots:
                    await self.set_tm_robots(new_tm_robots)

                if (new_tm_obstacles := self.node.conf.TaskMode.TM_OBSTACLES.value) != self.__param_tm_obstacles:
                    await self.set_tm_obstacles(new_tm_obstacles)

                for module in self.__modules:
                    module.before_reset()

                await self.tm_robots.reset(**kwargs)

                obstacles, dynamic_obstacles = await self.tm_obstacles.reset(**kwargs)

                async def respawn():
                    await asyncio.gather(
                        self.environment_manager.spawn_dynamic_obstacles(dynamic_obstacles),
                        self.environment_manager.spawn_obstacles(obstacles),
                    )

                await self.environment_manager.respawn(respawn)

                robot_outcomes = await asyncio.gather(
                    *(
                        mgr.reset(
                            ResetContext(
                                rng=self.node.conf.General.RNG.stream("robot-adapter", name),
                                start_pose=self.tm_robots.start_poses.get(name),
                                episode_index=self.node._episodes.current.episode_id,
                            )
                        )
                        for name, mgr in self.robots_manager.managers.items()
                    ),
                    return_exceptions=True,
                )
                for name, outcome in zip(self.robots_manager.managers, robot_outcomes, strict=True):
                    if isinstance(outcome, BaseException):
                        self._logger.warning(f"robot {name!r} adapter reset failed: {outcome!r}")

                for module in self.__modules:
                    module.after_reset()
            finally:
                await self.environment_manager.after_reset_episode()

        except Exception as e:
            self.node.get_logger().error(repr(e))
            raise

        finally:
            self.__reset_end.publish(std_msgs.Empty())

    @property
    def abort_reason(self) -> str | None:
        return self._abort_reason

    def abort_episode(self, reason: str) -> None:
        self._abort_reason = reason
        self._force_reset = True

    async def reset(self, **kwargs: object) -> None:
        self._force_reset = False
        self._abort_reason = None
        await self._reset_episode(**kwargs)

    @property
    async def is_done(self) -> bool:
        return self._force_reset or await self.tm_robots.done

    @property
    def tm_obstacles(self) -> TM_Obstacles:
        assert self.__tm_obstacles is not None, "obstacle task mode is not bound"
        return self.__tm_obstacles

    @property
    def tm_robots(self) -> TM_Robots:
        assert self.__tm_robots is not None, "robot task mode is not bound"
        return self.__tm_robots

    async def set_robot_position(self, pose: Pose):
        """Broadcast a teleport to all robots (back-compat shim for RViz / training UI)."""
        await self.tm_robots.set_position(pose)
        self.node._flip_integrity()

    async def set_robot_goal(self, pose: Pose):
        """Broadcast a goal to all robots (back-compat shim for RViz / training UI)."""
        await self.tm_robots.set_goal(pose)
        self.node._flip_integrity()

    async def submit_task(self, request: TaskRequest, robot_name: str) -> None:
        """Submit a typed task request to a specific robot; bypasses TM_Robots."""
        robot = self._ctx.robots[robot_name]
        await robot.submit_task(request)

    def set_info(self, info: str) -> None:
        """Update the live info string shown in the RViz panel for the current episode."""
        self.node.set_episode_info(info)

    _TaskKindAlias = TaskKind  # keep TaskKind import live

    def force_reset(self):
        self._force_reset = True
