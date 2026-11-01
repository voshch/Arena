import typing
from collections.abc import Sequence

from arena_runtime._node import NodeInterface
from arena_runtime.env.request import TaskRequest
from arena_runtime.env.task_layer import ResetPlan, TaskContext
from arena_runtime.shared import Pose

from task_generator.modes import Modes
from task_generator.tasks.registry import (
    _REGISTRY_NAMESPACE,
    MODULE_MODES,
    OBSTACLES_MODES,
    ROBOTS_MODES,
)
from task_generator.tasks.robots.composite import (
    TM_Composite,
    _scoped_ctx,
    get_extra_tm_loader,
)
from task_generator.tasks.robots.fleet_manager import FleetManager, TaskModeSpec, load_task_config

from .obstacles import TM_Obstacles
from .robots import TM_Robots

if typing.TYPE_CHECKING:
    from arena_runtime.env.episode_node import EpisodeNode

    from task_generator.tasks.layer import ModeParams


class Task(NodeInterface["EpisodeNode"]):
    """The task modes bound to one env."""

    __param_tm_robots: Modes.TM_Robots | None
    __param_tm_obstacles: Modes.TM_Obstacles
    __param_tm_config: str = ""
    __composite_fleet: frozenset[str] = frozenset()

    __tm_robots: TM_Robots | None = None
    __tm_obstacles: TM_Obstacles | None = None

    @classmethod
    async def create(cls, *, node: "EpisodeNode", ctx: TaskContext, params: "ModeParams", modules: Sequence[Modes.TM_Module] = ()) -> "Task":
        self = cls(node=node, ctx=ctx, params=params, modules=modules)
        tm_config = self._params.TM_CONFIG.value
        if not tm_config:
            await self.set_tm_robots(self._params.TM_ROBOTS.value)
        await self.set_tm_obstacles(self._params.TM_OBSTACLES.value)
        await ctx.robots_manager.set_up()
        if tm_config:
            await self.set_tm_robots_from_config(tm_config)
        return self

    _ctx: TaskContext

    def __init__(self, *, node: "EpisodeNode", ctx: TaskContext, params: "ModeParams", modules: Sequence[Modes.TM_Module] = ()) -> None:
        super().__init__(node=node)
        self._ctx = ctx
        self._params = params

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

    async def set_tm_robots(self, tm_robots: Modes.TM_Robots):
        assert tm_robots in ROBOTS_MODES, f"TaskMode '{tm_robots}' for robots is not registered!"
        cls = ROBOTS_MODES.get(tm_robots)
        meta = ROBOTS_MODES.meta(tm_robots)
        new_mode = cls(ctx=self._ctx, namespace=meta.namespace, node=self.node)
        await self._tear_down_tm_robots()
        self.__tm_robots = new_mode
        self.__param_tm_robots = tm_robots
        self.__param_tm_config = ""

    async def set_tm_robots_from_config(self, path: str) -> None:
        """Bind the composite described by a task config file, task.robots is ignored."""
        self._logger.info(f"task config {path} takes precedence over tm_robots")
        await self.set_tm_robots_composite(load_task_config(path))
        self.__param_tm_config = path
        self.__param_tm_robots = self._params.TM_ROBOTS.value

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
                enum_key = Modes.TM_Robots(spec.kind)
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
        self.__param_tm_robots = None
        self.__composite_fleet = frozenset(self._ctx.robots.keys())

    async def set_tm_obstacles(self, tm_obstacles: Modes.TM_Obstacles):
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

    async def prepare_reset(self, seed: int) -> ResetPlan:
        """Rebind the modes the params select, then reset them, with the sim paused."""
        if new_tm_config := self._params.TM_CONFIG.value:
            if new_tm_config != self.__param_tm_config or frozenset(self._ctx.robots.keys()) != self.__composite_fleet:
                await self.set_tm_robots_from_config(new_tm_config)
            elif (new_tm_robots := self._params.TM_ROBOTS.value) != self.__param_tm_robots:
                self._logger.warning(f"task.robots ignored while task config {new_tm_config} is bound")
                self.__param_tm_robots = new_tm_robots
        elif (new_tm_robots := self._params.TM_ROBOTS.value) != self.__param_tm_robots:
            await self.set_tm_robots(new_tm_robots)

        if (new_tm_obstacles := self._params.TM_OBSTACLES.value) != self.__param_tm_obstacles:
            await self.set_tm_obstacles(new_tm_obstacles)

        for module in self.__modules:
            module.before_reset()

        await self.tm_robots.reset()
        obstacles, dynamic_obstacles = await self.tm_obstacles.reset(seed=seed)
        return ResetPlan(obstacles=obstacles, dynamic_obstacles=dynamic_obstacles, lights=self.tm_obstacles.lights)

    def after_reset(self) -> None:
        for module in self.__modules:
            module.after_reset()

    @property
    async def done(self) -> bool:
        return await self.tm_robots.done

    @property
    def tm_obstacles(self) -> TM_Obstacles:
        assert self.__tm_obstacles is not None, "obstacle task mode is not bound"
        return self.__tm_obstacles

    @property
    def tm_robots(self) -> TM_Robots:
        assert self.__tm_robots is not None, "robot task mode is not bound"
        return self.__tm_robots

    async def set_robot_position(self, pose: Pose):
        """Broadcast a teleport to all robots."""
        await self.tm_robots.set_position(pose)
        self._ctx.compromise("robots moved by hand")

    async def set_robot_goal(self, pose: Pose):
        """Broadcast a goal to all robots."""
        await self.tm_robots.set_goal(pose)
        self._ctx.compromise("robot goal set by hand")

    async def submit_task(self, request: TaskRequest, robot_name: str) -> None:
        """Submit a typed task request to a specific robot; bypasses TM_Robots."""
        robot = self._ctx.robots[robot_name]
        await robot.submit_task(request)
