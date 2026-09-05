"""Robot navigation-stack adapters."""

from __future__ import annotations

import asyncio
import enum
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import attrs
from arena_rclpy_mixins.registry import ClassRegistry
from arena_robots.Sensor import SensorSpec, SensorType, SensorTypeOrStr
from arena_viz.kinds import DisplayKind
from launch.actions import GroupAction

if TYPE_CHECKING:
    from arena_rclpy_mixins.shared import Namespace
    from arena_robots.bringup import Bringup
    from arena_robots.clients import Client
    from arena_robots.task_kinds import TaskKind

    from task_generator.manager.robot_manager.robot_manager import RobotManager
    from task_generator.shared import Pose
    from task_generator.tasks.robots.request import TaskPhase


@attrs.frozen
class ResetContext:
    """Immutable per-episode context handed to adapter reset methods."""

    rng: object
    start_pose: Pose | None = None
    episode_index: int = 0


class ActuatorCap(enum.StrEnum):
    """Canonical actuator-capability vocabulary."""

    MOBILE = "mobile"
    DRONE = "drone"
    MANIPULATOR = "arm"


type Cap = ActuatorCap | str


@attrs.frozen
class AdapterCtx:
    """Immutable config-time snapshot handed to an adapter."""

    namespace: Namespace
    robot_name: str
    frame: str
    task_generator_node: str
    env_namespace: str
    use_sim_time: bool
    base_frame: str
    odom_frame: str
    sensors: list[SensorSpec]
    tf_buffer: object
    node_handle: object


@attrs.frozen
class AdapterDisplayHint:
    """Declarative display entry attached to an adapter kind. Mirrors AdapterDisplay.msg."""

    name: str
    topic: str
    kind: DisplayKind
    topic_type: str = ""
    style_json: str = ""
    topic_must_exist: bool = False


@attrs.frozen
class AdapterMeta:
    """Canonical metadata block for an adapter class.

    Supply exactly one of `client` (shorthand for single-accept adapters) or
    `clients` (explicit per-TaskKind map for multi-accept adapters).  The
    constructor normalizes `client` into `clients` immediately via
    `__attrs_post_init__`, so callers can always read `meta.clients`.
    """

    accepts: frozenset[TaskKind] = attrs.field(converter=frozenset)
    bringup: type[Bringup]
    cap: str
    displays: tuple[AdapterDisplayHint, ...] = attrs.field(default=(), converter=tuple)
    client: type[Client] | None = None
    clients: dict[TaskKind, type[Client]] | None = None

    def __attrs_post_init__(self) -> None:
        if (self.client is None) == (self.clients is None):
            raise ValueError("AdapterMeta: supply exactly one of 'client' or 'clients'")
        if self.client is not None:
            normalized: dict[TaskKind, type[Client]] = {tk: self.client for tk in self.accepts}
            object.__setattr__(self, "clients", normalized)
            object.__setattr__(self, "client", None)

    @classmethod
    def attach(cls, **kwargs: object) -> Callable[[type], type]:
        meta = cls(**kwargs)

        def wrap(target: type) -> type:
            target._adapter_meta = meta
            return target

        return wrap


ADAPTERS: dict[str, ClassRegistry[str, type[Adapter]]] = {
    "mobile": ClassRegistry(),
    "arm": ClassRegistry(),
}


class Adapter(ABC):
    """Abstract base class for robot navstack adapters. Metadata is registry-driven."""

    kind: ClassVar[str]
    cap_displays: ClassVar[tuple[AdapterDisplayHint, ...]] = (
        AdapterDisplayHint(
            name="Robot Model",
            topic="{ns}/robot_description",
            topic_type="std_msgs/String",
            kind=DisplayKind.ROBOT_MODEL,
        ),
        AdapterDisplayHint(
            name="Odometry",
            topic="{ns}/odom",
            topic_type="nav_msgs/Odometry",
            kind=DisplayKind.ODOM,
        ),
    )

    def __init__(self, robot_manager: RobotManager, **bringup_kwargs: object) -> None:
        self.rm = robot_manager
        self._bringup_kwargs = bringup_kwargs
        self.bringup = self.bringup_cls(
            robot_manager.robot_view,
            str(robot_manager.namespace),
            parts=robot_manager.robot.resolved_request,
        )
        meta = self._meta()
        assert meta.clients is not None
        self._clients: dict[TaskKind, Client] = {
            tk: cls(
                robot_manager.robot_view,
                str(robot_manager.namespace),
                node=robot_manager.node,
                tf_buffer=robot_manager.tf_buffer,
            )
            for tk, cls in meta.clients.items()
        }

    @classmethod
    def _meta(cls) -> AdapterMeta:
        return cls._adapter_meta

    @property
    def accepts(self) -> frozenset[TaskKind]:
        return self._meta().accepts

    @property
    def bringup_cls(self) -> type[Bringup]:
        return self._meta().bringup

    @property
    def client_cls(self) -> type[Client]:
        meta = self._meta()
        assert meta.clients is not None
        if len(meta.clients) == 1:
            return next(iter(meta.clients.values()))
        raise RuntimeError("multi-kind adapter; use client_for(tk)")

    @property
    def client(self) -> Client:
        if len(self._clients) == 1:
            return next(iter(self._clients.values()))
        raise RuntimeError("multi-kind adapter; use client_for(tk)")

    def client_for(self, tk: TaskKind) -> Client:
        return self._clients[tk]

    @property
    def displays(self) -> tuple[AdapterDisplayHint, ...]:
        return (*self.cap_displays, *self._meta().displays)

    @property
    def requires(self) -> frozenset[str]:
        return self.bringup.requires

    def launch_description(self, ctx: AdapterCtx) -> GroupAction:
        return GroupAction(
            [
                *self.bringup._launch_actions(
                    use_sim_time=ctx.use_sim_time,
                    frame=ctx.frame,
                    task_generator_node=ctx.task_generator_node,
                    env_namespace=ctx.env_namespace,
                    sensors=ctx.sensors,
                    **self._bringup_kwargs,
                ),
            ]
        )

    async def ensure_services(self) -> None:
        """Bring up any shared singletons this adapter consumes. Called before per-robot launch."""
        return None

    async def wait_until_ready(
        self,
        robot: RobotManager,
        node_paths: set[str],
    ) -> None:
        await asyncio.gather(*(c.wait_ready() for c in self._clients.values()))

    async def await_ready(
        self,
        robot: RobotManager,
        node_paths: set[str],
        timeout: float,
    ) -> None:
        """wait_until_ready, bounded when timeout is finite. Expiry raises, failing the episode instead of wedging the env."""
        if not math.isfinite(timeout):
            await self.wait_until_ready(robot, node_paths)
            return
        try:
            async with asyncio.timeout(timeout) as scope:
                await self.wait_until_ready(robot, node_paths)
        except TimeoutError:
            if not scope.expired():
                raise
            endpoints = ", ".join(c.action_endpoint() for c in self._clients.values())
            raise TimeoutError(f"adapter {self.kind!r} for robot {robot.name!r} not ready after {timeout:.0f}s, still waiting on action servers [{endpoints}] or adapter nodes (the preceding 'waiting on' warnings name the holdout)") from None

    @abstractmethod
    async def dispatch_phase(
        self,
        phase: TaskPhase,
        robot: RobotManager,
    ) -> None: ...

    def is_phase_done(
        self,
        phase: TaskPhase,
        robot: RobotManager,
    ) -> bool | None:
        return self.client_for(phase.kind).is_done()

    async def before_move(
        self,
        pose: Pose,
        robot: RobotManager,
    ) -> None:
        return None

    async def on_move(
        self,
        pose: Pose,
        robot: RobotManager,
    ) -> None:
        return None

    async def on_reset(self, robot: RobotManager, ctx: ResetContext) -> None:
        return None

    async def on_controllers_active(self, robot: RobotManager) -> None:
        """Called once per robot bring-up after every spawned controller reports active."""
        return None

    async def teardown(self) -> None:
        """Release any out-of-band resources this adapter owns (subprocesses, threads). Called on robot destroy and env shutdown."""
        return None


__all__ = [
    "ADAPTERS",
    "ActuatorCap",
    "Adapter",
    "AdapterCtx",
    "AdapterDisplayHint",
    "AdapterMeta",
    "Cap",
    "DisplayKind",
    "ResetContext",
    "SensorSpec",
    "SensorType",
    "SensorTypeOrStr",
]

from . import arm, mobile  # noqa: F401, E402
