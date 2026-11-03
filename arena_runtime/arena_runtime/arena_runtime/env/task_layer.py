from __future__ import annotations

import asyncio
import importlib.metadata
import os
import typing
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence

import attrs
from arena_rclpy_mixins.Time import Time

if typing.TYPE_CHECKING:
    import tf2_ros
    from arena_rclpy_mixins import ArenaMixinNode
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer, ROSParamT
    from arena_simulation_setup.shared import Light
    from arena_simulation_setup.tree.World.Scenario import EpisodeCondition, TimelineEntry
    from diagnostic_msgs.msg import KeyValue

    from arena_runtime.acoustics import BaseAcousticsSimulator
    from arena_runtime.env.colors import RobotColors
    from arena_runtime.env.constants import Constants
    from arena_runtime.env.constants.rng import EpisodeRng
    from arena_runtime.env.environment_manager import EnvironmentManager
    from arena_runtime.env.interactive import MarkerHub
    from arena_runtime.env.realizer import Realizer
    from arena_runtime.env.request import TaskRequest
    from arena_runtime.env.robot_manager import RobotManager, RobotsManager
    from arena_runtime.env.robot_manager.task_runner import PhaseState
    from arena_runtime.env.world_manager.world_manager_ros import WorldManagerROS
    from arena_runtime.shared import DynamicObstacle, Obstacle, Pose, PositionRadius
    from arena_runtime.sim import BaseSim

TASK_LAYER_GROUP = "arena_runtime.task_layer"
TM_AXES = ("robots", "obstacles", "modules")
"""The axes the tm_* fields of recordings made before modes named."""
DEFAULT_TASK_LAYER = "task_generator"
TASK_LAYER_ENV = "ARENA_TASK_LAYER"
TASK_PARAM_PREFIX = "task."


TaskModes = dict[str, list[str]]
"""Mode names per axis, in the task layer's own axes and vocabulary."""


@attrs.define
class EpisodeRecord:
    episode_id: int = 0
    world: str = ""
    seed: int = -1
    modes: TaskModes = attrs.Factory(dict)
    robots: list[str] = attrs.Factory(list)
    outcome_state: int = 0
    outcome_info: str = ""
    goal_uuid: str = ""
    goal_dist_start: float = 0.0
    goal_dist_min: float = 0.0
    path_length: float = 0.0
    integrity: list[str] = attrs.Factory(list)
    start_time: Time = attrs.Factory(Time)


@attrs.define
class EpisodeContext:
    """What a task mode may tell the running episode."""

    compromise: Callable[[str], None]
    abort_episode: Callable[[str], None]
    force_reset: Callable[[], None]
    register_timeline: Callable[[Sequence[TimelineEntry], int], None]
    register_conditions: Callable[[Sequence[EpisodeCondition]], None]
    register_sound_levels: Callable[[dict[str, float]], None]


@attrs.define
class InteractiveContext:
    """The env's rviz side: interactive markers, robot colors and the tf buffer."""

    markers: MarkerHub
    robot_colors: RobotColors
    tf_buffer: tf2_ros.Buffer


@attrs.define
class TaskConfiguration:
    """The part of the env Configuration task modes read."""

    RNG: EpisodeRng
    HUMAN: ROSParamT[Constants.HumanSimulator]
    SAFE_DIST: ROSParamT[float]
    GOAL_TOLERANCE_RADIUS: ROSParamT[float]
    TIMEOUT: ROSParamT[float]
    KG_DERIVE: ROSParamT[str]

    @classmethod
    def of(cls, conf: type) -> TaskConfiguration:
        return cls(
            RNG=conf.General.RNG,
            HUMAN=conf.Arena.HUMAN,
            SAFE_DIST=conf.Obstacles.SAFE_DIST,
            GOAL_TOLERANCE_RADIUS=conf.Robot.GOAL_TOLERANCE_RADIUS,
            TIMEOUT=conf.Robot.TIMEOUT,
            KG_DERIVE=conf.Arena.KG_DERIVE,
        )


@attrs.define
class TaskContext:
    """The env's parts a task mode works on."""

    environment_manager: EnvironmentManager
    robots_manager: RobotsManager
    world_manager: WorldManagerROS
    realizer: Realizer
    simulator: BaseSim
    acoustics_simulator: BaseAcousticsSimulator
    reset_lock: asyncio.Lock
    conf: TaskConfiguration
    episode: EpisodeContext
    interactive: InteractiveContext | None = None

    @property
    def robots(self) -> dict[str, RobotManager]:
        return self.robots_manager.managers


@attrs.define
class RobotReset:
    """One robot's part of a ResetPlan."""

    start_pose: Pose
    task: TaskRequest | None = None
    humans: frozenset[str] = frozenset()


@attrs.define
class ResetPlan:
    """What the bound modes want for the next episode, applied by the env."""

    robots: dict[str, RobotReset] = attrs.Factory(dict)
    forbidden_zones: list[PositionRadius] = attrs.Factory(list)
    obstacles: list[Obstacle] = attrs.Factory(list)
    dynamic_obstacles: list[DynamicObstacle] = attrs.Factory(list)
    lights: Sequence[Light] = ()

    def merge(self, other: ResetPlan) -> None:
        """Take in a later plan: its entry replaces a robot's whole entry, its other fields add up."""
        self.robots.update(other.robots)
        self.forbidden_zones.extend(other.forbidden_zones)
        self.obstacles.extend(other.obstacles)
        self.dynamic_obstacles.extend(other.dynamic_obstacles)
        self.lights = (*self.lights, *other.lights)


@attrs.define
class TaskModeOverrides:
    """What the next episode changes: an axis left out of modes, an empty world and a negative seed keep what is live."""

    modes: TaskModes = attrs.Factory(dict)
    world: str = ""
    seed: int = -1

    def merge(self, other: TaskModeOverrides) -> None:
        self.modes.update(other.modes)
        self.world = other.world or self.world
        self.seed = other.seed if other.seed >= 0 else self.seed


class TaskParams(typing.Protocol):
    """The live mode selection."""

    def resolve(self, modes: TaskModes) -> TaskModes:
        """The full selection select(modes) would make live."""
        ...

    def select(self, modes: TaskModes) -> None: ...


class EpisodeTask(typing.Protocol):
    """One env's bound task modes, as the episode loop drives them."""

    @property
    def done(self) -> Awaitable[bool]: ...

    async def reset_robots(self, record: EpisodeRecord) -> ResetPlan: ...

    async def reset_obstacles(self, plan: ResetPlan, record: EpisodeRecord) -> None: ...

    async def tick(self, ended: Mapping[str, PhaseState], assigned: Collection[str]) -> None:
        """One sim tick of a running episode: the phase each robot ended on it and the robots given a task from outside since the last tick."""
        ...

    def after_reset(self) -> None: ...

    async def teardown(self) -> None: ...


class TaskLayer(typing.Protocol):
    """What an env needs from the package that holds the task modes."""

    def declare_task_params(self, node: ROSParamServer) -> TaskParams: ...

    async def create_task(self, *, node: ArenaMixinNode, ctx: TaskContext, params: TaskParams) -> EpisodeTask: ...

    def validate_modes(self, modes: TaskModes) -> TaskModes:
        """The listed axes' selection in canonical names, ValueError naming the first axis or mode that would not run."""
        ...

    def declare_mode_schema(self, node: ROSParamServer, axis: str, mode: str) -> None: ...

    def mode_namespaces(self) -> dict[str, dict[str, str]]: ...

    def fleet_only_robots(self) -> list[str]: ...

    async def next_episode(self, *, previous: EpisodeRecord, queued: TaskModeOverrides) -> TaskModeOverrides: ...


def task_layer_name(name: str = "") -> str:
    """name, else $ARENA_TASK_LAYER, else task_generator."""
    return name or os.environ.get(TASK_LAYER_ENV) or DEFAULT_TASK_LAYER


def task_layer_entry_point(name: str = "") -> importlib.metadata.EntryPoint:
    selected = task_layer_name(name)
    providers = list(importlib.metadata.entry_points(group=TASK_LAYER_GROUP))
    for provider in providers:
        if provider.name == selected:
            return provider
    raise LookupError(f"no task layer {selected!r} under the {TASK_LAYER_GROUP} entry point group, registered: {sorted(entry.name for entry in providers) or 'none'}")


def task_layer(name: str = "") -> TaskLayer:
    return task_layer_entry_point(name).load()


def registered_modes(layer: TaskLayer) -> TaskModes:
    return {axis: list(modes) for axis, modes in layer.mode_namespaces().items()}


def mode_param(axis: str) -> str:
    """The parameter that selects the modes of one axis."""
    return f"{TASK_PARAM_PREFIX}{axis}"


def legacy_mode_params() -> dict[str, str]:
    """tm_<axis> for the axes old recordings named, and the parameter that replaces it."""
    return {f"tm_{axis}": mode_param(axis) for axis in TM_AXES}


def selection_args(modes: Mapping[str, Sequence[str]]) -> list[str]:
    """k:=v launch arguments selecting these modes, an axis with no names left out."""
    return [f"{mode_param(axis)}:={','.join(names)}" for axis, names in modes.items() if names]


def validate_modes(modes: TaskModes, vocabulary: Mapping[str, Sequence[str]], multi_axes: Collection[str]) -> TaskModes:
    """The listed axes' selection in canonical names, ValueError naming the first axis or mode that would not run."""
    validated: TaskModes = {}
    for axis, names in modes.items():
        allowed = vocabulary.get(axis)
        if allowed is None:
            raise ValueError(f"unknown task axis {axis!r}, allowed: {', '.join(vocabulary)}")
        if axis not in multi_axes and len(names) != 1:
            raise ValueError(f"task axis {axis!r} takes exactly one mode, got {list(names)!r}, allowed: {', '.join(allowed)}")
        canonical = []
        for name in names:
            if name.lower() not in allowed:
                raise ValueError(f"unknown {axis} mode {name!r}, allowed: {', '.join(allowed)}")
            canonical.append(name.lower())
        validated[axis] = list(dict.fromkeys(canonical))
    return validated


def declare_mode_schemas(layer: TaskLayer, node: ROSParamServer) -> None:
    for axis, modes in layer.mode_namespaces().items():
        for mode in modes:
            layer.declare_mode_schema(node, axis, mode)


def encode_modes(modes: TaskModes) -> list[KeyValue]:
    """The wire form of modes: one KeyValue per axis, its names comma-separated."""
    from diagnostic_msgs.msg import KeyValue

    return [KeyValue(key=axis, value=",".join(names)) for axis, names in modes.items()]


def decode_modes(pairs: Iterable[KeyValue]) -> TaskModes:
    return {pair.key: [name for name in pair.value.split(",") if name] for pair in pairs}


def encode_namespaces(namespaces: Mapping[str, Mapping[str, str]]) -> list[KeyValue]:
    """The wire form of mode_namespaces(): one KeyValue per mode, keyed <axis>/<mode>."""
    from diagnostic_msgs.msg import KeyValue

    return [KeyValue(key=f"{axis}/{mode}", value=str(namespace)) for axis, modes in namespaces.items() for mode, namespace in modes.items()]


def decode_namespaces(pairs: Iterable[KeyValue]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for pair in pairs:
        axis, _, mode = pair.key.partition("/")
        out.setdefault(axis, {})[mode] = pair.value
    return out
