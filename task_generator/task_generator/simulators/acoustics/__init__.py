"""Acoustics simulator axis: `acoustics:=<backend>` selects the acoustic simulator launched next to the human simulator."""

from __future__ import annotations

import abc
import typing
from collections.abc import Sequence

import attrs
from arena_rclpy_mixins.registry import AsyncFactoryRegistry as Registry
from arena_rclpy_mixins.shared import Namespace
from arena_runtime._node import NodeInterface
from task_generator_msgs.msg import AdapterDisplay, AdapterPlugin, RecordedTopic

from task_generator.constants import Constants
from task_generator.simulators.human.utils import stimulus_edge

if typing.TYPE_CHECKING:
    from task_generator.simulators.human import BaseHumanSimulator


@attrs.frozen
class SoundEmission:
    """One sounding or retired environment sound of one tick."""

    entity: str
    name: str
    group_id: str
    kind: str
    asset_id: str
    variant_id: str
    model: str
    tags: tuple[str, ...]
    loop: bool
    reference_distance_m: float
    level_db: float
    seed: int
    frame_id: str
    position: tuple[float, float, float]
    yaw: float
    active: bool
    program_start_ns: int


class PedestrianHearing:
    """Per (agent, sound kind) audibility over all sources, notifying the human simulator on every edge."""

    def __init__(self, human: BaseHumanSimulator) -> None:
        self._human = human
        self._heard: dict[tuple[int, str], bool] = {}
        self._sources: dict[tuple[int, str], dict[str, bool]] = {}

    def clear(self) -> None:
        self._heard.clear()
        self._sources.clear()

    async def on_heard(self, agent_id: int, kind: str, source_id: str, audible: bool) -> None:
        key = (agent_id, kind)
        sources = self._sources.setdefault(key, {})
        sources[source_id] = audible
        heard = any(sources.values())
        if stimulus_edge(self._heard.get(key), heard):
            await self._human.notify_stimulus(agent_id, kind, 1.0 if heard else 0.0)
        self._heard[key] = heard


class BaseAcousticsSimulator(NodeInterface, abc.ABC):
    """Node-side counterpart of the acoustics backend. Declares what the task generator must provide for it."""

    def __init__(self, *args: object, namespace: Namespace, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._namespace = namespace

    @property
    @abc.abstractmethod
    def requires_map_server(self) -> bool:
        """Whether the backend reads the map server (acoustic scene from the occupancy grid)."""

    def displays(self) -> tuple[AdapterDisplay, ...]:
        """Env-level rviz displays of the backend."""
        return ()

    def robot_displays(self, robot: str) -> tuple[AdapterDisplay, ...]:
        """Per-robot rviz displays, published as the robot's `_acoustics` adapter entry."""
        del robot
        return ()

    def plugins(self) -> tuple[AdapterPlugin, ...]:
        """Rviz panels and tools of the backend."""
        return ()

    def recorded_topics(self) -> tuple[RecordedTopic, ...]:
        """Topics the episode recorder subscribes to for this backend."""
        return ()

    def pedestrian_hearing(self, human: BaseHumanSimulator) -> PedestrianHearing | None:
        """Pedestrian hearing feeding the human simulator's stimuli, None when pedestrians hear nothing."""
        del human
        return None

    def emit_sounds(self, sounds: Sequence[SoundEmission]) -> None:
        """Environment sounds of one tick, a removed entity is simply absent."""
        del sounds

    def clear_sounds(self) -> None:
        """Forget every emitted sound at an episode reset."""


AcousticsSimulatorRegistry = Registry[Constants.AcousticsSimulator, BaseAcousticsSimulator]()


@AcousticsSimulatorRegistry.register(Constants.AcousticsSimulator.NONE)
async def none(**kwargs: object) -> BaseAcousticsSimulator:
    from .noop import NoopAcousticsSimulator

    return NoopAcousticsSimulator(**kwargs)


@AcousticsSimulatorRegistry.register(Constants.AcousticsSimulator.ARENA)
async def arena(**kwargs: object) -> BaseAcousticsSimulator:
    from .arena import ArenaAcousticsSimulator

    return ArenaAcousticsSimulator(**kwargs)
