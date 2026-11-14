"""The EgoBench world generator: one sample's world from its own generator, with the sample's encounter scenario packed beside it."""

from __future__ import annotations

import json
import random
import typing
from collections.abc import Callable

import pydantic

from arena_simulation_setup.generative.core.axes import Scalar
from arena_simulation_setup.generative.core.generated import Generated, Rejected, binding
from arena_simulation_setup.generative.core.profiles import PROFILES
from arena_simulation_setup.generative.study.manifest import check_world_fields
from arena_simulation_setup.generative.world import BaseConfiguration, WorldGenerator, WorldGeneratorImpl
from arena_simulation_setup.generative.world.layout import Layout
from arena_simulation_setup.tree.World import World

from . import scenario_corridor, scenario_open
from .manifest import Encounter, Pedestrian, Space
from .sampler import Sample

SCENARIO_GENERATORS: dict[Space, Callable[[Layout, Sample], Generated | Rejected]] = {
    Space.CORRIDOR: scenario_corridor.generate,
    Space.OPEN: scenario_open.generate,
}


class Unplaceable(ValueError):
    """The sample's encounter has no placement in its world."""


class WorldGeneratorEgoBench(WorldGeneratorImpl):
    """A corridor or open world with one pedestrian encounter scenario, failing when the encounter does not fit."""

    class Configuration(BaseConfiguration):
        space: str = Space.CORRIDOR.value
        world: str = pydantic.Field('{"corridor_width": 3.0}', json_schema_extra={"widget": "text"})  # the space's generator configuration as JSON
        world_seed: int = pydantic.Field(0, ge=0)
        encounter: str = Encounter.ONCOMING.value
        pedestrian: str = Pedestrian.FIXED.value
        lateral_offset: float = 1.2  # m, + = pedestrian on the robot's left
        robot_speed: float = pydantic.Field(0.5, gt=0)
        ped_speed: float = pydantic.Field(1.2, gt=0)
        agent_type: str = "adult"

        @pydantic.field_validator("space")
        @classmethod
        def _space(cls, value: str) -> str:
            return Space(value).value

        @pydantic.field_validator("agent_type")
        @classmethod
        def _agent_type(cls, value: str) -> str:
            if value not in PROFILES:
                raise ValueError(f"no pedestrian profile {value}, there are {', '.join(PROFILES)}")
            return value

        @pydantic.field_validator("encounter")
        @classmethod
        def _encounter(cls, value: str) -> str:
            return Encounter(value).value

        @pydantic.field_validator("pedestrian")
        @classmethod
        def _pedestrian(cls, value: str) -> str:
            return Pedestrian(value).value

    config: Configuration
    sample: Sample
    _world: WorldGenerator
    _generated: Generated | None

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        world = json.loads(self.config.world)
        if not isinstance(world, dict):
            raise ValueError(f"world must be a JSON object of {self.config.space} settings, got {self.config.world!r}")
        values: dict[str, Scalar] = world
        space = Space(self.config.space)
        check_world_fields(space.generator, values)
        self.sample = Sample(
            space=space,
            world=values,
            world_seed=self.config.world_seed,
            encounter=Encounter(self.config.encounter),
            pedestrian=Pedestrian(self.config.pedestrian),
            lateral_offset=self.config.lateral_offset,
            robot_speed=self.config.robot_speed,
            ped_speed=self.config.ped_speed,
            agent_type=self.config.agent_type,
        )
        self._world = WorldGenerator(space.generator, values, self.config.world_seed)
        self._generated = None
        self.warnings = []
        self.diagnostics = None

    @classmethod
    def of(cls, sample: Sample) -> WorldGeneratorEgoBench:
        """The generator for one manifest sample."""
        return cls(
            {
                "space": sample.space.value,
                "world": json.dumps(sample.world),
                "world_seed": sample.world_seed,
                "encounter": sample.encounter.value,
                "pedestrian": sample.pedestrian.value,
                "lateral_offset": sample.lateral_offset,
                "robot_speed": sample.robot_speed,
                "ped_speed": sample.ped_speed,
                "agent_type": sample.agent_type,
            },
            random.Random(sample.world_seed),
        )

    def compute(self) -> World:
        world = self._world.compute()
        self.diagnostics = self._world.diagnostics
        layout = self._world.layout()
        if layout is None:
            raise Unplaceable(f"the {self.config.space} generator leaves no layout to place an encounter on")
        result = SCENARIO_GENERATORS[self.sample.space](layout, self.sample)
        if isinstance(result, Rejected):
            raise Unplaceable(result.reason)
        self._generated = result
        return world

    def generated(self) -> Generated:
        if self._generated is None:
            self.compute()
        if self._generated is None:
            raise Unplaceable("compute placed no encounter")
        return self._generated

    def files(self) -> dict[str, bytes]:
        return {**self._world.files(), **super().files()}

    def scenarios(self) -> dict[str, Generated]:
        return {self.sample.scenario_key: self.generated()}

    def layout(self) -> Layout | None:
        return self._world.layout()

    def params(self) -> dict[str, typing.Any]:
        return binding(self.sample.scenario_key)
