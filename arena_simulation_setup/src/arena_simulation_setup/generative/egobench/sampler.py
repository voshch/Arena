"""Samples of an EgoBench manifest: one world configuration and one encounter each, deterministic per seed."""

from __future__ import annotations

import hashlib
import itertools
import json
import random

import attrs

from arena_simulation_setup.generative.world import WorldGenerator

from .manifest import Axis, Encounter, Manifest, Pedestrian, Scalar, Space, axis_draw, axis_values


@attrs.frozen
class Sample:
    space: Space
    world: dict[str, Scalar]
    world_seed: int
    encounter: Encounter
    pedestrian: Pedestrian
    lateral_offset: float
    robot_speed: float
    ped_speed: float
    agent_type: str

    @property
    def world_key(self) -> str:
        """Name shared by every sample on the same generated world."""
        digest = hashlib.sha1(json.dumps([self.space.value, self.world, self.world_seed], sort_keys=True).encode()).hexdigest()[:8]
        return f"egobench_{self.space.value}_{digest}"

    @property
    def scenario_key(self) -> str:
        """Name of this sample's scenario inside its world."""
        return f"{self.encounter.value}__{self.pedestrian.value}__off{self.lateral_offset:.2f}__r{self.robot_speed:.2f}__p{self.ped_speed:.2f}__{self.agent_type}"

    def to_dict(self) -> dict[str, Scalar | dict[str, Scalar]]:
        return {
            "space": self.space.value,
            "world": dict(self.world),
            "world_seed": self.world_seed,
            "encounter": self.encounter.value,
            "pedestrian": self.pedestrian.value,
            "lateral_offset": self.lateral_offset,
            "robot_speed": self.robot_speed,
            "ped_speed": self.ped_speed,
            "agent_type": self.agent_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Scalar | dict[str, Scalar]]) -> Sample:
        world = data["world"]
        if not isinstance(world, dict):
            raise TypeError(f"sample world must be a mapping, got {world!r}")
        return cls(
            space=Space(data["space"]),
            world=dict(world),
            world_seed=int(_scalar(data["world_seed"])),
            encounter=Encounter(data["encounter"]),
            pedestrian=Pedestrian(data["pedestrian"]),
            lateral_offset=float(_scalar(data["lateral_offset"])),
            robot_speed=float(_scalar(data["robot_speed"])),
            ped_speed=float(_scalar(data["ped_speed"])),
            agent_type=str(_scalar(data["agent_type"])),
        )


def _scalar(value: Scalar | dict[str, Scalar]) -> Scalar:
    if isinstance(value, dict):
        raise TypeError(f"expected a scalar, got {value!r}")
    return value


def _world(space: Space, values: dict[str, Scalar]) -> dict[str, Scalar]:
    """The world configuration, checked by its generator's model."""
    WorldGenerator.config_model(space.generator).model_validate(values)
    return values


def _world_seed(manifest: Manifest, variant: int) -> int:
    return manifest.seed * 1000 + variant


def _make(manifest: Manifest, space: Space, world: dict[str, Scalar], variant: int, encounter: Encounter, pedestrian: Pedestrian, offset: Scalar, robot: Scalar, ped: Scalar, agent: Scalar) -> Sample:
    return Sample(
        space=space,
        world=_world(space, world),
        world_seed=_world_seed(manifest, variant),
        encounter=encounter,
        pedestrian=pedestrian,
        lateral_offset=float(offset),
        robot_speed=float(robot),
        ped_speed=float(ped),
        agent_type=str(agent),
    )


def _grid(manifest: Manifest) -> list[Sample]:
    samples: list[Sample] = []
    for space, axes in manifest.worlds.items():
        names = list(axes)
        for values in itertools.product(*(axis_values(axes[name]) for name in names)):
            world = dict(zip(names, values, strict=True))
            for variant, encounter, pedestrian, offset, robot, ped, agent in itertools.product(
                range(manifest.world_variants),
                manifest.encounters,
                manifest.pedestrians,
                axis_values(manifest.lateral_offset),
                axis_values(manifest.robot_speed),
                axis_values(manifest.ped_speed),
                axis_values(manifest.agent_type),
            ):
                samples.append(_make(manifest, space, world, variant, encounter, pedestrian, offset, robot, ped, agent))
    return samples


def _random(manifest: Manifest) -> list[Sample]:
    rng = random.Random(manifest.seed)
    spaces = list(manifest.worlds)
    samples: list[Sample] = []
    for _ in range(manifest.samples):
        space = rng.choice(spaces)
        axes: dict[str, Axis] = manifest.worlds[space]
        world = {name: axis_draw(axis, rng) for name, axis in axes.items()}
        samples.append(
            _make(
                manifest,
                space,
                world,
                rng.randrange(manifest.world_variants),
                rng.choice(manifest.encounters),
                rng.choice(manifest.pedestrians),
                axis_draw(manifest.lateral_offset, rng),
                axis_draw(manifest.robot_speed, rng),
                axis_draw(manifest.ped_speed, rng),
                axis_draw(manifest.agent_type, rng),
            )
        )
    return samples


def sample(manifest: Manifest) -> list[Sample]:
    """Every sample of the manifest, the whole grid or `samples` random draws, capped at `samples` on a grid when set."""
    if manifest.sampling == "random":
        return _random(manifest)
    samples = _grid(manifest)
    return samples[: manifest.samples] if manifest.samples else samples
