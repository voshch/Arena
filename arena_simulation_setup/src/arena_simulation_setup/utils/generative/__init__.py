import abc
import enum
import random
import typing

import pydantic

from arena_simulation_setup.tree.World import LevelDescription

from .layout import Diagnostics, GridFrame, Note


class WorldGeneratorType(enum.Enum):
    """
    Enum for world generator types.
    """

    EMPTY = "empty"
    HALLWAY = "hallway"
    BARN = "barn"
    SKETCH = "sketch"
    LETTER = "letter"


class BaseConfiguration(pydantic.BaseModel):
    """Common to every generator. Extent is not here: a sketch or a text sizes itself from what is drawn."""

    resolution: float = 0.05  # m / px
    wall_gap: float = 0.05  # gap between adjacent walls


class PedestrianConfiguration(BaseConfiguration):
    pedestrians: int = pydantic.Field(-1, ge=-1)  # -1 leaves the episode's obstacle mode alone


class WorldGeneratorImpl(abc.ABC):
    """
    Abstract base class for world generators.
    """

    config: BaseConfiguration
    rng: random.Random
    diagnostics: 'Diagnostics | None'
    warnings: list['Note']

    def __init__(self, configuration: dict, rng: random.Random) -> None:
        super().__init__()
        self.rng = rng
        self.diagnostics = None
        self.warnings = []
        self.configure(configuration)

    def normalize(self, source: str) -> str:
        """Canonical form of this generator's textual input, unchanged unless the generator defines one."""
        return source

    @abc.abstractmethod
    def configure(self, configuration: dict): ...

    @abc.abstractmethod
    def compute(self) -> LevelDescription: ...

    def files(self) -> dict[str, bytes]:
        """Auxiliary artifacts packed into the world tar by relative path, none by default."""
        return {}

    def frame(self) -> 'GridFrame | None':
        """Cell grid of the last compute in world metres, for generators drawn on one."""
        return None

    def params(self) -> dict[str, typing.Any]:
        """Episode binding applied when this world is queued: tm_robots/tm_obstacles plus
        robots_params/obstacles_params leaf dicts. Empty (default) leaves the prior episode untouched."""
        return {}


class WorldGenerator:
    __registry: typing.ClassVar[dict[WorldGeneratorType, typing.Callable[[], type[WorldGeneratorImpl]]]] = {}
    _active: WorldGeneratorImpl

    @classmethod
    def register(cls, name: WorldGeneratorType) -> typing.Callable[[typing.Callable[[], type[WorldGeneratorImpl]]], typing.Callable[[], type[WorldGeneratorImpl]]]:
        def wrap(impl: typing.Callable[[], type[WorldGeneratorImpl]]) -> typing.Callable[[], type[WorldGeneratorImpl]]:
            cls.__registry[name] = impl
            return impl

        return wrap

    @classmethod
    def available(cls) -> list[WorldGeneratorType]:
        return list(cls.__registry.keys())

    @classmethod
    def config_model(cls, generator: WorldGeneratorType) -> type[BaseConfiguration]:
        if generator not in cls.__registry:
            raise ValueError(f"Generator {generator} has no implementation")
        return cls.__registry[generator]().Configuration

    def compute(self) -> LevelDescription:
        return self._active.compute()

    def files(self) -> dict[str, bytes]:
        return self._active.files()

    def params(self) -> dict[str, typing.Any]:
        return self._active.params()

    def normalize(self, source: str) -> str:
        return self._active.normalize(source)

    def frame(self) -> GridFrame | None:
        return self._active.frame()

    @property
    def diagnostics(self) -> Diagnostics | None:
        return self._active.diagnostics

    @property
    def warnings(self) -> list[Note]:
        return self._active.warnings

    def update_generator(self, generator: WorldGeneratorType, configuration: dict, seed: int = -1):
        if generator not in self.__registry:
            raise ValueError(f"Generator {generator} has no implementation")
        rng = random.Random(seed if seed >= 0 else None)  # negative seed = nondeterministic
        self._active: WorldGeneratorImpl = self.__registry[generator]()(configuration, rng)

    def __init__(self, generator: WorldGeneratorType, configuration: dict, seed: int = -1):
        self.update_generator(generator, configuration, seed)


class WithPedestrians(WorldGeneratorImpl):
    """A generator whose world carries a pedestrian count, bound as random dynamic obstacles."""

    config: PedestrianConfiguration

    def params(self) -> dict[str, typing.Any]:
        count = self.config.pedestrians
        if count < 0:
            return {}
        return {'tm_obstacles': 'random', 'obstacles_params': {'dynamic.n': [count, count]}}


@WorldGenerator.register(WorldGeneratorType.EMPTY)
def lazy_Empty() -> type[WorldGeneratorImpl]:
    from .empty import WorldGeneratorEmpty

    return WorldGeneratorEmpty


@WorldGenerator.register(WorldGeneratorType.HALLWAY)
def lazy_Hallway() -> type[WorldGeneratorImpl]:
    from .hallway import WorldGeneratorHallway

    return WorldGeneratorHallway


@WorldGenerator.register(WorldGeneratorType.BARN)
def lazy_Barn() -> type[WorldGeneratorImpl]:
    from .barn import WorldGeneratorBarn

    return WorldGeneratorBarn


@WorldGenerator.register(WorldGeneratorType.SKETCH)
def lazy_Sketch() -> type[WorldGeneratorImpl]:
    from .sketch import WorldGeneratorSketch

    return WorldGeneratorSketch


@WorldGenerator.register(WorldGeneratorType.LETTER)
def lazy_Letter() -> type[WorldGeneratorImpl]:
    from .letter import WorldGeneratorLetter

    return WorldGeneratorLetter
