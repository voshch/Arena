import typing

import attrs
from arena_simulation_setup.shared import Ceiling, Elevator, Schedule, Signal

from task_generator.shared import (
    Door,
    DynamicObstacle,
    Entity,
    Floor,
    FrameNamespace,
    Orientation,
    Pose,
    Position,
    Wall,
)

EntityPropsT = typing.TypeVar('EntityPropsT', bound=Entity)


class Realizer:
    @attrs.frozen()
    class _Configuration:
        x: float = 0.0
        y: float = 0.0
        prefix: str = ''

    _config: _Configuration
    _level_origins: dict[str, tuple[float, float]]

    @typing.overload
    def __init__(self, config: "Realizer._Configuration", level_origins: None = None): ...
    @typing.overload
    def __init__(self, config: None = None, level_origins: dict[str, tuple[float, float]] = ...): ...
    @typing.overload
    def __init__(self, config: None = None, level_origins: None = None): ...

    def __init__(self, config: "Realizer._Configuration | None" = None, level_origins: dict[str, tuple[float, float]] | None = None):
        self._config = config if config is not None else Realizer._Configuration()
        self._level_origins = level_origins if level_origins is not None else {}

    def set_origin(self, x: float, y: float, level_id: str = "") -> None:
        if level_id == "":
            self._config = attrs.evolve(self._config, x=x, y=y)
        else:
            if level_id not in self._level_origins:
                raise KeyError(f"level_id {level_id} is not registered on realizer")
            self._level_origins[level_id] = (x, y)

    def reset_level_origins(self, origins: dict[str, tuple[float, float]]) -> None:
        """Replace the per-level origin table, dropping any levels from a prior world."""
        self._level_origins = {str(level_id): (float(x), float(y)) for level_id, (x, y) in origins.items()}

    def get_config(self) -> "Realizer._Configuration":
        return self._config

    def get_level_origin(self, level_id: str = "") -> tuple[float, float]:
        if level_id == "":
            return (0.0, 0.0)
        if level_id not in self._level_origins:
            raise KeyError(f"level_id {level_id} is not registered on realizer")
        return self._level_origins[level_id]

    @typing.overload
    def realize(self) -> str: ...

    @typing.overload
    def realize(self, target: str) -> str: ...

    @typing.overload
    def realize(self, target: Position, level_id: str = "") -> Position: ...

    def _prefix(self, *s: str) -> str:
        return str(FrameNamespace(self._config.prefix)(*(p for p in s if p)))

    def prefix(self, *s: str) -> str:
        """Public: return the env-prefixed identifier for ``s`` (no pose realization)."""
        return self._prefix(*s)

    def _realize_position(self, position: Position, level_id: str = "") -> Position:
        level_x, level_y = self.get_level_origin(level_id)
        return Position(
            x=position.x + self._config.x + level_x,
            y=position.y + self._config.y + level_y,
            z=position.z,
        )

    def _realize_position_inv(self, position: Position, level_id: str = "") -> Position:
        level_x, level_y = self.get_level_origin(level_id)
        return Position(
            x=position.x - self._config.x - level_x,
            y=position.y - self._config.y - level_y,
            z=position.z,
        )

    def _realize_orientation(self, orientation: Orientation, level_id: str = "") -> Orientation:
        return Orientation(*orientation)

    def _realize_pose(self, pose: Pose, level_id: str = "") -> Pose:
        return Pose(self._realize_position(pose.position, level_id), self._realize_orientation(pose.orientation, level_id))

    def ezilear(self, target: Pose, level_id: str = "") -> Pose:
        """Inverse of realize: shift a map-frame pose back into abstract space."""
        return Pose(self._realize_position_inv(target.position, level_id), self._realize_orientation(target.orientation, level_id))

    @typing.overload
    def realize(self, target: EntityPropsT, level_id: str = "") -> EntityPropsT: ...

    @typing.overload
    def realize(self, target: Pose, level_id: str = "") -> Pose: ...

    def _realize_entity(self, entity: EntityPropsT, level_id: str = "") -> EntityPropsT:
        return attrs.evolve(
            entity,
            pose=self._realize_pose(entity.pose, level_id),
        )

    def _realize_dynamic_obstacle(self, obstacle: DynamicObstacle, level_id: str = "") -> DynamicObstacle:
        return attrs.evolve(
            obstacle,
            pose=self._realize_pose(obstacle.pose, level_id),
            waypoints=[self._realize_position(w.position, w.level_id if w.level_id is not None else "") for w in obstacle.waypoints],
        )

    @typing.overload
    def realize(self, target: Wall, level_id: str = "") -> Wall: ...

    def _realize_wall(self, wall: Wall, level_id: str = "") -> Wall:
        return attrs.evolve(
            wall,
            start=self._realize_position(wall.start, level_id),
            end=self._realize_position(wall.end, level_id),
        )

    @typing.overload
    def realize(self, target: Floor, level_id: str = "") -> Floor: ...

    def _realize_floor(self, floor: Floor, level_id: str = "") -> Floor:
        return attrs.evolve(
            floor,
            name=self._prefix(floor.name, level_id),
            pos=self._realize_position(floor.pos, level_id),
        )

    @typing.overload
    def realize(self, target: Ceiling, level_id: str = "") -> Ceiling: ...

    def _realize_ceiling(self, ceiling: Ceiling, level_id: str = "") -> Ceiling:
        return attrs.evolve(
            ceiling,
            name=self._prefix(ceiling.name, level_id),
            pos=self._realize_position(ceiling.pos, level_id),
        )

    @typing.overload
    def realize(self, target: Door, level_id: str = "") -> Door: ...

    def _realize_door(self, door: Door, level_id: str = "") -> Door:
        return attrs.evolve(
            door,
            name=self._prefix(door.name, level_id),
            start=self._realize_position(door.start, level_id),
            end=self._realize_position(door.end, level_id),
        )

    @typing.overload
    def realize(self, target: Elevator, level_id: str = "") -> Elevator: ...

    def _realize_destination(self, destination: str) -> str:
        """Translate a `<level_id>.<elevator_name>` destination into the realized elevator name."""
        if not destination:
            return destination
        dest_level_id, dest_name = destination.split('.', 1) if '.' in destination else ('', destination)
        return self._prefix(dest_name, dest_level_id)

    def _realize_elevator(self, elevator: Elevator, level_id: str = "") -> Elevator:
        return attrs.evolve(
            elevator,
            name=self._prefix(elevator.name, level_id),
            position=self._realize_position(elevator.position, level_id),
            destination=self._realize_destination(elevator.destination),
        )

    @typing.overload
    def realize(self, target: Schedule, level_id: str = "") -> Schedule: ...

    def _realize_schedule(self, schedule: Schedule, level_id: str = "") -> Schedule:
        return attrs.evolve(schedule, name=self._prefix(schedule.name, level_id))

    @typing.overload
    def realize(self, target: Signal, level_id: str = "") -> Signal: ...

    def _realize_signal(self, signal: Signal, level_id: str = "") -> Signal:
        return attrs.evolve(signal, name=self._prefix(signal.name, level_id))

    def realize_polygon(self, corners: typing.Sequence[Position], level_id: str = "") -> list[tuple[float, float]]:
        """Env-realized xy ring for a zone polygon, for occupancy_cap attach."""
        ring: list[tuple[float, float]] = []
        for corner in corners:
            realized = self._realize_position(corner, level_id)
            ring.append((realized.x, realized.y))
        return ring

    def realize(self, target: object = None, level_id: str = "") -> object:
        if target is None:
            return self._prefix(level_id)

        if isinstance(target, str):
            return self._prefix(target)

        if isinstance(target, Position):
            return self._realize_position(target, level_id)

        if isinstance(target, Pose):
            return self._realize_pose(target, level_id)

        if isinstance(target, Wall):
            return self._realize_wall(target, level_id)

        res = None

        if isinstance(target, DynamicObstacle):
            res = self._realize_dynamic_obstacle(target, level_id)

        elif isinstance(target, Entity):
            res = self._realize_entity(target, level_id)

        elif isinstance(target, Door):
            res = self._realize_door(target, level_id)

        elif isinstance(target, Floor):
            res = self._realize_floor(target, level_id)

        elif isinstance(target, Ceiling):
            res = self._realize_ceiling(target, level_id)

        elif isinstance(target, Elevator):
            res = self._realize_elevator(target, level_id)

        elif isinstance(target, Schedule):
            res = self._realize_schedule(target, level_id)

        elif isinstance(target, Signal):
            res = self._realize_signal(target, level_id)

        if res is None:
            raise TypeError(f'realization not implemented for type {type(target)}')

        res.sim_path = self._prefix(res.name, level_id)
        return res
