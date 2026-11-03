import typing

import attrs
from arena_simulation_setup.shared import Ceiling, Elevator, Light, Placement, Schedule, Signal, Sound, Stairs

from arena_runtime.shared import (
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
PlacedT = typing.TypeVar('PlacedT', bound=Wall | Floor | Ceiling | Door | Elevator | Stairs | Light | Entity)


class Realizer:
    @attrs.frozen()
    class _Configuration:
        x: float = 0.0
        y: float = 0.0
        prefix: str = ''

    _config: _Configuration
    _level_origins: dict[str, tuple[float, float]]
    _elevator_levels: dict[str, str]
    _stair_levels: dict[str, str]
    _levels: list[str]
    _stack_offsets: dict[str, tuple[float, float, float]]
    _level_bounds: dict[str, tuple[float, float, float, float]]

    @typing.overload
    def __init__(self, config: "Realizer._Configuration", level_origins: None = None): ...
    @typing.overload
    def __init__(self, config: None = None, level_origins: dict[str, tuple[float, float]] = ...): ...
    @typing.overload
    def __init__(self, config: None = None, level_origins: None = None): ...

    def __init__(self, config: "Realizer._Configuration | None" = None, level_origins: dict[str, tuple[float, float]] | None = None):
        self._config = config if config is not None else Realizer._Configuration()
        self._level_origins = level_origins if level_origins is not None else {}
        self._elevator_levels = {}
        self._stair_levels = {}
        self._levels = []
        self._stack_offsets = {}
        self._level_bounds = {}

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

    def reset_elevator_levels(self, levels: typing.Mapping[str, str]) -> None:
        """Replace the level of every elevator cabin by cabin name, dropping those of a prior world."""
        self._elevator_levels = {str(cabin): str(level_id) for cabin, level_id in levels.items()}

    def reset_stair_levels(self, levels: typing.Mapping[str, str]) -> None:
        """Replace the level of every stair half by name, dropping those of a prior world."""
        self._stair_levels = {str(name): str(level_id) for name, level_id in levels.items()}

    def reset_placement(self, offsets: typing.Mapping[str, tuple[float, float, float]], bounds: typing.Mapping[str, tuple[float, float, float, float]]) -> None:
        """Replace the simulator offset and the level-local bounds of every level, an empty `offsets` keeping the simulator on the abstract layout."""
        self._stack_offsets = {str(level_id): (float(x), float(y), float(z)) for level_id, (x, y, z) in offsets.items()}
        self._level_bounds = {str(level_id): bound for level_id, bound in bounds.items()}

    def placement(self) -> Placement:
        """The map from the abstract frame into the simulator frame for the current world and env offset."""
        offsets = self._stack_offsets or {level_id: (x, y, 0.0) for level_id, (x, y) in self._level_origins.items()}
        return Placement(origins=dict(self._level_origins), offsets=dict(offsets), bounds=dict(self._level_bounds), env=(self._config.x, self._config.y))

    def place(self, target: PlacedT, level_id: str) -> PlacedT:
        """A realized entity of `level_id` moved from the abstract frame into the simulator frame."""
        dx, dy, dz = self.placement().shift(level_id)
        if (dx, dy, dz) == (0.0, 0.0, 0.0):
            return target

        def at(position: Position) -> Position:
            return Position(position.x + dx, position.y + dy, position.z + dz)

        if isinstance(target, Wall):
            return attrs.evolve(target, start=at(target.start), end=at(target.end))
        if isinstance(target, Floor):
            return attrs.evolve(target, pos=at(target.pos))
        if isinstance(target, Ceiling):
            return attrs.evolve(target, pos=at(target.pos), z=target.z + dz)
        if isinstance(target, Door):
            return attrs.evolve(target, start=at(target.start), end=at(target.end))
        if isinstance(target, Elevator | Stairs):
            placed = attrs.evolve(target, position=at(target.position))
            if isinstance(target, Stairs) and isinstance(placed, Stairs):
                placed.joined = target.joined
            return placed
        if isinstance(target, Light):
            return attrs.evolve(target, pose=None if target.pose is None else Pose(at(target.pose.position), target.pose.orientation), fixtures=[at(fixture) for fixture in target.fixtures])
        if isinstance(target, Entity):
            return attrs.evolve(target, pose=Pose(at(target.pose.position), target.pose.orientation))
        raise TypeError(f'placement not implemented for type {type(target)}')

    def reset_levels(self, level_ids: typing.Sequence[str]) -> None:
        """Replace the levels the env runs, dropping those of a prior world."""
        self._levels = [str(level_id) for level_id in level_ids]

    def placement_level(self, zone: str, level: str, listing: str = "") -> str:
        """The level a placed pose is local to: none when drawn from a zone, else its own, the one listing or holding it, or the sole level the env runs."""
        if zone:
            return ""
        return level or listing or (self._levels[0] if len(self._levels) == 1 else "")

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
            pose=self._realize_pose(entity.pose, self.placement_level(entity.placement_zone, entity.level, level_id)),
        )

    def _realize_dynamic_obstacle(self, obstacle: DynamicObstacle, level_id: str = "") -> DynamicObstacle:
        return attrs.evolve(
            obstacle,
            pose=self._realize_pose(obstacle.pose, self.placement_level(obstacle.placement_zone, obstacle.level, level_id)),
            waypoints=[self._realize_position(w.position, self.placement_level(w.zone, w.level, obstacle.level or level_id)) for w in obstacle.waypoints],
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

    def _realize_destination(self, cabin: str) -> str:
        """The realized name of a destination cabin, prefixed with the level it stands on."""
        return self._prefix(cabin, self._elevator_levels.get(cabin, ''))

    def _realize_elevator(self, elevator: Elevator, level_id: str = "") -> Elevator:
        return attrs.evolve(
            elevator,
            name=self._prefix(elevator.name, level_id),
            position=self._realize_position(elevator.position, level_id),
            destinations=[self._realize_destination(cabin) for cabin in elevator.destinations],
        )

    @typing.overload
    def realize(self, target: Stairs, level_id: str = "") -> Stairs: ...

    def _realize_stair(self, stair: Stairs, level_id: str = "") -> Stairs:
        realized = attrs.evolve(
            stair,
            name=self._prefix(stair.name, level_id),
            position=self._realize_position(stair.position, level_id),
            to=self._prefix(stair.to, self._stair_levels.get(stair.to, '')) if stair.to else '',
        )
        realized.joined = stair.joined
        return realized

    @typing.overload
    def realize(self, target: Schedule, level_id: str = "") -> Schedule: ...

    def _realize_schedule(self, schedule: Schedule, level_id: str = "") -> Schedule:
        return attrs.evolve(schedule, name=self._prefix(schedule.name, level_id))

    @typing.overload
    def realize(self, target: Signal, level_id: str = "") -> Signal: ...

    def _realize_signal(self, signal: Signal, level_id: str = "") -> Signal:
        return attrs.evolve(signal, name=self._prefix(signal.name, level_id))

    @typing.overload
    def realize(self, target: Sound, level_id: str = "") -> Sound: ...

    def _realize_sound(self, sound: Sound, level_id: str = "") -> Sound:
        return attrs.evolve(sound, name=self._prefix(sound.name, level_id))

    @typing.overload
    def realize(self, target: Light, level_id: str = "") -> Light: ...

    def _realize_light(self, light: Light, level_id: str = "") -> Light:
        env = self._prefix().strip("/")
        placed = None if light.spec.ambient else light.pose
        if placed is not None:
            level_id = self.placement_level(light.placement_zone, light.level, level_id)
        frame = light.frame
        if frame and env and frame != env and not frame.startswith(f"{env}/"):
            frame = f"{env}/{frame}"
        return attrs.evolve(
            light,
            name=self._prefix(light.name, level_id),
            frame=frame,
            pose=light.pose if placed is None else self._realize_pose(placed, level_id),
            fixtures=[self._realize_position(fixture, level_id) for fixture in light.fixtures],
        )

    def realize_polygon(self, corners: typing.Sequence[Position], level_id: str = "") -> list[tuple[float, float]]:
        """Env-realized xy ring for a zone polygon, for occupancy_cap attach."""
        ring: list[tuple[float, float]] = []
        for corner in corners:
            realized = self._realize_position(corner, level_id)
            ring.append((realized.x, realized.y))
        return ring

    def realize_reference(self, reference: str, level_id: str = "") -> str:
        """Realized name of an authored entity reference, `<level_id>.<name>` for another registered level, else a name on `level_id`."""
        if '.' in reference:
            ref_level_id, ref_name = reference.split('.', 1)
            if ref_level_id in self._level_origins or ref_level_id == level_id:
                return self._prefix(ref_name, ref_level_id)
        return self._prefix(reference, level_id)

    def realize_xy(self, x: float, y: float, level_id: str = "") -> tuple[float, float]:
        """Env-realized xy of a world point, for semantic params authored in world coordinates."""
        realized = self._realize_position(Position(x=x, y=y, z=0.0), level_id)
        return realized.x, realized.y

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

        elif isinstance(target, Stairs):
            res = self._realize_stair(target, level_id)

        elif isinstance(target, Schedule):
            res = self._realize_schedule(target, level_id)

        elif isinstance(target, Signal):
            res = self._realize_signal(target, level_id)

        elif isinstance(target, Sound):
            res = self._realize_sound(target, level_id)

        elif isinstance(target, Light):
            res = self._realize_light(target, level_id)

        if res is None:
            raise TypeError(f'realization not implemented for type {type(target)}')

        res.sim_path = self._prefix(res.name, level_id)
        return res
