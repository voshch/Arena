import heapq
import itertools
import math
from collections.abc import Sequence

import attrs
import numpy as np
import scipy.ndimage
import shapely
import shapely.affinity
from arena_simulation_setup.shared import Elevator, Stairs
from arena_simulation_setup.tree.World import Level, World

from arena_runtime._node import NodeInterface
from arena_runtime.shared import WORLD_ENTITY_PREFIX, Position, PositionRadius

from .utils import MultiLevelMap, WorldMap, WorldOccupancy

_UNKNOWN_FOOTPRINT_RADIUS = 1.0
_CABIN_BACK_CLEARANCE = 0.45
_CABIN_DOOR_STANDOFF = 1.2
_CABIN_ROBOT_MARGIN = 0.1
_DOOR_OUTWARD = {'+x': (1.0, 0.0), '-x': (-1.0, 0.0), '+y': (0.0, 1.0), '-y': (0.0, -1.0)}
_STAIRS_MOUTH_STANDOFF = 1.0
_STAIRS_MIN_OVERSHOOT = 0.05


def _door_standoff(robot_radius: float) -> float:
    return max(_CABIN_DOOR_STANDOFF, robot_radius + _CABIN_ROBOT_MARGIN)


@attrs.frozen
class ElevatorHop:
    """One elevator ride: board `board_name` on `board_level`, get off at `dest_name` on `dest_level`. Positions in the map frame."""

    board_level: str
    board_name: str
    board_position: Position
    dest_level: str
    dest_name: str
    dest_position: Position | None
    target: str  # authored destination token for the boarding elevator's `target` field, '' for single-destination elevators
    radius: float  # inscribed cabin radius: the robot centre within it counts as inside the cabin


@attrs.frozen
class ElevatorRide:
    """Map-frame drive geometry of one hop for a robot disc of a given radius."""

    hop: ElevatorHop
    board_target: Position  # near the boarding cabin's back wall, the robot's disc inside the cabin
    board_approach: Position  # straight out from the boarding cabin's door
    dest_exit: Position | None  # straight out from the destination cabin's door
    door_standoff: float  # distance of the approach and exit points beyond their door face
    boarding_radius: float  # Elevator.boarding_radius of the boarding cabin for that robot radius


@attrs.frozen
class StairsHop:
    """One stair crossing: drive into the boarding half `board_name` on `board_level` and land on its partner `dest_name` on `dest_level`."""

    board_level: str
    board_name: str
    dest_level: str
    dest_name: str
    kind: str


@attrs.frozen
class StairsRide:
    """Map-frame drive geometry of one stair crossing for a robot disc of a given radius."""

    hop: StairsHop
    approach: Position  # on the axis outside the boarding mouth
    board_target: Position  # past the trigger line, the robot's disc clear of the seam wall unless it is too large to cross that way
    dest_exit: Position  # on the axis outside the partner's mouth
    board_yaw: float  # heading from the boarding mouth towards its seam
    exit_yaw: float  # heading out of the partner's mouth
    overshoot: float  # distance of board_target past the trigger line
    mouth_standoff: float  # distance of the approach and exit points beyond their mouth


LevelHop = ElevatorHop | StairsHop


def _occupancy_to_available(occupancy: np.ndarray, safe_dist_cells: float) -> np.ndarray:
    """Return (row, col) cells whose Euclidean safe_dist_cells neighbourhood is fully not-full. Off-map counts as full."""
    free = np.pad(WorldOccupancy.not_full(occupancy), 1, constant_values=False)
    distances = scipy.ndimage.distance_transform_edt(free)
    assert isinstance(distances, np.ndarray)
    clearance = distances[1:-1, 1:-1]
    return np.transpose(np.where(clearance > safe_dist_cells))


def _zone_mask(world_map: WorldMap, polygons: Sequence[shapely.Polygon]) -> np.ndarray:
    """Cells whose tf_grid2pos point lies inside any polygon."""
    mask = np.zeros(world_map.shape[:2], dtype=bool)
    for poly in polygons:
        mask |= world_map.tf_poly2mask(poly, offset=0.0)
    return mask


def _sample_from_candidates(
    available: np.ndarray,
    n: int,
    safe_dist_cells: float,
    rng: np.random.Generator,
    *,
    max_depth: int = 10,
) -> np.ndarray:
    """Pick n cells from `available` (row, col) keeping safe_dist_cells separation between picks.

    Returns an (n, 2) int array. Raises RuntimeError if fewer than n cells fit.
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.int64)

    if len(available) < n:
        raise RuntimeError(f"need {n} positions, only {len(available)} candidate cells available")

    accepted = np.zeros((n, 2), dtype=np.int64)
    accepted_n = 0

    for _ in range(max_depth):
        need = n - accepted_n
        if need <= 0:
            break
        if need > len(available):
            raise RuntimeError(f"need {need} more positions, only {len(available)} candidate cells available")
        for idx in rng.choice(len(available), need, replace=False):
            candidate = available[idx]
            if accepted_n and np.any(np.linalg.norm(accepted[:accepted_n] - candidate, axis=1) < safe_dist_cells):
                continue
            accepted[accepted_n] = candidate
            accepted_n += 1
            if accepted_n >= n:
                break

    if accepted_n < n:
        raise RuntimeError(f"failed to find {n} positions with safe_dist={safe_dist_cells} cells after {max_depth} retries")

    return accepted


def _sample_grid_positions(
    occupancy: np.ndarray,
    n: int,
    safe_dist_cells: float,
    rng: np.random.Generator,
    *,
    max_depth: int = 10,
) -> np.ndarray:
    """Pick n (row, col) cells with Euclidean safe_dist_cells clearance from non-empty cells and from each other."""
    available = _occupancy_to_available(occupancy, safe_dist_cells)
    return _sample_from_candidates(available, n, safe_dist_cells, rng, max_depth=max_depth)


class WorldManager(NodeInterface):
    """Used to get new goal, robot and obstacle positions from the static map."""

    _world: World
    _map: WorldMap  # main map used that has all of the levels
    _multi_map: MultiLevelMap | None  # utility reference maps for level-specific get position operations; actual occupancy is managed by _map
    _zone_masks: dict[str, np.ndarray]  # per level, cells inside that level's zones on map_for_floor(level)
    _stair_masks: dict[str, np.ndarray]  # per level and '' for the whole map, cells on a stair half

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._multi_map = None
        self._zone_masks = {}
        self._stair_masks = {}

    @property
    def world(self) -> World:
        return self._world

    def world_compacted(self) -> Level:
        """All loaded levels merged into one Level, shifted into the map frame."""
        origins = {fid: self._map.level_origins.get(fid, (0.0, 0.0)) for fid in self._world.level_ids}
        return self._world.compact_world(origins=origins)

    def level_of_point(self, x: float, y: float) -> str:
        """Return the loaded level id whose map-frame footprint contains (x, y), or '' if none."""
        for level_id, level in self._world.levels.items():
            corners = [corner for zone in level.zones for corner in zone.polygon]
            if not corners:
                continue
            ox, oy = self._map.level_origins.get(level_id, (0.0, 0.0))
            xs = [corner.x + ox for corner in corners]
            ys = [corner.y + oy for corner in corners]
            if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                return level_id
        return ""

    def _elevator_position(self, level_id: str, elevator_name: str) -> Position | None:
        level = self._world.get_level(level_id)
        if level is None:
            return None
        ox, oy = self._map.level_origins.get(level_id, (0.0, 0.0))
        for elevator in level.all_elevators:
            if elevator.name == elevator_name:
                return Position(x=elevator.position.x + ox, y=elevator.position.y + oy, z=elevator.position.z)
        return None

    def find_elevator(self, name: str) -> tuple[str, Elevator] | None:
        """(level id, elevator) for an authored elevator name, optionally `<level_id>.<name>`; None if unknown."""
        if '.' in name:
            level_id, bare = name.split('.', 1)
            level = self._world.get_level(level_id)
            if level is not None:
                for elevator in level.all_elevators:
                    if elevator.name == bare:
                        return level_id, elevator
        for level_id, level in self._world.levels.items():
            for elevator in level.all_elevators:
                if elevator.name == name:
                    return level_id, elevator
        return None

    def elevator_position(self, name: str) -> Position | None:
        """Map-frame cabin centre of an authored elevator name (optionally `<level_id>.<name>`), None if unknown."""
        found = self.find_elevator(name)
        return None if found is None else self._elevator_position(found[0], found[1].name)

    def elevator_destinations(self, elevator: Elevator) -> list[tuple[str, str, str]]:
        """Loaded destinations of an elevator as (level id, cabin name, authored name) in authored order, those on unloaded levels dropped."""
        cabins = self._world.elevator_levels()
        return [(cabins[cabin], cabin, cabin) for cabin in elevator.destinations if cabin in cabins]

    def _hop(self, board_level: str, elevator: Elevator, destination: tuple[str, str, str], n_destinations: int) -> ElevatorHop:
        dest_level, dest_name, token = destination
        board_position = self._elevator_position(board_level, elevator.name)
        assert board_position is not None
        return ElevatorHop(
            board_level=board_level,
            board_name=elevator.name,
            board_position=board_position,
            dest_level=dest_level,
            dest_name=dest_name,
            dest_position=self._elevator_position(dest_level, dest_name),
            target=token if n_destinations > 1 else '',
            radius=min(elevator.size[0], elevator.size[1]) / 2.0,
        )

    def _level_elevator(self, level_id: str, name: str) -> Elevator | None:
        level = self._world.get_level(level_id)
        return None if level is None else next((elevator for elevator in level.all_elevators if elevator.name == name), None)

    def elevator_ride(self, hop: ElevatorHop, robot_radius: float) -> ElevatorRide:
        """Ride target, door legs and boarding radius of `hop` for a robot disc of `robot_radius`."""
        board = self._level_elevator(hop.board_level, hop.board_name)
        assert board is not None
        dest = self._level_elevator(hop.dest_level, hop.dest_name)
        return ElevatorRide(
            hop=hop,
            board_target=self._cabin_back_target(board, hop.board_position, robot_radius),
            board_approach=self._cabin_door_standoff(board, hop.board_position, robot_radius),
            dest_exit=None if dest is None or hop.dest_position is None else self._cabin_door_standoff(dest, hop.dest_position, robot_radius),
            boarding_radius=board.boarding_radius(robot_radius),
            door_standoff=_door_standoff(robot_radius),
        )

    @staticmethod
    def _cabin_door_standoff(elevator: Elevator, centre: Position, robot_radius: float) -> Position:
        """Point straight out from the cabin's door face, `_CABIN_DOOR_STANDOFF` or the robot radius plus a margin away."""
        ox, oy = _DOOR_OUTWARD.get(str(elevator.door_side), (1.0, 0.0))
        out = (elevator.size[0] if ox else elevator.size[1]) / 2.0 + _door_standoff(robot_radius)
        return Position(x=centre.x + ox * out, y=centre.y + oy * out, z=centre.z)

    @staticmethod
    def _cabin_back_target(elevator: Elevator, centre: Position, robot_radius: float) -> Position:
        """Point on the door axis, `_CABIN_BACK_CLEARANCE` or the robot radius plus a margin in front of the back wall, never past the centre."""
        ox, oy = _DOOR_OUTWARD.get(str(elevator.door_side), (1.0, 0.0))
        half_depth = (elevator.size[0] if ox else elevator.size[1]) / 2.0
        inset = max(0.0, half_depth - max(_CABIN_BACK_CLEARANCE, robot_radius + _CABIN_ROBOT_MARGIN))
        return Position(x=centre.x - ox * inset, y=centre.y - oy * inset, z=centre.z)

    def _hops_from(self, level_id: str, climbs: frozenset[str]) -> list[LevelHop]:
        """Every hop leaving a level, crossings of stairs the robot climbs first, then elevator rides."""
        level = self._world.get_level(level_id)
        if level is None:
            return []
        hops: list[LevelHop] = []
        for stair in level.stairs:
            hop = self._stair_hop(level_id, stair)
            if hop is not None and hop.kind in climbs:
                hops.append(hop)
        for elevator in level.elevators:
            destinations = self.elevator_destinations(elevator)
            hops.extend(self._hop(level_id, elevator, destination, len(destinations)) for destination in destinations)
        return hops

    def level_hops(self, from_level: str, to_level: str, climbs: frozenset[str] = frozenset()) -> list[LevelHop]:
        """Fewest hops (map frame) from from_level to to_level over elevators and the stairs whose kind is in `climbs`, stairs winning ties, empty if none connect them."""
        if not from_level or not to_level or from_level == to_level:
            return []
        best: dict[str, tuple[int, int]] = {from_level: (0, 0)}
        prev: dict[str, LevelHop] = {}
        order = itertools.count()
        heap: list[tuple[tuple[int, int], int, str]] = [((0, 0), next(order), from_level)]
        while heap:
            cost, _, level_id = heapq.heappop(heap)
            if cost > best[level_id]:
                continue
            if level_id == to_level:
                break
            for hop in self._hops_from(level_id, climbs):
                reached = (cost[0] + 1, cost[1] + isinstance(hop, ElevatorHop))
                if hop.dest_level not in best or reached < best[hop.dest_level]:
                    best[hop.dest_level] = reached
                    prev[hop.dest_level] = hop
                    heapq.heappush(heap, (reached, next(order), hop.dest_level))
        if to_level not in prev:
            return []
        hops: list[LevelHop] = []
        cursor = to_level
        while cursor != from_level:
            hop = prev[cursor]
            hops.append(hop)
            cursor = hop.board_level
        hops.reverse()
        return hops

    def find_stair(self, name: str) -> tuple[str, Stairs] | None:
        """(level id, stair half) for an authored stair half name, optionally `<level_id>.<name>`, None if unknown."""
        if '.' in name:
            level_id, bare = name.split('.', 1)
            stair = self._level_stair(level_id, bare)
            if stair is not None:
                return level_id, stair
        for level_id in self._world.levels:
            stair = self._level_stair(level_id, name)
            if stair is not None:
                return level_id, stair
        return None

    def _level_stair(self, level_id: str, name: str) -> Stairs | None:
        level = self._world.get_level(level_id)
        return None if level is None else next((stair for stair in level.stairs if stair.name == name), None)

    def _stair_hop(self, board_level: str, stair: Stairs) -> StairsHop | None:
        """The crossing of a boarding half to its partner, None for an exit half or a partner on an unloaded level."""
        dest_level = self._world.stair_levels().get(stair.to) if stair.boards else None
        if dest_level is None:
            return None
        return StairsHop(board_level=board_level, board_name=stair.name, dest_level=dest_level, dest_name=stair.to, kind=stair.kind)

    def stair_hop(self, name: str, *, to: str = '', climbs: frozenset[str] = frozenset()) -> StairsHop:
        """One explicit crossing of the boarding stair half `name`, `to` naming its partner if given."""
        found = self.find_stair(name)
        if found is None:
            raise ValueError(f"via: unknown stair {name!r}")
        board_level, stair = found
        if stair.kind not in climbs:
            raise ValueError(f"via: stair {name!r} is {stair.kind}, the robot climbs {sorted(climbs)}")
        hop = self._stair_hop(board_level, stair)
        if hop is None:
            raise ValueError(f"via: stair {name!r} boards no loaded half (exit-only escalator half or partner on an unloaded level)")
        if to and to not in (hop.dest_name, f'{hop.dest_level}.{hop.dest_name}'):
            raise ValueError(f"via: stair {name!r} lands on {hop.dest_name!r}, not {to!r}")
        return hop

    def via_hop(self, name: str, *, to: str = '', toward: Position | None = None, climbs: frozenset[str] = frozenset()) -> LevelHop:
        """One explicit hop through the stair half or elevator `name`."""
        if self.find_stair(name) is not None:
            return self.stair_hop(name, to=to, climbs=climbs)
        return self.elevator_hop(name, to=to, toward=toward)

    def via_position(self, name: str) -> Position | None:
        """Map-frame boarding point of a stair half (its mouth) or elevator (its cabin centre), None if unknown."""
        found = self.find_stair(name)
        if found is not None:
            return self._stair_point(found[0], found[1], 0.0)
        return self.elevator_position(name)

    def _stair_point(self, level_id: str, stair: Stairs, along: float) -> Position:
        ox, oy = self._map.level_origins.get(level_id, (0.0, 0.0))
        point = stair.local(along, 0.0)
        return Position(x=point.x + ox, y=point.y + oy, z=point.z)

    def stair_ride(self, hop: StairsHop, robot_radius: float) -> StairsRide:
        """Approach, trigger and exit points of `hop` for a robot disc of `robot_radius`."""
        board = self._level_stair(hop.board_level, hop.board_name)
        dest = self._level_stair(hop.dest_level, hop.dest_name)
        assert board is not None and dest is not None
        trigger = board.length - board.inset
        along = board.length + robot_radius if board.joined else max(trigger + _STAIRS_MIN_OVERSHOOT, board.length - robot_radius - _STAIRS_MIN_OVERSHOOT)
        standoff = max(_STAIRS_MOUTH_STANDOFF, robot_radius + _CABIN_ROBOT_MARGIN)
        return StairsRide(
            hop=hop,
            approach=self._stair_point(hop.board_level, board, -standoff),
            board_target=self._stair_point(hop.board_level, board, along),
            dest_exit=self._stair_point(hop.dest_level, dest, -standoff),
            board_yaw=board.yaw,
            exit_yaw=math.remainder(dest.yaw + math.pi, math.tau),
            overshoot=along - trigger,
            mouth_standoff=standoff,
        )

    def elevator_hop(self, name: str, *, to: str = '', toward: Position | None = None) -> ElevatorHop:
        """One explicit ride on the authored elevator `name` to `to`, its only destination, or the cabin closest to `toward`."""
        found = self.find_elevator(name)
        if found is None:
            raise ValueError(f"via: unknown elevator {name!r}")
        board_level, elevator = found
        destinations = self.elevator_destinations(elevator)
        if not destinations:
            raise ValueError(f"via: elevator {name!r} has no loaded destination")
        if to:
            matches = [d for d in destinations if to in (d[2], d[1], f'{d[0]}.{d[1]}')]
            if not matches:
                raise ValueError(f"via: elevator {name!r} has no destination {to!r} (destinations: {[d[2] for d in destinations]})")
            chosen = matches[0]
        elif len(destinations) == 1 or toward is None:
            chosen = destinations[0]
        else:

            def distance(destination: tuple[str, str, str]) -> float:
                position = self._elevator_position(destination[0], destination[1])
                return math.inf if position is None else math.hypot(position.x - toward.x, position.y - toward.y)

            chosen = min(destinations, key=distance)
        return self._hop(board_level, elevator, chosen, len(destinations))

    @property
    def map(self) -> WorldMap:
        return self._map

    @property
    def level_maps(self) -> MultiLevelMap:
        return self._multi_map if self._multi_map is not None else MultiLevelMap.from_single(self._map)

    def map_for_floor(self, level_id: str = "") -> WorldMap:
        if self._multi_map is None:
            return self._map
        return self._multi_map.get_map(level_id) or self._map

    @property
    def _shape(self) -> tuple[int, int]:
        return self._map.shape[0], self._map.shape[1]

    @property
    def origin(self) -> Position:
        return self._map.origin

    @property
    def resolution(self) -> float:
        return self._map.resolution

    def update_world(
        self,
        world_map: WorldMap,
        world_description: World,
        multi_level_map: MultiLevelMap | None = None,
        static_footprints: Sequence[shapely.Polygon | None] | None = None,
    ):
        self._map = world_map
        self._multi_map = multi_level_map

        counter = itertools.count(0)
        for entity in itertools.chain(world_description.all_static_entities, world_description.all_dynamic_entities):
            if not entity.name:
                entity.name = f'{next(counter)}_{entity.model.name}'
            entity.name = f'{WORLD_ENTITY_PREFIX}{entity.name}'

        self._world = world_description

        self._zone_masks = {}
        self._stair_masks = {}
        whole_map_stairs: list[shapely.Polygon] = []
        for level_id, level in self.world.levels.items():
            polygons = [shapely.Polygon([(corner.x, corner.y) for corner in zone.polygon]) for zone in level.zones if len(zone.polygon) >= 3]
            if polygons:
                self._zone_masks[level_id] = _zone_mask(self.map_for_floor(level_id), polygons)
            stairs = [shapely.Polygon([(corner.x, corner.y) for corner in stair.corners()]) for stair in level.stairs]
            if stairs:
                self._stair_masks[level_id] = _zone_mask(self.map_for_floor(level_id), stairs)
                ox, oy = self.map.level_origins.get(level_id, (0.0, 0.0))
                whole_map_stairs.extend(shapely.affinity.translate(stair, xoff=ox, yoff=oy) for stair in stairs)
        if whole_map_stairs:
            self._stair_masks[''] = _zone_mask(self.map, whole_map_stairs)

        footprints = itertools.repeat(None) if static_footprints is None else static_footprints
        placed = ((level_id, obstacle) for level_id, level in self.world.levels.items() for obstacle in level.all_static_entities)
        for (level_id, obstacle), footprint in zip(placed, footprints, strict=static_footprints is not None):
            if footprint is None:
                r = _UNKNOWN_FOOTPRINT_RADIUS
                x, y = obstacle.pose.position.x, obstacle.pose.position.y
                footprint = shapely.box(x - r, y - r, x + r, y + r)
            level_origin = self.map.level_origins.get(level_id, (0.0, 0.0))
            self.map.occupancy.obstacle_occupy(self.map.tf_poly2mask(shapely.affinity.translate(footprint, xoff=level_origin[0], yoff=level_origin[1])))
            level_map = self.level_maps.get_map(level_id)
            if level_map is not None:
                level_map.occupancy.obstacle_occupy(level_map.tf_poly2mask(footprint))

    def forbid(self, forbidden_zones: list[PositionRadius], level_id: str = ""):
        world_map = self.map_for_floor(level_id)
        for zone in forbidden_zones:
            world_map.occupancy.forbidden_occupy(*world_map.tf_posr2rect(zone))

    def forbid_clear(self, level_id: str = ""):
        world_map = self.map_for_floor(level_id)
        world_map.occupancy.forbidden_clear()

    def get_positions_on_map(
        self,
        n: int,
        safe_dist: float,
        forbidden_zones: list[PositionRadius] | None = None,
        forbid: bool = True,
        polygon: shapely.Polygon | None = None,
        level_id: str = "",
    ) -> list[Position]:
        """Sample n map positions with Euclidean safe_dist (metres) clearance from obstacles and from each other.

        If `polygon` is given, candidate cells are restricted to those whose world coords lie inside it.
        If level_id is specified, return the local coordinate of the spawn position specific to that level.
        If not, return the position in the global (all-level) context.

        Raises RuntimeError if fewer than n positions fit.
        """
        if forbidden_zones is None:
            forbidden_zones = []

        # For level-specific queries, default to sampling inside that level's zones
        # (map rasters include padding around geometry, which is otherwise spawnable).
        zone_mask = self._zone_masks.get(level_id) if polygon is None and level_id else None

        select_from_whole_map = level_id == ""
        level_map = self.map if select_from_whole_map or self._multi_map is None else self._multi_map.select_map(level_id)
        origin = (0.0, 0.0) if select_from_whole_map else self.map.get_origin(level_id)
        origin_cell = (origin[1] / self.resolution, origin[0] / self.resolution)  # row, col order
        fork = level_map.occupancy.fork()
        for zone in forbidden_zones:
            fork.occupy(*level_map.tf_posr2rect(zone))

        safe_dist_cells = safe_dist / self.resolution
        rng = self.node.conf.General.RNG.stream("world", "positions")
        available = _occupancy_to_available(fork.grid, safe_dist_cells)

        stair_mask = self._stair_masks.get(level_id)
        if len(available) and stair_mask is not None:
            available = available[~stair_mask[available[:, 0], available[:, 1]]]
        if len(available) and polygon is not None:
            available = available[shapely.contains_xy(polygon, *level_map.tf_grid2xy(available[:, 0], available[:, 1]))]
        elif len(available) and zone_mask is not None:
            available = available[zone_mask[available[:, 0], available[:, 1]]]

        cells = _sample_from_candidates(available, n, safe_dist_cells, rng)

        if forbid:
            if select_from_whole_map:
                halo = int(math.ceil(safe_dist_cells))
                for row, col in cells:
                    fork.occupy((int(row) - halo, int(col) - halo), (int(row) + halo, int(col) + halo))
                fork.commit()

            # if selecting from a specific level, reflect that occupancy to the main map
            else:
                whole_map_fork = self.map.occupancy.fork()
                halo = int(math.ceil(safe_dist_cells))
                for row, col in cells:
                    whole_map_fork.occupy((int(row + origin_cell[0]) - halo, int(col + origin_cell[1]) - halo), (int(row + origin_cell[0]) + halo, int(col + origin_cell[1]) + halo))
                whole_map_fork.commit()

        return [level_map.tf_grid2pos((int(row), int(col))) for row, col in cells]

    def get_position_on_map(self, safe_dist: float, forbidden_zones: list[PositionRadius] | None = None, forbid: bool = True, level_id: str = "", polygon: shapely.Polygon | None = None) -> Position:
        return self.get_positions_on_map(n=1, safe_dist=safe_dist, forbidden_zones=forbidden_zones, forbid=forbid, level_id=level_id, polygon=polygon)[0]

    id_gen = itertools.count()
