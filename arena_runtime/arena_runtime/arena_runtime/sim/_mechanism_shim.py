"""Shim implementations of MechanismITF for simulators without native door/elevator support."""

from __future__ import annotations

import asyncio
import enum
import math
import typing
from collections.abc import Sequence

import attrs
import rclpy.impl.rcutils_logger
from arena_simulation_setup.shared.world import CABIN_DOOR_INSET as DOOR_INSET
from arena_simulation_setup.shared.world import CABIN_DOOR_THRESHOLD, CABIN_DOOR_WIDTH, ActivationDistance, Stairs, StairsBox
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier

from arena_runtime.shared import Door, Elevator, Obstacle, Orientation, Pose, Position

from ._articulation import link_light_pose, step_articulated
from ._interface import _BOX_FLOOR_CLEARANCE

if typing.TYPE_CHECKING:
    from ._interface import MechanismITF

MECHANISM_TICK_RATE = 30.0  # Hz, sim time
WALL_THICKNESS = 0.05
# Sim seconds an unaccounted teleported rider holds a cabin before it resumes service
TELEPORT_GRACE = 10.0
HINGE_MAX_ANGLE = math.pi / 2.0

Disc = tuple[str, tuple[float, float], float]
Segment = tuple[tuple[float, float], tuple[float, float]]
DoorKind = typing.Literal['sliding', 'sliding_split', 'teleport', 'sliding_top', 'hinged']

_DOOR_AXES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    '+x': ((1.0, 0.0), (0.0, 1.0)),
    '-x': ((-1.0, 0.0), (0.0, 1.0)),
    '+y': ((0.0, 1.0), (1.0, 0.0)),
    '-y': ((0.0, -1.0), (1.0, 0.0)),
}


def _door_basis(elevator: Elevator) -> tuple[float, float, tuple[float, float], tuple[float, float], float]:
    outward, tangent = _DOOR_AXES[elevator.door_side or '+x']
    hx, hy = elevator.size[0] / 2.0, elevator.size[1] / 2.0
    out_extent = hx if outward[0] != 0 else hy
    tan_extent = hy if outward[0] != 0 else hx
    face_cx = elevator.position.x + outward[0] * out_extent
    face_cy = elevator.position.y + outward[1] * out_extent
    return face_cx, face_cy, outward, tangent, tan_extent


def _door_slot(elevator: Elevator) -> tuple[Position, Position]:
    face_cx, face_cy, outward, tangent, tan_extent = _door_basis(elevator)
    slot_half = tan_extent - WALL_THICKNESS / 2.0
    cx = face_cx - outward[0] * DOOR_INSET
    cy = face_cy - outward[1] * DOOR_INSET
    z = elevator.position.z
    return (
        Position(cx - tangent[0] * slot_half, cy - tangent[1] * slot_half, z),
        Position(cx + tangent[0] * slot_half, cy + tangent[1] * slot_half, z),
    )


def _elevator_wall_segments(elevator: Elevator) -> list[Segment]:
    """Back and side cabin walls as map-frame xy segments [back, side_pos, side_neg] for ped navigation."""
    face_cx, face_cy, _outward, tangent, tan_extent = _door_basis(elevator)
    back_cx = 2.0 * elevator.position.x - face_cx
    back_cy = 2.0 * elevator.position.y - face_cy
    tx, ty = tangent[0] * tan_extent, tangent[1] * tan_extent
    back_pos = (back_cx + tx, back_cy + ty)
    back_neg = (back_cx - tx, back_cy - ty)
    front_pos = (face_cx + tx, face_cy + ty)
    front_neg = (face_cx - tx, face_cy - ty)
    return [(back_neg, back_pos), (back_pos, front_pos), (back_neg, front_neg)]


def _elevator_wall_geometries(elevator: Elevator) -> list[tuple[str, tuple[float, float, float], Pose]]:
    face_cx, face_cy, outward, tangent, tan_extent = _door_basis(elevator)
    ex, ey, ez = elevator.size
    pos = elevator.position
    wall_z = pos.z + ez / 2.0

    def pose(cx: float, cy: float, cz: float | None = None) -> Pose:
        return Pose(position=Position(x=cx, y=cy, z=wall_z if cz is None else cz), orientation=Orientation.from_yaw(0.0))

    if outward[0] != 0:
        back_size = (WALL_THICKNESS, ey, ez)
        side_size = (ex, WALL_THICKNESS, ez)
    else:
        back_size = (ex, WALL_THICKNESS, ez)
        side_size = (WALL_THICKNESS, ey, ez)
    back_cx = 2.0 * pos.x - face_cx
    back_cy = 2.0 * pos.y - face_cy
    side_dx = tangent[0] * tan_extent
    side_dy = tangent[1] * tan_extent
    # cabins may overhang the authored floor polygons, so the cabin carries its own floor slab
    floor_z = pos.z + _BOX_FLOOR_CLEARANCE - WALL_THICKNESS / 2.0
    return [
        ('back', back_size, pose(back_cx, back_cy)),
        ('side_pos', side_size, pose(pos.x + side_dx, pos.y + side_dy)),
        ('side_neg', side_size, pose(pos.x - side_dx, pos.y - side_dy)),
        ('floor', (ex, ey, WALL_THICKNESS), pose(pos.x, pos.y, floor_z)),
    ]


class _DoorState(enum.StrEnum):
    CLOSED = 'closed'
    OPENING = 'opening'
    OPEN = 'open'
    CLOSING = 'closing'


class _CarState(enum.StrEnum):
    PRESENT = 'present'  # car here, door free to open
    DEPARTING = 'departing'  # car here, door closed, leaving this tick
    ABSENT = 'absent'  # car elsewhere in the group, door closed
    ARRIVING = 'arriving'  # car travelling here, door closed


@attrs.define
class _DoorRuntime:
    door: Door
    closed_pose: Pose
    open_pose: Pose
    effective_kind: DoorKind
    state: _DoorState = _DoorState.CLOSED
    progress: float = 0.0  # 0 = closed, 1 = open
    last_trigger_sim_time: float = -math.inf
    last_applied_progress: float = -1.0  # forces first move_box call
    # A close was refused because an agent is in the slab path, cleared once the path is clear.
    close_blocked: bool = False
    # Last wall state pushed to the human simulator (door cannot open for peds right now).
    ped_wall: bool = False


Occupants = tuple[tuple[str, tuple[float, float]], ...]


@attrs.define
class _Arrival:
    """One committed ride: occupants leaving `source` land in the owning cabin at `eta`."""

    eta: float
    source: str
    occupants: Occupants


@attrs.define
class _ElevatorRuntime:
    elevator: Elevator
    door_name: str
    # Realized destination cabin name(s), comma separated for a multi-destination elevator.
    destination_name: str
    # Rides heading here, ordered by eta.
    arrivals: list[_Arrival] = attrs.field(factory=list)
    # Tracks just-teleported occupants; value = whether inside-confirmed post-teleport.
    just_arrived: dict[str, bool] = attrs.field(factory=dict)
    # Sim time of the last arrival, unconfirmed just_arrived entries expire TELEPORT_GRACE after it.
    arrived_at: float = -math.inf
    # Where the group's car is relative to this cabin.
    car: _CarState = _CarState.PRESENT
    # The group's first cabin by name, where its car starts.
    home: bool = True
    # Every cabin of this cabin's group by name, linked through destinations.
    group: tuple[str, ...] = ()
    # Sim time of the oldest unanswered call while the car is elsewhere, inf when none.
    call_since: float = math.inf
    # Sim time the current boarders started boarding, inf when none are aboard.
    boarded_at: float = math.inf
    # Sim time the car last arrived here, -inf for the car's start.
    present_since: float = -math.inf
    # Cabin the departing car heads to.
    next_stop: str = ''
    # Two doors of this cabin's group were open at the last tick.
    door_conflict: bool = False
    # Occupants already dispatched this leg, still in the cabin until their teleport fires.
    dispatched: set[str] = attrs.field(factory=set)
    door_reported_open: bool = False
    # Sim time after which still-present dispatched riders are released.
    dispatch_deadline: float = math.inf
    # Agents inside the cabin footprint at the last step.
    live_occupants: tuple[str, ...] = ()
    # Multi-destination: written target (realized name) overrides the round-robin cursor.
    target_override: str | None = None
    rr_index: int = 0
    # Outside agents in the call radius at the last step.
    hall_callers: frozenset[str] = frozenset()

    @property
    def departing(self) -> bool:
        return self.car == _CarState.DEPARTING

    @property
    def destinations(self) -> tuple[str, ...]:
        return tuple(part.strip() for part in self.destination_name.split(',') if part.strip())

    def current_destination(self) -> str:
        """Destination the next departure goes to: written target, else round-robin, '' if none."""
        if self.target_override is not None:
            return self.target_override
        destinations = self.destinations
        if not destinations:
            return ''
        return destinations[self.rr_index % len(destinations)]

    def resolve_target(self, value: str) -> str:
        """Map an authored (`name`, `<level>.<name>`) or realized destination to its realized name."""
        token = value.strip()
        destinations = self.destinations
        if token in destinations:
            return token
        if '.' in token:
            level_id, name = token.split('.', 1)
            matches = [d for d in destinations if d == f"{name}/{level_id}" or d.endswith(f"/{name}/{level_id}")]
        else:
            matches = [d for d in destinations if d.endswith(f"/{token}")]
            if not matches:
                matches = [d for d in destinations if (d.split('/')[-2:-1] or [''])[0] == token]
        if len(matches) != 1:
            raise ValueError('malformed value')
        return matches[0]

    @property
    def full(self) -> bool:
        capacity = self.elevator.capacity
        return capacity is not None and len(self.live_occupants) >= capacity

    @property
    def arriving_eta(self) -> float:
        """ETA of the next arrival, -inf when nothing is in transit here."""
        return self.arrivals[0].eta if self.arrivals else -math.inf

    @property
    def pending_occupants(self) -> Occupants:
        """Every occupant of every queued arrival, in eta order."""
        return tuple(occ for arrival in self.arrivals for occ in arrival.occupants)

    def queue_arrival(self, arrival: _Arrival) -> None:
        self.arrivals.append(arrival)
        self.arrivals.sort(key=lambda a: a.eta)


@attrs.define
class _ElevatorStepResult:
    teleport_job: tuple[str, str, list[tuple[str, tuple[float, float]]]] | None = None
    missing_destination: bool = False
    # Riders released after TELEPORT_GRACE (dispatched still in the source / never seen at the destination).
    dispatch_timeout: tuple[str, ...] = ()
    arrival_timeout: tuple[str, ...] = ()


@attrs.define
class _StairsRuntime:
    stair: Stairs
    body: Stairs | None = None
    treads: dict[int, str] = attrs.field(factory=dict)
    # '' when unpaired
    partner: str = ''
    travel: float = 0.0
    running: bool = True
    reported_running: bool = True
    velocities: dict[str, tuple[float, float, float]] = attrs.field(factory=dict)
    tread_models: dict[int, Obstacle] = attrs.field(factory=dict)
    latched: set[str] = attrs.field(factory=set)
    members: tuple[str, ...] = ()
    peds: frozenset[str] = frozenset()
    just_crossed: bool = False


def _reset_stair(runtime: _StairsRuntime) -> None:
    """Reset a stair runtime to spawn defaults (belt at rest position, running, no latches)."""
    runtime.travel = 0.0
    runtime.running = True
    runtime.latched = set()
    runtime.members = ()
    runtime.peds = frozenset()
    runtime.just_crossed = False


def _reset_door(runtime: _DoorRuntime) -> None:
    """Reset a door runtime to spawn defaults (last_applied_progress = -1 forces a re-close next tick)."""
    runtime.state = _DoorState.CLOSED
    runtime.progress = 0.0
    runtime.last_trigger_sim_time = -math.inf
    runtime.last_applied_progress = -1.0
    runtime.close_blocked = False


def _reset_elevator(runtime: _ElevatorRuntime) -> None:
    """Reset an elevator runtime to spawn defaults (car at the group's first cabin, empty queue, round-robin from the first destination)."""
    runtime.arrivals = []
    runtime.just_arrived = {}
    runtime.arrived_at = -math.inf
    runtime.car = _CarState.PRESENT if runtime.home else _CarState.ABSENT
    runtime.call_since = math.inf
    runtime.boarded_at = math.inf
    runtime.present_since = -math.inf
    runtime.next_stop = ''
    runtime.door_conflict = False
    runtime.dispatched = set()
    runtime.dispatch_deadline = math.inf
    runtime.live_occupants = ()
    runtime.target_override = None
    runtime.rr_index = 0
    runtime.hall_callers = frozenset()


def reset_mechanisms(mech: MechanismITF) -> None:
    """Reset every door, elevator and stair runtime to spawn defaults for a deterministic episode start."""
    for runtime in mech._door_runtime.values():
        _reset_door(runtime)
    for stair_runtime in mech._stair_runtime.values():
        _reset_stair(stair_runtime)
    for name, runtime in mech._elevator_runtime.items():
        _reset_elevator(runtime)
        cabin = mech._door_runtime.get(f"{name}/door")
        if cabin is not None:
            _reset_door(cabin)


def _effective_kind(
    logger: rclpy.impl.rcutils_logger.RcutilsLogger,
    door: Door,
) -> DoorKind:
    """Return the animation kind (unknown kinds animate as sliding)."""
    if door.kind == 'hinged':
        return 'hinged'
    if door.kind == 'teleport':
        return 'teleport'
    if door.kind == 'sliding_top':
        return 'sliding_top'
    if door.kind == 'sliding_split':
        return 'sliding_split'
    return 'sliding'


def _door_geometry(door: Door) -> tuple[tuple[float, float, float], Pose]:
    """Return (size, closed_pose) for a door spanning start..end."""
    sx, sy, sz = door.start.x, door.start.y, door.start.z
    ex, ey, ez = door.end.x, door.end.y, door.end.z
    length = math.hypot(ex - sx, ey - sy)
    yaw = math.atan2(ey - sy, ex - sx)
    cx = (sx + ex) / 2.0
    cy = (sy + ey) / 2.0
    cz = (sz + ez) / 2.0 + door.height / 2.0
    return (length, door.width, door.height), Pose(
        position=Position(x=cx, y=cy, z=cz),
        orientation=Orientation.from_yaw(yaw),
    )


def _leaf_geometry(door: Door, effective_kind: str) -> tuple[tuple[float, float, float], Pose]:
    """Return (size, closed_pose) of the moving box: the whole span, or the start half for a split door."""
    size, closed_pose = _door_geometry(door)
    if effective_kind != 'sliding_split':
        return size, closed_pose
    length, width, height = size
    yaw = closed_pose.orientation.to_yaw()
    return (length / 2.0, width, height), Pose(
        position=Position(
            x=closed_pose.position.x - math.cos(yaw) * length / 4.0,
            y=closed_pose.position.y - math.sin(yaw) * length / 4.0,
            z=closed_pose.position.z,
        ),
        orientation=closed_pose.orientation,
    )


def _leaf_names(name: str, effective_kind: str) -> list[str]:
    """Box names of a door: `<name>`, or `<name>_a` (start half) and `<name>_b` (end half) for a split door."""
    return [f'{name}_a', f'{name}_b'] if effective_kind == 'sliding_split' else [name]


def _mirrored(door: Door, pose: Pose) -> Pose:
    """Pose point-mirrored through the span midpoint, keeping height and orientation."""
    return Pose(
        position=Position(
            x=door.start.x + door.end.x - pose.position.x,
            y=door.start.y + door.end.y - pose.position.y,
            z=pose.position.z,
        ),
        orientation=pose.orientation,
    )


def _leaf_poses(name: str, runtime: _DoorRuntime) -> dict[str, Pose]:
    """Box name to pose at runtime.progress, the end-half leaf of a split door mirrors the start half."""
    pose = _interp_pose(runtime)
    if runtime.effective_kind != 'sliding_split':
        return {name: pose}
    first, second = _leaf_names(name, runtime.effective_kind)
    return {first: pose, second: _mirrored(runtime.door, pose)}


def _hinge_angle(door: Door, progress: float) -> float:
    """Slab yaw at progress: closed yaw rotated by swing_direction * progress * 90 deg about start."""
    yaw = math.atan2(door.end.y - door.start.y, door.end.x - door.start.x)
    return yaw + door.swing_direction * progress * HINGE_MAX_ANGLE


def _hinged_pose(door: Door, closed_pose: Pose, progress: float) -> Pose:
    """Slab pose of a hinged door rotated about its start point by progress."""
    length = math.hypot(door.end.x - door.start.x, door.end.y - door.start.y)
    angle = _hinge_angle(door, progress)
    return Pose(
        position=Position(
            x=door.start.x + math.cos(angle) * length / 2.0,
            y=door.start.y + math.sin(angle) * length / 2.0,
            z=closed_pose.position.z,
        ),
        orientation=Orientation.from_yaw(angle),
    )


def _door_open_pose(door: Door, closed_pose: Pose, effective_kind: str) -> Pose:
    """Return the fully-open pose for a door (Z-drop for teleport, axis-slide for sliding, half-span slide back past start for the start leaf of a split door, 90 deg for hinged)."""
    if effective_kind == 'hinged':
        return _hinged_pose(door, closed_pose, 1.0)
    if effective_kind == 'teleport':
        return Pose(
            position=Position(
                x=closed_pose.position.x,
                y=closed_pose.position.y,
                z=closed_pose.position.z - 100.0,
            ),
            orientation=closed_pose.orientation,
        )
    if effective_kind == 'sliding_top':
        return Pose(
            position=Position(
                x=closed_pose.position.x,
                y=closed_pose.position.y,
                z=closed_pose.position.z + door.height,
            ),
            orientation=closed_pose.orientation,
        )
    # sliding: full-length slide along start->end axis, matching arena_isaac (axis * door.S.x)
    sx, sy = door.start.x, door.start.y
    ex, ey = door.end.x, door.end.y
    length = math.hypot(ex - sx, ey - sy)
    if length == 0:
        return closed_pose
    ux = (ex - sx) / length
    uy = (ey - sy) / length
    travel = -length / 2.0 if effective_kind == 'sliding_split' else length
    return Pose(
        position=Position(
            x=closed_pose.position.x + ux * travel,
            y=closed_pose.position.y + uy * travel,
            z=closed_pose.position.z,
        ),
        orientation=closed_pose.orientation,
    )


def _elevator_synthesized_door(elevator: Elevator) -> Door:
    """Wrap the inset door slot in a Door so the regular door animation pipeline handles it."""
    start, end = _door_slot(elevator)
    return Door(
        name=f'{elevator.name}/door',
        start=start,
        end=end,
        kind=elevator.door_kind,
        width=CABIN_DOOR_WIDTH,
        height=elevator.size[2],
        activation_distance=ActivationDistance((elevator.activation_distance, elevator.activation_distance)),
        transition_time=elevator.transition_time,
        hold_time=elevator.hold_time,
    )


def _inside_cabin(elevator: Elevator, pos_xy: tuple[float, float]) -> bool:
    """True if pos_xy falls within the cabin footprint (inclusive of boundary)."""
    ex, ey, _ = elevator.size
    return abs(pos_xy[0] - elevator.position.x) <= ex / 2.0 and abs(pos_xy[1] - elevator.position.y) <= ey / 2.0


def _is_triggered(door: Door, positions: list[tuple[float, float]]) -> bool:
    """True if any position is within activation_distance of the door segment."""
    radius = max(door.activation_distance)
    if radius <= 0.0:
        return False
    return _near_door_segment(door, positions, radius)


def _near_door_segment(door: Door, positions: list[tuple[float, float]], radius: float) -> bool:
    """True if any position is within `radius` of the door line segment (start->end)."""
    sx, sy = door.start.x, door.start.y
    ex, ey = door.end.x, door.end.y
    dx, dy = ex - sx, ey - sy
    seg_len_sq = dx * dx + dy * dy
    r2 = radius * radius
    for x, y in positions:
        if seg_len_sq <= 0.0:
            if (x - sx) ** 2 + (y - sy) ** 2 <= r2:
                return True
            continue
        t = ((x - sx) * dx + (y - sy) * dy) / seg_len_sq
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        px = sx + t * dx
        py = sy + t * dy
        if (x - px) ** 2 + (y - py) ** 2 <= r2:
            return True
    return False


def _segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    t = 0.0 if seg_len_sq <= 0.0 else min(1.0, max(0.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _swept_arc_blocked(door: Door, p_from: float, p_to: float, discs: Sequence[Disc]) -> bool:
    """True when the sector a hinged slab sweeps between two progress values overlaps any agent disc."""
    sx, sy = door.start.x, door.start.y
    length = math.hypot(door.end.x - sx, door.end.y - sy)
    if length <= 0.0 or not discs:
        return False
    half_w = door.width / 2.0
    lo, hi = min(p_from, p_to), max(p_from, p_to)
    a_lo, a_hi = _hinge_angle(door, lo), _hinge_angle(door, hi)
    yaw0 = _hinge_angle(door, 0.0)
    rel_min, rel_max = sorted((a_lo - yaw0, a_hi - yaw0))
    for _name, (x, y), radius in discs:
        for angle in (a_lo, a_hi):
            ex, ey = sx + math.cos(angle) * length, sy + math.sin(angle) * length
            if _segment_distance(x, y, sx, sy, ex, ey) <= radius + half_w:
                return True
        if rel_max > rel_min:
            rel = math.atan2(y - sy, x - sx) - yaw0
            rel = (rel + math.pi) % (2.0 * math.pi) - math.pi
            if rel_min <= rel <= rel_max and math.hypot(x - sx, y - sy) <= length + half_w + radius:
                return True
    return False


def _band_blocked(door: Door, bands: Sequence[tuple[float, float]], discs: Sequence[Disc]) -> bool:
    """True when any agent disc overlaps a slab-wide band along the span, each band (from, to) in span lengths from start."""
    sx, sy = door.start.x, door.start.y
    ex, ey = door.end.x, door.end.y
    length = math.hypot(ex - sx, ey - sy)
    if length <= 0.0 or not discs:
        return False
    ux, uy = (ex - sx) / length, (ey - sy) / length
    half_w = door.width / 2.0
    for a0, a1 in bands:
        along_center = length * (a0 + a1) / 2.0
        cx, cy = sx + ux * along_center, sy + uy * along_center
        half_l = length * abs(a1 - a0) / 2.0
        for _name, (x, y), radius in discs:
            dx, dy = x - cx, y - cy
            along = abs(dx * ux + dy * uy)
            across = abs(-dx * uy + dy * ux)
            if math.hypot(max(along - half_l, 0.0), max(across - half_w, 0.0)) <= radius:
                return True
    return False


def _swept_slab_blocked(runtime: _DoorRuntime, p_from: float, p_to: float, discs: Sequence[Disc]) -> bool:
    """True when the slab band swept between the two progress values overlaps any agent disc.
    Vertical kinds (teleport, sliding_top) occupy the slot itself regardless of progress,
    hinged doors sweep a sector about their start point, each leaf of a split door sweeps
    its own band."""
    door = runtime.door
    if runtime.effective_kind == 'hinged':
        return _swept_arc_blocked(door, p_from, p_to, discs)
    lo, hi = min(p_from, p_to), max(p_from, p_to)
    if runtime.effective_kind == 'sliding':
        return _band_blocked(door, [(lo, 1.0 + hi)], discs)
    if runtime.effective_kind == 'sliding_split':
        return _band_blocked(door, [(-hi / 2.0, 0.5 - lo / 2.0), (0.5 + lo / 2.0, 1.0 + hi / 2.0)], discs)
    return _band_blocked(door, [(0.0, 1.0)], discs)


def _sliding_entry_blocked(runtime: _DoorRuntime, p_from: float, p_to: float, discs: Sequence[Disc]) -> bool:
    """True when a wall pocket a sliding slab enters between the two progress values overlaps any agent disc: past the end, or past both ends for a split door."""
    if p_to <= p_from:
        return False
    if runtime.effective_kind == 'sliding_split':
        return _band_blocked(runtime.door, [(-p_to / 2.0, -p_from / 2.0), (1.0 + p_from / 2.0, 1.0 + p_to / 2.0)], discs)
    return _band_blocked(runtime.door, [(1.0 + p_from, 1.0 + p_to)], discs)


def _advance_state(runtime: _DoorRuntime, dt: float, now: float, discs: Sequence[Disc]) -> None:
    """Advance door state machine one tick: linear T-delta for sliding, instant snap for teleport.
    The slab never moves through an agent: a blocked close re-triggers the door (it reverses,
    tested against the full remaining path so nobody in the slot is approached), an opening step
    holds when the slab would enter an agent: the swing sector of a hinged door, the wall pocket
    of a sliding one. Vertical kinds open away from agents and are never open-blocked."""
    door = runtime.door
    if runtime.close_blocked and (runtime.progress <= 0.0 or not _swept_slab_blocked(runtime, 0.0, runtime.progress, discs)):
        runtime.close_blocked = False
    fresh = (now - runtime.last_trigger_sim_time) <= door.hold_time
    if runtime.effective_kind == 'teleport':
        if not fresh and _swept_slab_blocked(runtime, runtime.progress, 0.0, discs):
            runtime.last_trigger_sim_time = now
            runtime.close_blocked = runtime.progress > 0.0
            fresh = True
        runtime.progress = 1.0 if fresh else 0.0
        runtime.state = _DoorState.OPEN if fresh else _DoorState.CLOSED
        return
    # sliding (incl. sliding_top) and hinged: per-tick linear T delta over transition_time
    target = 1.0 if fresh else 0.0
    step = dt / max(door.transition_time, 1e-9)
    if target > runtime.progress:
        runtime.state = _DoorState.OPENING
        next_progress = min(1.0, runtime.progress + step)
        if runtime.effective_kind == 'hinged' and _swept_slab_blocked(runtime, runtime.progress, next_progress, discs):
            return
        if runtime.effective_kind in ('sliding', 'sliding_split') and _sliding_entry_blocked(runtime, runtime.progress, next_progress, discs):
            return
        runtime.progress = next_progress
        if runtime.progress >= 1.0:
            runtime.state = _DoorState.OPEN
    elif target < runtime.progress:
        if _swept_slab_blocked(runtime, 0.0, runtime.progress, discs):
            runtime.last_trigger_sim_time = now
            runtime.close_blocked = True
            runtime.state = _DoorState.OPEN if runtime.progress >= 1.0 else _DoorState.OPENING
            return
        runtime.state = _DoorState.CLOSING
        runtime.progress = max(0.0, runtime.progress - step)
        if runtime.progress <= 0.0:
            runtime.state = _DoorState.CLOSED
    else:
        runtime.state = _DoorState.OPEN if target == 1.0 else _DoorState.CLOSED


def _interp_pose(runtime: _DoorRuntime) -> Pose:
    """Pose at runtime.progress: rotation about start for hinged, else linear closed->open."""
    if runtime.effective_kind == 'hinged':
        return _hinged_pose(runtime.door, runtime.closed_pose, runtime.progress)
    a = runtime.closed_pose.position
    b = runtime.open_pose.position
    t = runtime.progress
    return Pose(
        position=Position(
            x=a.x + (b.x - a.x) * t,
            y=a.y + (b.y - a.y) * t,
            z=a.z + (b.z - a.z) * t,
        ),
        orientation=runtime.closed_pose.orientation,
    )


def _step_elevator(
    runtime: _ElevatorRuntime,
    door_runtime: _DoorRuntime,
    dest_runtime: _ElevatorRuntime | None,
    occupants: Sequence[tuple[str, tuple[float, float]]],
    outside_trigger: bool,
    now: float,
    outside_names: frozenset[str] = frozenset(),
    callers: frozenset[str] | None = None,
    group: typing.Mapping[str, _ElevatorRuntime] | None = None,
) -> _ElevatorStepResult:
    """`callers` are the outside agents in the call radius, None means one anonymous caller on outside_trigger, `group` the other cabins of the car's group by name."""
    result = _ElevatorStepResult()
    others = group or {}
    if callers is None:
        callers = frozenset({''}) if outside_trigger else frozenset()
    new_callers = callers - runtime.hall_callers
    runtime.hall_callers = callers

    # Update just_arrived tracking: confirm inside observation, clear on real exit.
    current = frozenset(name for name, _ in occupants)
    runtime.live_occupants = tuple(sorted(current))
    for name in list(runtime.just_arrived):
        if name in current:
            runtime.just_arrived[name] = True
        elif runtime.just_arrived[name] and name in outside_names:
            del runtime.just_arrived[name]
    if math.isfinite(runtime.arrived_at) and now > runtime.arrived_at + TELEPORT_GRACE:
        lost = tuple(name for name, confirmed in runtime.just_arrived.items() if not confirmed)
        for name in lost:
            del runtime.just_arrived[name]
        result.arrival_timeout = lost

    # Drop dispatched occupants once they have left the cabin (their teleport has fired).
    runtime.dispatched &= current
    if not runtime.dispatched:
        runtime.dispatch_deadline = math.inf
    elif now > runtime.dispatch_deadline:
        result.dispatch_timeout = tuple(sorted(runtime.dispatched))
        runtime.dispatched = set()
        runtime.dispatch_deadline = math.inf

    # The car lands here.
    if runtime.arrivals and now >= runtime.arrivals[0].eta:
        arrival = runtime.arrivals.pop(0)
        runtime.car = _CarState.PRESENT
        runtime.call_since = math.inf
        runtime.boarded_at = math.inf
        runtime.present_since = now
        door_runtime.last_trigger_sim_time = now
        if arrival.occupants:
            result.teleport_job = (arrival.source, runtime.elevator.name, list(arrival.occupants))
            runtime.just_arrived.update({name: False for name, _ in arrival.occupants})
            runtime.arrived_at = now
        return result

    # Our dispatched riders are mid-flight or the car is on its way here: the door stays shut.
    if runtime.car == _CarState.ARRIVING or runtime.arrivals or runtime.dispatched:
        return result

    new_occupants = current - runtime.just_arrived.keys()
    accepts = runtime.elevator.accept_outside_calls

    # The car is elsewhere: a caller outside or anyone shut inside calls it here.
    if runtime.car == _CarState.ABSENT:
        hall_call = accepts and (bool(new_callers) if runtime.elevator.call_mode == 'edge' else outside_trigger)
        if (new_occupants or hall_call) and runtime.call_since == math.inf:
            runtime.call_since = now
        return result

    if new_occupants:
        runtime.boarded_at = min(runtime.boarded_at, now)
    else:
        runtime.boarded_at = math.inf
    hold_time = runtime.elevator.hold_time
    pending = min((other.call_since for other in others.values()), default=math.inf)
    yielding = pending != math.inf and now - max(pending, runtime.present_since + runtime.elevator.transition_time) > hold_time

    # Outside trigger opens the door if this elevator accepts outside calls and has room.
    if outside_trigger and accepts and not runtime.full:
        if runtime.elevator.call_mode != 'edge':
            call = True
        else:
            idle = runtime.car != _CarState.DEPARTING and not new_occupants
            call = bool(new_callers) or (idle and (door_runtime.state == _DoorState.CLOSED or not current))
        if new_occupants and now - runtime.boarded_at > hold_time:
            call = False
        if not new_occupants and yielding:
            call = False
        if call:
            door_runtime.last_trigger_sim_time = now
            runtime.car = _CarState.PRESENT
            return result

    # Departing phase: the door has closed behind the riders, the car leaves for next_stop.
    if runtime.car == _CarState.DEPARTING:
        if door_runtime.state != _DoorState.CLOSED:
            return result
        target = others.get(runtime.next_stop)
        if target is None and dest_runtime is not None and runtime.next_stop == runtime.current_destination():
            target = dest_runtime
        if target is None:
            result.missing_destination = True
            runtime.car = _CarState.PRESENT
            door_runtime.last_trigger_sim_time = now
            return result
        eta = now + runtime.elevator.travel_time
        target.queue_arrival(_Arrival(eta=eta, source=runtime.elevator.name, occupants=tuple(occupants)))
        target.car = _CarState.ARRIVING
        runtime.car = _CarState.ABSENT
        runtime.dispatched = {name for name, _ in occupants}
        runtime.dispatch_deadline = eta + TELEPORT_GRACE
        runtime.just_arrived = {}
        runtime.boarded_at = math.inf
        if runtime.next_stop == runtime.current_destination() and runtime.target_override is None and len(runtime.destinations) > 1:
            runtime.rr_index = (runtime.rr_index + 1) % len(runtime.destinations)
        return result

    # Hold door open for just-arrived occupants (post-teleport), at most TELEPORT_GRACE.
    if runtime.just_arrived and now <= runtime.arrived_at + TELEPORT_GRACE:
        door_runtime.last_trigger_sim_time = now
        return result

    # Boarders aboard: once the door has fully closed, leave for the next destination.
    if new_occupants:
        if door_runtime.state == _DoorState.CLOSED:
            runtime.car = _CarState.DEPARTING
            runtime.next_stop = runtime.current_destination()
        return result

    # Only earlier arrivals inside: keep the door open for them unless a call elsewhere has waited past hold_time.
    if current and not yielding:
        door_runtime.last_trigger_sim_time = now
        return result

    # Idle car with the door closed answers the oldest call in its group, empty.
    if pending != math.inf:
        if door_runtime.state == _DoorState.CLOSED:
            runtime.car = _CarState.DEPARTING
            runtime.next_stop = min((other.call_since, name) for name, other in others.items())[1]
        return result

    return result


def _compute_teleport_destinations(
    elevator_runtime: dict[str, _ElevatorRuntime],
    source_name: str,
    dest_name: str,
    named_occupants: Sequence[tuple[str, tuple[float, float]]],
) -> dict[str, tuple[float, float]]:
    """Translate occupant positions from the source cabin frame into the destination cabin frame."""
    source_rt = elevator_runtime.get(source_name)
    dest_rt = elevator_runtime.get(dest_name)
    if source_rt is None or dest_rt is None:
        return {}
    src_pos = source_rt.elevator.position
    dst_pos = dest_rt.elevator.position
    out: dict[str, tuple[float, float]] = {}
    for agent_name, (x, y) in named_occupants:
        out[agent_name] = (dst_pos.x + (x - src_pos.x), dst_pos.y + (y - src_pos.y))
    return out


def _ensure_loop(mech: MechanismITF) -> None:
    """Start the mechanism tick loop if not already running."""
    if mech._mechanism_loop_task is None or mech._mechanism_loop_task.done():
        mech._mechanism_loop_task = asyncio.create_task(_loop(mech))


async def _loop(mech: MechanismITF) -> None:
    """Sim-time rate loop driving door animation and elevator state machine."""
    logger = mech.node.get_logger()
    with mech.node.sim_time_rate(MECHANISM_TICK_RATE) as (done, rate):
        while not done.is_set():
            try:
                dt = await rate.get()
            except asyncio.CancelledError:
                raise
            try:
                await _tick(mech, dt)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"mechanism shim tick failed: {e!r}")


def _report_incident(mech: MechanismITF, reason: str) -> None:
    mech.node.get_logger().warning(reason)
    if mech._incident_callback is not None:
        mech._incident_callback(reason)


async def _tick(mech: MechanismITF, dt: float) -> None:
    """Single tick: update triggers, advance state machines, dispatch move_box, articulated link moves and teleports."""
    robot_discs = list(mech.robot_discs())
    ped_discs: list[Disc] = list(mech._human_simulator.pedestrian_discs()) if mech._human_simulator is not None else []
    discs = robot_discs + ped_discs
    named_positions = [(name, xy) for name, xy, _radius in discs]
    ped_names: set[str] = {name for name, _xy, _radius in ped_discs}
    now = mech.node.sim_time.to_seconds()

    elev_occupant_idx: dict[str, set[int]] = {}
    all_occupant_idx: set[int] = set()
    for elev_name, runtime in mech._elevator_runtime.items():
        occ = {i for i, (_n, xy) in enumerate(named_positions) if _inside_cabin(runtime.elevator, xy)}
        elev_occupant_idx[elev_name] = occ
        all_occupant_idx.update(occ)
    outside_xys = [xy for i, (_n, xy) in enumerate(named_positions) if i not in all_occupant_idx]
    outside_names = frozenset(named_positions[i][0] for i in range(len(named_positions)) if i not in all_occupant_idx)
    all_xys = [xy for _n, xy in named_positions]

    elevator_door_names = set(mech._elevator_doors.values())
    teleport_jobs: list[tuple[str, str, list[tuple[str, tuple[float, float]]]]] = []
    logger = mech.node.get_logger()
    for elev_name, runtime in mech._elevator_runtime.items():
        door_runtime = mech._door_runtime.get(runtime.door_name)
        if door_runtime is None:
            continue
        occupants = [named_positions[i] for i in elev_occupant_idx[elev_name]]
        outside_trigger = _near_door_segment(door_runtime.door, outside_xys, CABIN_DOOR_THRESHOLD) if occupants else _is_triggered(door_runtime.door, outside_xys)
        callers: frozenset[str] | None = None
        if runtime.elevator.call_mode == 'edge':
            radius = CABIN_DOOR_THRESHOLD if occupants else max(door_runtime.door.activation_distance)
            callers = frozenset(name for i, (name, xy) in enumerate(named_positions) if i not in all_occupant_idx and radius > 0.0 and _near_door_segment(door_runtime.door, [xy], radius))
        destination = runtime.current_destination()
        dest_runtime = mech._elevator_runtime.get(destination)
        group = {name: mech._elevator_runtime[name] for name in runtime.group if name != elev_name and name in mech._elevator_runtime}
        if mech._semantics.elevator_recalled(elev_name, now):
            runtime.live_occupants = tuple(sorted(name for name, _ in occupants))
            if runtime.car in (_CarState.PRESENT, _CarState.DEPARTING):
                door_runtime.last_trigger_sim_time = now  # recall: hold cabin door open
                runtime.car = _CarState.PRESENT
                continue
            runtime.call_since = -math.inf
        result = _step_elevator(runtime, door_runtime, dest_runtime, occupants, outside_trigger, now, outside_names=outside_names, callers=callers, group=group)
        if result.missing_destination:
            logger.warning(f"Elevator {elev_name!r}: destination {destination!r} unknown; door held open.")
        if result.dispatch_timeout:
            logger.warning(f"Elevator {elev_name!r}: riders {list(result.dispatch_timeout)} still in the cabin {TELEPORT_GRACE:.0f}s after their ride landed (teleport failed?), releasing the cabin.")
        if result.arrival_timeout:
            logger.warning(f"Elevator {elev_name!r}: riders {list(result.arrival_timeout)} never seen in the cabin {TELEPORT_GRACE:.0f}s after arrival (teleport failed?), releasing the cabin.")
        if result.teleport_job is not None:
            teleport_jobs.append(result.teleport_job)

    # Regular doors take proximity from the full agent list. Elevator doors are gated above.
    for name, runtime in mech._door_runtime.items():
        if name in elevator_door_names:
            continue
        if _is_triggered(runtime.door, all_xys) and mech._semantics.trigger_allowed(name, now):
            runtime.last_trigger_sim_time = now

    mech._semantics.apply_plate_drives(now)
    _update_ped_walls(mech, now, elevator_door_names)

    pending: list[typing.Awaitable] = []
    for name, runtime in mech._door_runtime.items():
        _advance_state(runtime, dt, now, discs)
        if runtime.progress != runtime.last_applied_progress:
            runtime.last_applied_progress = runtime.progress
            pending.extend(mech.move_box(leaf, pose) for leaf, pose in _leaf_poses(name, runtime).items())
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    _check_door_conflicts(mech)
    await step_articulated(mech, dt, discs)

    if mech._human_simulator is not None:
        door_changes: dict[str, bool] = {}
        for elev_name, runtime in mech._elevator_runtime.items():
            door_runtime = mech._door_runtime.get(runtime.door_name)
            is_open = door_runtime is not None and door_runtime.state == _DoorState.OPEN
            if is_open != runtime.door_reported_open:
                runtime.door_reported_open = is_open
                door_changes[elev_name] = is_open
        if door_changes:
            await mech._human_simulator.elevator_doors_changed(door_changes)

    await _tick_stairs(mech, dt, robot_discs, ped_discs)

    # Run teleport jobs after door updates are dispatched.
    for source_name, dest_name, named_occupants in teleport_jobs:
        destinations = _compute_teleport_destinations(mech._elevator_runtime, source_name, dest_name, named_occupants)
        if not destinations:
            continue
        robot_destinations = {k: v for k, v in destinations.items() if k not in ped_names}
        ped_destinations = {k: v for k, v in destinations.items() if k in ped_names}
        failed: list[str] = []
        for sim_path, (x, y) in robot_destinations.items():
            current = mech.robot_pose(sim_path)
            z = current.position.z if current is not None else 0.0
            orientation = current.orientation if current is not None else Orientation(1, 0, 0, 0)
            try:
                failure = "" if await mech.set_robot_pose(sim_path, Pose(position=Position(x, y, z), orientation=orientation)) else "the simulator refused the move"
            except Exception as e:
                failure = repr(e)
            if failure:
                _report_incident(mech, f"{sim_path}: elevator teleport {source_name!r} -> {dest_name!r} failed: {failure}")
                failed.append(sim_path)
        if ped_destinations:
            if mech._human_simulator is not None:
                try:
                    failure = "" if await mech._human_simulator.pedestrian_teleport(ped_destinations) else "the human simulator refused the move"
                except Exception as e:
                    failure = repr(e)
                if failure:
                    _report_incident(mech, f"{', '.join(ped_destinations)}: elevator teleport {source_name!r} -> {dest_name!r} failed: {failure}")
                    failed.extend(ped_destinations)
            else:
                logger.info(f"Elevator teleport (no-op, no human sim): {len(ped_destinations)} peds {source_name!r} -> {dest_name!r}")
        if failed:
            _release_failed_riders(mech, source_name, dest_name, failed)

    mech._semantics.step(now)
    await _apply_lights(mech)


LIGHT_MOVE_EPSILON = 0.02


def light_anchor(frame_position: Position, frame_yaw: float, offset: Position) -> Position:
    """Map position of an offset given in a frame with the given map position and yaw."""
    return Pose(position=frame_position, orientation=Orientation.from_yaw(frame_yaw)) * offset


def _light_moved(last: tuple[Position, float] | None, position: Position, yaw: float) -> bool:
    if last is None:
        return True
    last_position, last_yaw = last
    turn = abs(math.remainder(yaw - last_yaw, math.tau))
    return math.dist((last_position.x, last_position.y, last_position.z), (position.x, position.y, position.z)) > LIGHT_MOVE_EPSILON or turn > LIGHT_MOVE_EPSILON


async def _apply_lights(mech: MechanismITF) -> None:
    """Move frame-anchored and link-riding lights that moved, then push every light whose output changed since its last push."""
    for entity, light in mech._lights.items():
        if light.link and light.owner in mech._articulated:
            target = link_light_pose(mech._articulated[light.owner], light)
            if target is None:
                continue
            position, yaw = target
            if _light_moved(mech._light_poses.get(entity), position, yaw):
                await mech._move_light(light, position, yaw)
                mech._light_poses[entity] = (position, yaw)
            continue
        if not light.frame:
            continue
        pose = mech.frame_pose(light.frame)
        if pose is None:
            continue
        position = light_anchor(pose[0], pose[1], light.position)
        if _light_moved(mech._light_poses.get(entity), position, pose[1]):
            await mech._move_light(light, position, pose[1])
            mech._light_poses[entity] = (position, pose[1])
    for entity, output in mech._semantics.light_outputs().items():
        light = mech._lights.get(entity)
        if light is None or mech._light_outputs.get(entity) == output:
            continue
        level, dead_fraction = output
        await mech._apply_light(light, level, light.alive(dead_fraction))
        mech._light_outputs[entity] = output


def _check_door_conflicts(mech: MechanismITF) -> None:
    """Report an elevator group with more than one cabin door not closed, once per occurrence."""
    for name, runtime in mech._elevator_runtime.items():
        if not runtime.home:
            continue
        unclosed = []
        for member in runtime.group or (name,):
            member_rt = mech._elevator_runtime.get(member)
            door = mech._door_runtime.get(member_rt.door_name) if member_rt is not None else None
            if door is not None and door.state != _DoorState.CLOSED:
                unclosed.append(member)
        conflict = len(unclosed) > 1
        if conflict and not runtime.door_conflict:
            _report_incident(mech, f"Elevator group {name!r}: doors of {unclosed} open at once")
        runtime.door_conflict = conflict


def _assign_groups(mech: MechanismITF) -> None:
    """Group the cabins linked through their destinations, one car per group, a group without a car gets it at its first cabin by name."""
    runtimes = mech._elevator_runtime
    parent = {name: name for name in runtimes}

    def find(name: str) -> str:
        while parent[name] != name:
            name = parent[name]
        return name

    for name, runtime in runtimes.items():
        for dest in runtime.destinations:
            if dest in parent:
                parent[find(dest)] = find(name)
    members: dict[str, list[str]] = {}
    for name in runtimes:
        members.setdefault(find(name), []).append(name)
    for names in members.values():
        names.sort()
        holders = [name for name in names if runtimes[name].car != _CarState.ABSENT]
        keep = holders[0] if holders else names[0]
        for name in names:
            runtime = runtimes[name]
            runtime.group = tuple(names)
            runtime.home = name == names[0]
            if name != keep:
                runtime.car = _CarState.ABSENT
        if not holders:
            runtimes[keep].car = _CarState.PRESENT


def _release_failed_riders(mech: MechanismITF, source_name: str, dest_name: str, names: Sequence[str]) -> None:
    """Unseal the source cabin and stop holding the destination door for riders whose teleport failed."""
    source = mech._elevator_runtime.get(source_name)
    dest = mech._elevator_runtime.get(dest_name)
    for name in names:
        if source is not None:
            source.dispatched.discard(name)
        if dest is not None:
            dest.just_arrived.pop(name, None)


def _call_walls(mech: MechanismITF, key: str, segments: list[Segment] | None) -> None:
    """Forward a wall set to the attached human simulator (no-op without one)."""
    hs = mech._human_simulator
    if hs is None:
        return
    try:
        hs.set_mechanism_walls(key, segments)
    except Exception as e:
        mech.node.get_logger().warning(f"mechanism shim: set_mechanism_walls({key!r}) failed: {e!r}")


def _door_blocks_peds(mech: MechanismITF, name: str, runtime: _DoorRuntime, now: float) -> bool:
    """True when the door cannot open for pedestrians right now."""
    if mech._semantics.door_driven(name):
        return False
    if max(runtime.door.activation_distance) <= 0.0:
        return True
    return not mech._semantics.trigger_allowed(name, now)


def _update_ped_walls(mech: MechanismITF, now: float, elevator_door_names: set[str]) -> None:
    """Push `door:<name>` ped walls to the human simulator on transitions only."""
    for name, runtime in mech._door_runtime.items():
        if name in elevator_door_names:
            continue
        wall = _door_blocks_peds(mech, name, runtime, now)
        if wall == runtime.ped_wall:
            continue
        runtime.ped_wall = wall
        door = runtime.door
        _call_walls(mech, f"door:{name}", [((door.start.x, door.start.y), (door.end.x, door.end.y))] if wall else None)


async def shim_spawn_doors(mech: MechanismITF, doors: Sequence[Door]) -> bool:
    """Spawn box geometry for each door and register door runtimes."""
    ok = True
    logger = mech.node.get_logger()
    for door in doors:
        kind = _effective_kind(logger, door)
        size, closed_pose = _leaf_geometry(door, kind)
        runtime = _DoorRuntime(
            door=door,
            closed_pose=closed_pose,
            open_pose=_door_open_pose(door, closed_pose, kind),
            effective_kind=kind,
        )
        spawned = [leaf for leaf, pose in _leaf_poses(door.name, runtime).items() if await mech.spawn_box(leaf, size, pose, door.material)]
        if len(spawned) == len(_leaf_names(door.name, kind)):
            mech._door_primitives[door.name] = spawned
            mech._door_runtime[door.name] = runtime
        else:
            for leaf in spawned:
                await mech.delete_box(leaf)
            ok = False
    if mech._door_runtime:
        _ensure_loop(mech)
    return ok


async def shim_remove_doors(mech: MechanismITF, names: Sequence[str]) -> bool:
    """Delete box geometry and drop door runtimes for the given names."""
    ok = True
    for name in names:
        for prim_name in mech._door_primitives.pop(name, []):
            if not await mech.delete_box(prim_name):
                ok = False
        runtime = mech._door_runtime.pop(name, None)
        if runtime is not None and runtime.ped_wall:
            _call_walls(mech, f"door:{name}", None)
    return ok


async def shim_spawn_elevators(mech: MechanismITF, elevators: Sequence[Elevator]) -> bool:
    """Spawn wall geometry and synthesized doors for each elevator."""
    ok = True
    synthesized_doors: list[Door] = []
    for elevator in elevators:
        spawned: list[str] = []
        for suffix, size, pose in _elevator_wall_geometries(elevator):
            name = f'{elevator.name}/{suffix}'
            if await mech.spawn_box(name, size, pose):
                spawned.append(name)
            else:
                ok = False
        mech._elevator_primitives[elevator.name] = spawned
        door = _elevator_synthesized_door(elevator)
        synthesized_doors.append(door)
        mech._elevator_doors[elevator.name] = door.name
        mech._elevator_runtime[elevator.name] = _ElevatorRuntime(
            elevator=elevator,
            door_name=door.name,
            destination_name=','.join(elevator.destinations),
            car=_CarState.ABSENT,
        )
        _call_walls(mech, f"elevator:{elevator.name}", _elevator_wall_segments(elevator))
    _assign_groups(mech)
    if synthesized_doors and not await shim_spawn_doors(mech, synthesized_doors):
        ok = False
    return ok


async def shim_remove_elevators(mech: MechanismITF, names: Sequence[str]) -> bool:
    """Delete elevator wall geometry and synthesized doors for the given names."""
    ok = True
    door_names: list[str] = []
    for name in names:
        for prim_name in mech._elevator_primitives.pop(name, []):
            if not await mech.delete_box(prim_name):
                ok = False
        door_name = mech._elevator_doors.pop(name, None)
        if door_name is not None:
            door_names.append(door_name)
        if mech._elevator_runtime.pop(name, None) is not None:
            _call_walls(mech, f"elevator:{name}", None)
    _assign_groups(mech)
    if door_names and not await shim_remove_doors(mech, door_names):
        ok = False
    return ok


def _tread_name(stair: Stairs, index: int) -> str:
    return f'{stair.name}/tread_{index}'


def stair_body(stair: Stairs) -> Stairs | None:
    """The geometry a half spawns: itself, the whole stair for the bottom half of a joined pair, none for its top half."""
    if not stair.joined or stair.whole:
        return stair
    return None if stair.top else stair.whole_stair()


class StairsPrim(typing.NamedTuple):
    name: str
    material: MaterialIdentifier
    box: StairsBox
    velocity: tuple[float, float, float] | None = None
    comb: bool = False


def stair_boxes(stair: Stairs) -> list[StairsPrim]:
    """Box primitives of a stair half, the velocity None on static boxes, the landing plates combs."""
    body = stair_body(stair)
    if body is None:
        return []
    if body is not stair:
        return [prim._replace(name=prim.name.replace(body.name, stair.name, 1)) for prim in stair_boxes(body)]
    static = [StairsPrim(f'{stair.name}/{name}', material, box, comb=name.startswith('plate_')) for name, material, box in stair.static_boxes()]
    treads = [StairsPrim(_tread_name(stair, k), stair.material_of('steps'), box, velocity) for k, (box, velocity) in enumerate(zip(stair.treads(0.0), stair.tread_velocities(0.0), strict=True))] if stair.kind == 'escalator' else []
    return [*static, *treads]


def stair_models(stair: Stairs) -> tuple[list[Obstacle], dict[int, Obstacle]]:
    """Objects dressing a stair half: those that stay put, and per tread index the one riding that tread."""
    body = stair_body(stair)
    if body is None:
        return [], {}
    still = [Obstacle(name=f'{stair.name}/{name}', model=model, pose=pose) for name, model, pose in body.static_models()]
    riding = {index: Obstacle(name=f'{stair.name}/model_tread_{index}', model=body.surface_models['steps'], pose=pose) for index, pose in enumerate(body.tread_models(0.0))}
    return still, riding


def stair_level(stair: Stairs) -> str:
    """The level of a realized stair half, the last part of its `<prefix>/<name>/<level_id>` name."""
    return stair.name.rsplit('/', 1)[-1]


def stair_partner(mech: MechanismITF, runtime: _StairsRuntime) -> _StairsRuntime | None:
    return mech._stair_runtime.get(runtime.partner) if runtime.partner else None


def _pair_stairs(mech: MechanismITF) -> None:
    """Link every half to the other half of its pair: its `to`, else the half whose `to` names it."""
    runtimes = mech._stair_runtime
    boarders = {runtime.stair.to: name for name, runtime in runtimes.items() if runtime.stair.boards}
    for name, runtime in runtimes.items():
        partner = runtime.stair.to if runtime.stair.boards else boarders.get(name, '')
        runtime.partner = partner if partner in runtimes else ''


async def _teleport_robot(mech: MechanismITF, runtime: _StairsRuntime, partner: _StairsRuntime, sim_path: str) -> None:
    pose = mech.robot_pose(sim_path)
    if pose is None:
        _report_incident(mech, f"{sim_path}: stair teleport {runtime.stair.name!r} -> {partner.stair.name!r} failed: robot pose unknown")
        return
    roll, pitch, yaw = pose.orientation.to_euler()
    x, y, landing_yaw, dz = runtime.stair.landing(partner.stair, pose.position.x, pose.position.y, yaw, runtime.travel, partner.travel)
    target = Pose(position=Position(x, y, pose.position.z + dz), orientation=Orientation.from_euler((roll, pitch, landing_yaw)))
    try:
        failure = "" if await mech.set_robot_pose(sim_path, target) else "the simulator refused the move"
    except Exception as e:
        failure = repr(e)
    if failure:
        _report_incident(mech, f"{sim_path}: stair teleport {runtime.stair.name!r} -> {partner.stair.name!r} failed: {failure}")
    else:
        runtime.just_crossed = True


async def _tick_stairs(mech: MechanismITF, dt: float, robot_discs: Sequence[Disc], ped_discs: Sequence[Disc]) -> None:
    """Track who stands on each half, teleport robots across boarding seams, run escalator belts."""
    runtimes = mech._stair_runtime
    if not runtimes:
        return
    robots = {name: xy for name, xy, _radius in robot_discs}
    peds = {name: xy for name, xy, _radius in ped_discs}
    previous = {name: runtime.peds for name, runtime in runtimes.items()}
    for runtime in runtimes.values():
        stair = runtime.stair
        runtime.just_crossed = False
        runtime.peds = frozenset(name for name, (x, y) in peds.items() if stair.contains(x, y))
        runtime.members = tuple(sorted(name for name, (x, y) in robots.items() if stair.contains(x, y))) + tuple(sorted(runtime.peds))
        runtime.latched = {name for name in runtime.latched if name not in robots or stair.crossed(*robots[name])}

    jobs: list[tuple[_StairsRuntime, _StairsRuntime, str]] = []
    level_changes: list[tuple[str, str]] = []
    for name, runtime in runtimes.items():
        partner = stair_partner(mech, runtime)
        if partner is not None and runtime.stair.joined:
            if previous[name] & partner.peds:
                runtime.just_crossed = True
            for sim_path, xy in robots.items():
                if runtime.stair.past_seam(*xy):
                    level_changes.append((sim_path, stair_level(partner.stair)))
                    runtime.just_crossed = True
            continue
        if partner is None or not runtime.stair.boards:
            continue
        if previous[name] & partner.peds:
            runtime.just_crossed = True
        for sim_path, xy in robots.items():
            if sim_path in runtime.latched or not runtime.stair.crossed(*xy):
                continue
            runtime.latched.add(sim_path)
            partner.latched.add(sim_path)
            jobs.append((runtime, partner, sim_path))
    for runtime, partner, sim_path in jobs:
        await _teleport_robot(mech, runtime, partner, sim_path)
    for sim_path, level in level_changes:
        await mech.robot_changed_level(sim_path, level)

    moves: dict[str, Pose] = {}
    velocities: dict[str, tuple[float, float, float]] = {}
    for runtime in runtimes.values():
        stair = runtime.body
        if not runtime.treads or stair is None:
            continue
        if runtime.running:
            runtime.travel += stair.belt * dt
        wanted = stair.tread_velocities(runtime.travel)
        for index, ((_size, pose), velocity) in enumerate(zip(stair.treads(runtime.travel), wanted, strict=True)):
            name = runtime.treads.get(index)
            if name is None:
                continue
            moves[name] = pose
            velocity = velocity if runtime.running else (0.0, 0.0, 0.0)
            if runtime.velocities.get(name) != velocity:
                runtime.velocities[name] = velocity
                velocities[name] = velocity
    if velocities:
        await mech.set_box_velocities(velocities)
    if moves:
        await mech.move_boxes(moves)
    riders = [attrs.evolve(model, pose=pose) for runtime in runtimes.values() if runtime.tread_models and runtime.body is not None for index, pose in enumerate(runtime.body.tread_models(runtime.travel)) if (model := runtime.tread_models.get(index)) is not None]
    if riders:
        await mech.obstacle_move(riders)

    if mech._human_simulator is not None:
        changes = {name: runtime.running for name, runtime in runtimes.items() if runtime.running != runtime.reported_running}
        if changes:
            for name, running in changes.items():
                runtimes[name].reported_running = running
            await mech._human_simulator.stairs_changed(changes)


async def shim_spawn_stairs(mech: MechanismITF, stairs: Sequence[Stairs]) -> bool:
    """Spawn the box primitives of each stair half and register stair runtimes."""
    plan = {stair.name: (stair, stair_boxes(stair)) for stair in stairs}
    jobs = [prim for _stair, prims in plan.values() for prim in prims]

    async def spawn(prim: StairsPrim) -> bool:
        size, pose = prim.box
        return await mech.spawn_box(prim.name, size, pose, prim.material, velocity=prim.velocity, comb=prim.comb)

    results = await asyncio.gather(*(spawn(prim) for prim in jobs))
    spawned = {prim.name for prim, ok in zip(jobs, results, strict=True) if ok}
    dressing = {stair.name: stair_models(stair) for stair in stairs}
    objects = [model for still, riding in dressing.values() for model in (*still, *riding.values())]
    shown = {model.name for model, ok in zip(objects, await mech.obstacle_spawn(objects) if objects else (), strict=True) if ok}
    for stair, prims in plan.values():
        mech._stair_primitives[stair.name] = [prim.name for prim in prims if prim.name in spawned]
        still, riding = dressing[stair.name]
        mech._stair_models[stair.name] = [model for model in (*still, *riding.values()) if model.name in shown]
        body = stair_body(stair)
        treads = {index: _tread_name(stair, index) for index in range(body.tread_count)} if body is not None and stair.kind == 'escalator' else {}
        live = {index: name for index, name in treads.items() if name in spawned}
        initial = body.tread_velocities(0.0) if body is not None and live else []
        mech._stair_runtime[stair.name] = _StairsRuntime(stair=stair, body=body, treads=live, velocities={name: initial[index] for index, name in live.items()}, tread_models={index: model for index, model in riding.items() if model.name in shown})
    _pair_stairs(mech)
    for stair in stairs:
        if stair.boards and not mech._stair_runtime[stair.name].partner:
            mech.node.get_logger().info(f"Stairs {stair.name!r}: partner half {stair.to!r} not spawned, its seam lands nowhere")
    if mech._stair_runtime:
        _ensure_loop(mech)
    return all(results)


async def shim_remove_stairs(mech: MechanismITF, names: Sequence[str]) -> bool:
    """Delete the box primitives and dressing objects and drop the stair runtimes of the given names."""
    ok = True
    for name in names:
        for prim_name in mech._stair_primitives.pop(name, []):
            if not await mech.delete_box(prim_name):
                ok = False
        models = mech._stair_models.pop(name, [])
        if models and not all(await mech.obstacle_delete(models)):
            ok = False
        mech._stair_runtime.pop(name, None)
    _pair_stairs(mech)
    return ok
