"""Shim implementations of MechanismITF for simulators without native door/elevator support."""

from __future__ import annotations

import asyncio
import enum
import math
import typing
from collections.abc import Sequence

import attrs
import rclpy.impl.rcutils_logger
from task_generator.shared import Door, Elevator, Orientation, Pose, Position

from ._interface import _BOX_FLOOR_CLEARANCE

if typing.TYPE_CHECKING:
    from ._interface import MechanismITF

MECHANISM_TICK_RATE = 30.0  # Hz, sim time
DOOR_INSET = 0.05
WALL_THICKNESS = 0.05

Disc = tuple[str, tuple[float, float], float]

_DOOR_AXES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    '+x': ((1.0, 0.0), (0.0, 1.0)),
    '-x': ((-1.0, 0.0), (0.0, 1.0)),
    '+y': ((0.0, 1.0), (1.0, 0.0)),
    '-y': ((0.0, -1.0), (1.0, 0.0)),
}


def _door_basis(elevator: Elevator) -> tuple[float, float, tuple[float, float], tuple[float, float], float]:
    outward, tangent = _DOOR_AXES[elevator.door_side]
    hx, hy = elevator.size[0] / 2.0, elevator.size[1] / 2.0
    out_extent = hx if outward[0] != 0 else hy
    tan_extent = hy if outward[0] != 0 else hx
    face_cx = elevator.position.x + outward[0] * out_extent
    face_cy = elevator.position.y + outward[1] * out_extent
    return face_cx, face_cy, outward, tangent, tan_extent


def _door_slot(elevator: Elevator) -> tuple[Position, Position]:
    face_cx, face_cy, outward, tangent, tan_extent = _door_basis(elevator)
    slot_half = tan_extent - DOOR_INSET
    cx = face_cx - outward[0] * DOOR_INSET
    cy = face_cy - outward[1] * DOOR_INSET
    z = elevator.position.z
    return (
        Position(cx - tangent[0] * slot_half, cy - tangent[1] * slot_half, z),
        Position(cx + tangent[0] * slot_half, cy + tangent[1] * slot_half, z),
    )


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


@attrs.define
class _DoorRuntime:
    door: Door
    closed_pose: Pose
    open_pose: Pose
    effective_kind: typing.Literal['sliding', 'teleport', 'sliding_top']
    state: _DoorState = _DoorState.CLOSED
    progress: float = 0.0  # 0 = closed, 1 = open
    last_trigger_sim_time: float = -math.inf
    last_applied_progress: float = -1.0  # forces first move_box call


@attrs.define
class _ElevatorRuntime:
    elevator: Elevator
    door_name: str
    destination_name: str
    # Scheduled teleport: ETA when pending_occupants arrive at destination.
    arriving_eta: float = -math.inf
    # Occupants staged for teleport at arriving_eta.
    pending_occupants: tuple[tuple[str, tuple[float, float]], ...] = ()
    # Tracks just-teleported occupants; value = whether inside-confirmed post-teleport.
    just_arrived: dict[str, bool] = attrs.field(factory=dict)
    # True while door is in the process of closing to dispatch a teleport.
    departing: bool = False
    # Occupants already dispatched this leg, still in the cabin until their teleport fires.
    dispatched: set[str] = attrs.field(factory=set)


@attrs.define
class _ElevatorStepResult:
    teleport_job: tuple[str, str, list[tuple[str, tuple[float, float]]]] | None = None
    missing_destination: bool = False


def _reset_door(runtime: _DoorRuntime) -> None:
    """Reset a door runtime to spawn defaults (last_applied_progress = -1 forces a re-close next tick)."""
    runtime.state = _DoorState.CLOSED
    runtime.progress = 0.0
    runtime.last_trigger_sim_time = -math.inf
    runtime.last_applied_progress = -1.0


def reset_mechanisms(mech: MechanismITF) -> None:
    """Reset every door and elevator runtime to spawn defaults for a deterministic episode start."""
    for runtime in mech._door_runtime.values():
        _reset_door(runtime)
    for name, runtime in mech._elevator_runtime.items():
        runtime.arriving_eta = -math.inf
        runtime.pending_occupants = ()
        runtime.just_arrived = {}
        runtime.departing = False
        runtime.dispatched = set()
        cabin = mech._door_runtime.get(f"{name}/door")
        if cabin is not None:
            _reset_door(cabin)


def _effective_kind(
    logger: rclpy.impl.rcutils_logger.RcutilsLogger,
    door: Door,
) -> typing.Literal['sliding', 'teleport', 'sliding_top']:
    """Return the animation kind, falling back to teleport for hinged with a warn-once log."""
    if door.kind == 'hinged':
        logger.warning(
            f"mechanism shim: door {door.name!r} kind='hinged' not implemented. Falling back to 'teleport'.",
        )
        return 'teleport'
    if door.kind == 'teleport':
        return 'teleport'
    if door.kind == 'sliding_top':
        return 'sliding_top'
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


def _door_open_pose(door: Door, closed_pose: Pose, effective_kind: str) -> Pose:
    """Return the fully-open pose for a door (Z-drop for teleport, axis-slide for sliding)."""
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
    return Pose(
        position=Position(
            x=closed_pose.position.x + ux * length,
            y=closed_pose.position.y + uy * length,
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
        kind='sliding',
        width=0.05,
        height=elevator.size[2],
        activation_distance=(elevator.activation_distance, elevator.activation_distance),
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


def _swept_slab_blocked(runtime: _DoorRuntime, p_from: float, p_to: float, discs: Sequence[Disc]) -> bool:
    """True when the slab band swept between the two progress values overlaps any agent disc.
    Vertical kinds (teleport, sliding_top) occupy the slot itself regardless of progress."""
    door = runtime.door
    sx, sy = door.start.x, door.start.y
    ex, ey = door.end.x, door.end.y
    length = math.hypot(ex - sx, ey - sy)
    if length <= 0.0 or not discs:
        return False
    ux, uy = (ex - sx) / length, (ey - sy) / length
    lo, hi = (0.0, 0.0) if runtime.effective_kind != 'sliding' else (min(p_from, p_to) * length, max(p_from, p_to) * length)
    cx = (sx + ex) / 2.0 + ux * (lo + hi) / 2.0
    cy = (sy + ey) / 2.0 + uy * (lo + hi) / 2.0
    half_l = length / 2.0 + (hi - lo) / 2.0
    half_w = door.width / 2.0
    for _name, (x, y), radius in discs:
        dx, dy = x - cx, y - cy
        along = abs(dx * ux + dy * uy)
        across = abs(-dx * uy + dy * ux)
        if math.hypot(max(along - half_l, 0.0), max(across - half_w, 0.0)) <= radius:
            return True
    return False


def _advance_state(runtime: _DoorRuntime, dt: float, now: float, discs: Sequence[Disc]) -> None:
    """Advance door state machine one tick: linear T-delta for sliding, instant snap for teleport.
    The slab never moves through an agent: a blocked close re-triggers the door (it reverses,
    tested against the full remaining path so nobody in the slot is approached), a blocked
    opening step holds in place. Vertical kinds open away from agents and are never open-blocked."""
    door = runtime.door
    fresh = (now - runtime.last_trigger_sim_time) <= door.hold_time
    if runtime.effective_kind == 'teleport':
        if not fresh and _swept_slab_blocked(runtime, runtime.progress, 0.0, discs):
            runtime.last_trigger_sim_time = now
            fresh = True
        runtime.progress = 1.0 if fresh else 0.0
        runtime.state = _DoorState.OPEN if fresh else _DoorState.CLOSED
        return
    # sliding (incl. sliding_top): per-tick linear T delta over transition_time
    target = 1.0 if fresh else 0.0
    step = dt / max(door.transition_time, 1e-9)
    if target > runtime.progress:
        runtime.state = _DoorState.OPENING
        next_progress = min(1.0, runtime.progress + step)
        if runtime.effective_kind == 'sliding' and _swept_slab_blocked(runtime, runtime.progress, next_progress, discs):
            return
        runtime.progress = next_progress
        if runtime.progress >= 1.0:
            runtime.state = _DoorState.OPEN
    elif target < runtime.progress:
        if _swept_slab_blocked(runtime, 0.0, runtime.progress, discs):
            runtime.last_trigger_sim_time = now
            runtime.state = _DoorState.OPEN if runtime.progress >= 1.0 else _DoorState.OPENING
            return
        runtime.state = _DoorState.CLOSING
        runtime.progress = max(0.0, runtime.progress - step)
        if runtime.progress <= 0.0:
            runtime.state = _DoorState.CLOSED
    else:
        runtime.state = _DoorState.OPEN if target == 1.0 else _DoorState.CLOSED


def _interp_pose(runtime: _DoorRuntime) -> Pose:
    """Linear interpolation between closed and open pose at runtime.progress."""
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
) -> _ElevatorStepResult:
    result = _ElevatorStepResult()

    # Update just_arrived tracking: confirm inside observation, clear on real exit.
    current = frozenset(name for name, _ in occupants)
    for name in list(runtime.just_arrived):
        if name in current:
            runtime.just_arrived[name] = True
        elif runtime.just_arrived[name] and name in outside_names:
            del runtime.just_arrived[name]

    # Drop dispatched occupants once they have left the cabin (their teleport has fired).
    runtime.dispatched &= current

    # Check for a scheduled arrival.
    if runtime.arriving_eta > -math.inf and now >= runtime.arriving_eta:
        runtime.arriving_eta = -math.inf
        door_runtime.last_trigger_sim_time = now
        if runtime.pending_occupants:
            src_name = runtime.destination_name
            result.teleport_job = (src_name, runtime.elevator.name, list(runtime.pending_occupants))
            runtime.just_arrived = {name: False for name, _ in runtime.pending_occupants}
            runtime.pending_occupants = ()
        return result

    # While a teleport is in transit, suppress all door triggers.
    if runtime.arriving_eta > -math.inf:
        return result

    # Our dispatched rider is mid-flight: the cabin is away, keep the door shut until they land.
    if runtime.dispatched:
        return result

    # Outside trigger opens the door if this elevator accepts outside calls.
    if outside_trigger and runtime.elevator.accept_outside_calls:
        door_runtime.last_trigger_sim_time = now
        runtime.departing = False
        return result

    # Track new (non-just-arrived) occupants in the cabin.
    new_occupants = current - runtime.just_arrived.keys()

    # Departing phase: door closing with occupants, slab gate holds it open while blocked
    if runtime.departing:
        if door_runtime.state == _DoorState.CLOSED:
            if dest_runtime is None:
                result.missing_destination = True
                runtime.departing = False
                door_runtime.last_trigger_sim_time = now
                return result
            dest_runtime.arriving_eta = now + runtime.elevator.travel_time
            dest_runtime.pending_occupants = tuple(occupants)
            runtime.dispatched = {name for name, _ in occupants}
            runtime.departing = False
            runtime.just_arrived = {}
            return result
        # Door still closing: hold off.
        return result

    # Hold door open for just-arrived occupants (post-teleport)
    if runtime.just_arrived:
        door_runtime.last_trigger_sim_time = now
        return result

    # No new occupants: let the door close naturally (do not refresh trigger).
    if not new_occupants:
        return result

    # New occupant present, door has fully closed: start departing.
    if door_runtime.state == _DoorState.CLOSED:
        runtime.departing = True
        return result

    # Door is still open or closing: do not refresh, let it close.
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


async def _tick(mech: MechanismITF, dt: float) -> None:
    """Single tick: update triggers, advance state machines, dispatch move_box and teleports."""
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
        outside_trigger = _is_triggered(door_runtime.door, outside_xys)
        dest_runtime = mech._elevator_runtime.get(runtime.destination_name)
        if mech._semantics.elevator_recalled(elev_name, now):
            door_runtime.last_trigger_sim_time = now  # recall: hold cabin door open
            runtime.departing = False
            continue
        result = _step_elevator(runtime, door_runtime, dest_runtime, occupants, outside_trigger, now, outside_names=outside_names)
        if result.missing_destination:
            logger.warning(f"Elevator {elev_name!r}: destination {runtime.destination_name!r} unknown; door held open.")
        if result.teleport_job is not None:
            teleport_jobs.append(result.teleport_job)

    # Regular doors take proximity from the full agent list. Elevator doors are gated above.
    for name, runtime in mech._door_runtime.items():
        if name in elevator_door_names:
            continue
        if _is_triggered(runtime.door, all_xys) and mech._semantics.trigger_allowed(name, now):
            runtime.last_trigger_sim_time = now

    mech._semantics.apply_plate_drives(now)

    pending: list[typing.Awaitable] = []
    for name, runtime in mech._door_runtime.items():
        _advance_state(runtime, dt, now, discs)
        if runtime.progress != runtime.last_applied_progress:
            runtime.last_applied_progress = runtime.progress
            pending.append(mech.move_box(name, _interp_pose(runtime)))
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    # Run teleport jobs after door updates are dispatched.
    for source_name, dest_name, named_occupants in teleport_jobs:
        destinations = _compute_teleport_destinations(mech._elevator_runtime, source_name, dest_name, named_occupants)
        if not destinations:
            continue
        robot_destinations = {k: v for k, v in destinations.items() if k not in ped_names}
        ped_destinations = {k: v for k, v in destinations.items() if k in ped_names}
        for sim_path, (x, y) in robot_destinations.items():
            current = mech.robot_pose(sim_path)
            z = current.position.z if current is not None else 0.0
            orientation = current.orientation if current is not None else Orientation(1, 0, 0, 0)
            try:
                await mech.set_robot_pose(sim_path, Pose(position=Position(x, y, z), orientation=orientation))
            except Exception as e:
                logger.warning(f"Elevator robot teleport {source_name!r} -> {dest_name!r} failed for {sim_path!r}: {e!r}")
        if ped_destinations:
            if mech._human_simulator is not None:
                try:
                    await mech._human_simulator.pedestrian_teleport(ped_destinations)
                except Exception as e:
                    logger.warning(f"Elevator ped teleport {source_name!r} -> {dest_name!r} failed: {e!r}")
            else:
                logger.info(f"Elevator teleport (no-op, no human sim): {len(ped_destinations)} peds {source_name!r} -> {dest_name!r}")

    mech._semantics.step(now)


async def shim_spawn_doors(mech: MechanismITF, doors: Sequence[Door]) -> bool:
    """Spawn box geometry for each door and register door runtimes."""
    ok = True
    logger = mech.node.get_logger()
    for door in doors:
        kind = _effective_kind(logger, door)
        size, closed_pose = _door_geometry(door)
        open_pose = _door_open_pose(door, closed_pose, kind)
        if await mech.spawn_box(door.name, size, closed_pose):
            mech._door_primitives[door.name] = [door.name]
            mech._door_runtime[door.name] = _DoorRuntime(
                door=door,
                closed_pose=closed_pose,
                open_pose=open_pose,
                effective_kind=kind,
            )
        else:
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
        mech._door_runtime.pop(name, None)
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
            destination_name=elevator.destination,
        )
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
        mech._elevator_runtime.pop(name, None)
    if door_names and not await shim_remove_doors(mech, door_names):
        ok = False
    return ok
