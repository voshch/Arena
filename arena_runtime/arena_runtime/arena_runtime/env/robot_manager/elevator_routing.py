"""Expand GoTo phases into elevator rides and stair crossings: explicit `via` hops plus automatic cross-level routing."""

from __future__ import annotations

import math
import typing

import attrs

from arena_runtime.env.request import ElevatorLegPhase, ElevatorRidePhase, GoToPhase, LoopPhase, StairsLegPhase, StairsRidePhase
from arena_runtime.env.world_manager.world_manager import ElevatorHop
from arena_runtime.shared import Orientation, Pose, Position

if typing.TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from arena_runtime.env.request import TaskPhase
    from arena_runtime.env.world_manager.world_manager import ElevatorRide, LevelHop, StairsRide, WorldManager

VIA_DESTINATION_SEP = '>'


def parse_via(entry: str) -> tuple[str, str]:
    """`"<elevator or stair>"` or `"<elevator or stair>><destination>"` -> (name, destination or '')."""
    name, _, to = entry.partition(VIA_DESTINATION_SEP)
    return name.strip(), to.strip()


MIN_RIDE_TOLERANCE = 0.05


def ride_tolerance(boarding_radius: float, goal_tolerance: float) -> float:
    """Ride arrival tolerance: the goal tolerance capped by the boarding radius, `MIN_RIDE_TOLERANCE` when the robot barely fits."""
    return min(goal_tolerance, boarding_radius) if boarding_radius > 0.0 else MIN_RIDE_TOLERANCE


def ride_phase(ride: ElevatorRide, orientation: Orientation, goal_tolerance: float) -> ElevatorRidePhase:
    """Ride phase for one hop: arrive in the boarding cabin, complete once the robot is in the destination cabin."""
    hop = ride.hop
    return ElevatorRidePhase(
        pose=Pose(position=ride.board_target, orientation=orientation),
        target=hop.board_name,
        tolerance_radius=ride_tolerance(ride.boarding_radius, goal_tolerance),
        tolerance_angle=math.pi,
        hold_time=0.0,
        until=None if hop.dest_position is None else f"robot in {hop.dest_name}",
        destination=hop.target,
        exit_cabin="" if hop.dest_position is None else hop.dest_name,
    )


LEG_TOLERANCE = 1.6
EXIT_CLEARANCE = 0.1


def exit_tolerance(door_standoff: float) -> float:
    """Exit leg tolerance, met only with the robot center `EXIT_CLEARANCE` past the door face."""
    return min(LEG_TOLERANCE, door_standoff - EXIT_CLEARANCE)


def ride_phases(ride: ElevatorRide, orientation: Orientation, goal_tolerance: float) -> list[GoToPhase]:
    """Door-aligned approach leg, the ride, and a door-aligned exit leg out of the cabin (exit omitted when unknown)."""
    phases: list[GoToPhase] = [
        ElevatorLegPhase(pose=Pose(position=ride.board_approach, orientation=orientation), tolerance_radius=LEG_TOLERANCE, tolerance_angle=math.pi, hold_time=0.0),
        ride_phase(ride, orientation, goal_tolerance),
    ]
    if ride.dest_exit is not None:
        phases.append(ElevatorLegPhase(pose=Pose(position=ride.dest_exit, orientation=orientation), tolerance_radius=exit_tolerance(ride.door_standoff), tolerance_angle=math.pi, hold_time=0.0))
    return phases


STAIRS_ALIGN_TOLERANCE = 0.3
STAIRS_APPROACH_TOLERANCE = 0.25


def crossing_tolerance(overshoot: float) -> float:
    """Crossing tolerance, half the overshoot past the trigger line."""
    return overshoot / 2.0


def crossing_phases(ride: StairsRide, goal_tolerance: float) -> list[GoToPhase]:
    """Axis-aligned approach leg, the drive past the trigger line completing on the partner half, and an exit leg out of the partner's mouth."""
    hop = ride.hop
    board = Orientation.from_yaw(ride.board_yaw)
    return [
        StairsLegPhase(pose=Pose(position=ride.approach, orientation=board), tolerance_radius=min(goal_tolerance, STAIRS_APPROACH_TOLERANCE), tolerance_angle=STAIRS_ALIGN_TOLERANCE, hold_time=0.0),
        StairsRidePhase(
            pose=Pose(position=ride.board_target, orientation=board),
            target=hop.dest_name,
            tolerance_radius=crossing_tolerance(ride.overshoot),
            tolerance_angle=math.pi,
            hold_time=0.0,
        ),
        StairsLegPhase(pose=Pose(position=ride.dest_exit, orientation=Orientation.from_yaw(ride.exit_yaw)), tolerance_radius=exit_tolerance(ride.mouth_standoff), tolerance_angle=math.pi, hold_time=0.0),
    ]


def _via_toward(world_manager: WorldManager, vias: Sequence[str], index: int, goal: Position) -> Position:
    """Where the robot heads after hop `index`: the next via boarding point, else the goal."""
    if index + 1 < len(vias):
        position = world_manager.via_position(parse_via(vias[index + 1])[0])
        if position is not None:
            return position
    return goal


def route_phases(
    phases: Sequence[TaskPhase],
    start: Position | None,
    world_manager: WorldManager,
    *,
    robot_radius: float,
    goal_tolerance: float,
    climbs: Collection[str] = (),
    walks: Collection[str] = (),
) -> list[TaskPhase]:
    """Insert elevator rides and crossings of the stair kinds in `climbs` for a robot disc of `robot_radius` before GoTo phases in the abstract frame: explicit `via` hops, then the fewest-hop route to the goal level. Stair kinds in `walks` route without phases, the robot's planner crosses them."""
    current_level = world_manager.level_of_point(start.x, start.y) if start is not None else ""
    walked = frozenset(walks)
    climbed = frozenset(climbs) | walked

    def rides(hops: Sequence[LevelHop], orientation: Orientation) -> list[GoToPhase]:
        routed: list[GoToPhase] = []
        for hop in hops:
            if isinstance(hop, ElevatorHop):
                routed.extend(ride_phases(world_manager.elevator_ride(hop, robot_radius), orientation, goal_tolerance))
            elif hop.kind not in walked:
                routed.extend(crossing_phases(world_manager.stair_ride(hop, robot_radius), goal_tolerance))
        return routed

    def route(phases: Sequence[TaskPhase]) -> list[TaskPhase]:
        nonlocal current_level
        routed: list[TaskPhase] = []
        for phase in phases:
            if isinstance(phase, LoopPhase):
                routed.append(attrs.evolve(phase, phases=route(phase.phases)))
                continue
            if not isinstance(phase, GoToPhase) or isinstance(phase, (ElevatorRidePhase, ElevatorLegPhase, StairsRidePhase, StairsLegPhase)) or phase.pose is None:
                routed.append(phase)
                continue
            goal = phase.pose.position
            orientation = phase.pose.orientation
            for index, entry in enumerate(phase.via):
                name, to = parse_via(entry)
                hop = world_manager.via_hop(name, to=to, toward=_via_toward(world_manager, phase.via, index, goal), climbs=climbed)
                if current_level and hop.board_level != current_level:
                    routed.extend(rides(world_manager.level_hops(current_level, hop.board_level, climbed), orientation))
                routed.extend(rides([hop], orientation))
                current_level = hop.dest_level
            goal_level = world_manager.level_of_point(goal.x, goal.y)
            if current_level and goal_level and goal_level != current_level:
                routed.extend(rides(world_manager.level_hops(current_level, goal_level, climbed), orientation))
            if goal_level:
                current_level = goal_level
            routed.append(attrs.evolve(phase, via=()) if phase.via else phase)
        return routed

    return route(phases)
