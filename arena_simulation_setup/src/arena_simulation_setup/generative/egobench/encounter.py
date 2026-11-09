"""One pedestrian placed against a straight robot leg by encounter type, and the scenario that plays it on a fixed or a reactive route."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path

import attrs
import shapely

from arena_simulation_setup.generative.scenario.promptgen.agents import STAND_STEP, WALK_PATIENCE_S, TypeFiles, mesh_for
from arena_simulation_setup.generative.scenario.promptgen.profiles import profile
from arena_simulation_setup.generative.scenario.promptgen.realize import GO, release_point, within
from arena_simulation_setup.generative.scenario.zone_screen import RUNTIME_SAFE_DIST
from arena_simulation_setup.shared.pathing import Point
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT, parse_scenario

from .manifest import Encounter, Pedestrian
from .sampler import Sample

ROBOT_RADIUS = 0.35
PED_RADIUS = RUNTIME_SAFE_DIST
CLEARANCE = 0.1  # free space kept between a body and a wall
LEAD_IN_M = 5.0  # walk before the encounter point
LEAD_OUT_M = 4.0  # walk after a crossing
GAP_M = 3.0  # head start of a followed or leading pedestrian
MIN_RELEASE_M = 0.5  # robot travel before a release, so the trigger never holds at the start
ENCOUNTER_AT = 0.5  # fraction of the robot's leg where the encounter is designed
PED_NAME = "ped_0"
FIXED_PLANNER = "straight"  # HumanSim local planner that ignores every neighbour

PARALLEL = (Encounter.ONCOMING, Encounter.OVERTAKE, Encounter.OVERTAKEN, Encounter.FOLLOW, Encounter.LEAD, Encounter.STANDING)
CROSSING = (Encounter.CROSSING_LEFT, Encounter.CROSSING_RIGHT)


@attrs.frozen
class Leg:
    """A straight stretch of free space the robot drives along."""

    start: Point
    end: Point

    @property
    def length(self) -> float:
        return math.dist(self.start, self.end)

    @property
    def direction(self) -> Point:
        return ((self.end[0] - self.start[0]) / self.length, (self.end[1] - self.start[1]) / self.length)

    @property
    def normal(self) -> Point:
        """Unit vector to the left of the direction."""
        ux, uy = self.direction
        return (-uy, ux)

    @property
    def heading(self) -> float:
        return math.atan2(self.direction[1], self.direction[0])

    def at(self, arc: float, lateral: float = 0.0) -> Point:
        """The point `arc` along the leg, shifted `lateral` to the left."""
        (ux, uy), (nx, ny) = self.direction, self.normal
        return (self.start[0] + ux * arc + nx * lateral, self.start[1] + uy * arc + ny * lateral)

    def shifted(self, lateral: float) -> Leg:
        return Leg(start=self.at(0.0, lateral), end=self.at(self.length, lateral))

    def sub(self, start: float, end: float) -> Leg:
        return Leg(start=self.at(start), end=self.at(end))


@attrs.frozen
class Rejected:
    reason: str


@attrs.frozen
class Placement:
    robot: Leg
    spawn: Point
    facing: float
    path: tuple[Point, ...]  # empty for a standing pedestrian
    meet: Point
    release: float | None  # robot arc that releases the pedestrian, None = never released

    def ped_route(self) -> list[Point]:
        return [self.spawn, *self.path]


@attrs.frozen
class Generated:
    scenario: dict[str, object]
    types: dict[str, dict[str, object]]
    provenance: dict[str, object]


def _along(p: Point, v: Point, distance: float) -> Point:
    return (p[0] + v[0] * distance, p[1] + v[1] * distance)


def _heading(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def walked(sample: Sample, robot: Leg, meet: Point, robot_at_meet: float, v: Point, path: Sequence[Point], *, lead_in: float = LEAD_IN_M) -> Placement | Rejected:
    """A pedestrian walking `v` into `meet` as the robot reaches `robot_at_meet`, released early enough to get there."""
    travel = lead_in / sample.ped_speed
    release = robot_at_meet - sample.robot_speed * travel
    if release < MIN_RELEASE_M:
        release = MIN_RELEASE_M
        travel = (robot_at_meet - release) / sample.robot_speed
        if travel <= 0:
            return Rejected(f"the robot reaches the encounter {robot_at_meet:.2f} m along its leg before any release")
    spawn = _along(meet, v, -sample.ped_speed * travel)
    return Placement(robot=robot, spawn=spawn, facing=_heading(spawn, meet), path=tuple(path), meet=meet, release=release)


def overtaking(sample: Sample, robot: Leg, lane: Leg, behind: float, meet_arc: float) -> Placement | Rejected:
    """A pedestrian starting `behind` the robot on its lane, released so it draws level at `meet_arc`, at the start when it cannot."""
    vp, vr = sample.ped_speed, sample.robot_speed
    release = max(0.0, (meet_arc * (vp - vr) - vr * behind) / vp)
    if release < MIN_RELEASE_M:
        release = 0.0
    level = release + vr * (behind + release) / (vp - vr)
    if level > robot.length:
        return Rejected(f"a pedestrian at {vp:.2f} m/s draws level with the robot only {level:.2f} m along a {robot.length:.2f} m leg")
    return Placement(robot=robot, spawn=lane.start, facing=robot.heading, path=(lane.end,), meet=lane.at(behind + level), release=release)


def parallel(sample: Sample, run: Leg) -> Placement | Rejected:
    """An encounter along `run`: robot and pedestrian lanes split by the lateral offset, `+` puts the pedestrian on the robot's left."""
    offset = sample.lateral_offset
    behind = GAP_M if sample.encounter in (Encounter.LEAD, Encounter.OVERTAKEN) else 0.0
    if run.length <= behind:
        return Rejected(f"a {run.length:.2f} m run leaves no room behind the robot")
    robot = run.sub(behind, run.length).shifted(-offset / 2)
    lane = run.shifted(offset / 2)
    u = robot.direction
    back = (-u[0], -u[1])
    meet_arc = ENCOUNTER_AT * robot.length
    meet = lane.at(behind + meet_arc)
    match sample.encounter:
        case Encounter.STANDING:
            return Placement(robot=robot, spawn=meet, facing=robot.heading + math.pi, path=(), meet=meet, release=None)
        case Encounter.ONCOMING:
            return walked(sample, robot, meet, meet_arc, back, [lane.start])
        case Encounter.OVERTAKE:
            if sample.ped_speed >= sample.robot_speed:
                return Rejected(f"a pedestrian at {sample.ped_speed:.2f} m/s is not overtaken by a robot at {sample.robot_speed:.2f} m/s")
            return walked(sample, robot, meet, meet_arc, u, [lane.end])
        case Encounter.OVERTAKEN:
            if sample.ped_speed <= sample.robot_speed:
                return Rejected(f"a pedestrian at {sample.ped_speed:.2f} m/s does not overtake a robot at {sample.robot_speed:.2f} m/s")
            return overtaking(sample, robot, lane, behind, meet_arc)
        case Encounter.FOLLOW:
            spawn = lane.at(GAP_M)
            return Placement(robot=robot, spawn=spawn, facing=robot.heading, path=(lane.end,), meet=spawn, release=0.0)
        case Encounter.LEAD:
            spawn = lane.at(0.0)
            return Placement(robot=robot, spawn=spawn, facing=robot.heading, path=(lane.end,), meet=spawn, release=0.0)
    return Rejected(f"{sample.encounter.value} is no encounter along a run")


def crossing_direction(sample: Sample, robot: Leg) -> Point:
    """Walking direction of a crossing pedestrian: from the robot's left walks right and back."""
    nx, ny = robot.normal
    return (-nx, -ny) if sample.encounter is Encounter.CROSSING_LEFT else (nx, ny)


def crossed(sample: Sample, robot: Leg, cross_arc: float, path: Sequence[Point], *, lead_in: float = LEAD_IN_M) -> Placement | Rejected:
    """A pedestrian crossing the robot's leg at `cross_arc` while the robot is still `lateral_offset` short of it."""
    gap = abs(sample.lateral_offset)
    if cross_arc - gap < MIN_RELEASE_M:
        return Rejected(f"a crossing {cross_arc:.2f} m along the leg leaves no room for a {gap:.2f} m lead")
    return walked(sample, robot, robot.at(cross_arc), cross_arc - gap, crossing_direction(sample, robot), path, lead_in=lead_in)


def fits(placement: Placement, free: shapely.geometry.base.BaseGeometry) -> str | None:
    """Why the robot's leg or the pedestrian's route leaves the free space, None when both stay inside."""
    robot = shapely.LineString([placement.robot.start, placement.robot.end]).buffer(ROBOT_RADIUS + CLEARANCE)
    if not free.contains(robot):
        return "the robot's leg leaves the free space"
    points = placement.ped_route()
    ped = (shapely.LineString(points) if len(points) > 1 else shapely.Point(points[0])).buffer(PED_RADIUS + CLEARANCE)
    if not free.contains(ped):
        return "the pedestrian's route leaves the free space"
    return None


def summary(reasons: Sequence[str], limit: int = 3) -> str:
    """The first distinct rejection reasons, joined."""
    return "; ".join(list(dict.fromkeys(reasons))[:limit])


def _go_to(pt: Point, theta: float) -> dict[str, object]:
    return {"kind": "go_to", "target_pose": {"x": round(pt[0], 2), "y": round(pt[1], 2), "theta": round(theta, 3)}, "patience": {"mean": WALK_PATIENCE_S}}


def _pose(pt: Point, theta: float) -> list[float]:
    return [round(pt[0], 3), round(pt[1], 3), round(theta, 3)]


def release_time(sample: Sample, placement: Placement) -> float:
    """Seconds into the episode the robot reaches the release at its nominal speed."""
    return round(max(placement.release or 0.0, 0.0) / sample.robot_speed, 2)


def build(sample: Sample, placement: Placement, notes: Sequence[str] = ()) -> Generated:
    """The scenario, its agent types and the provenance of one placed sample."""
    rng = random.Random(f"{sample.world_key}:{sample.scenario_key}")
    files = TypeFiles(base_dir=Path())
    ref = files.standing(sample.agent_type, PED_NAME)
    timeline: list[dict[str, object]] = []
    if placement.release is not None:
        route = placement.ped_route()
        steps: dict[str, object] = {f"to_{i}": _go_to(b, _heading(a, b)) for i, (a, b) in enumerate(zip(route, route[1:], strict=False), start=1)}
        steps["stay"] = dict(STAND_STEP)
        ref = files.switching(ref, need=GO, to="walk", steps=steps)
        stimulus = [{"agents": [PED_NAME], "need": GO}]
        if placement.release <= 0.0 or sample.pedestrian is Pedestrian.FIXED:
            timeline.append({"at": release_time(sample, placement), "stimulus": stimulus})
        else:
            point, radius = release_point([placement.robot.start, placement.robot.end], placement.release)
            timeline.append({"when": within(radius, point), "stimulus": stimulus})
    if sample.pedestrian is Pedestrian.FIXED:
        files.docs[ref]["local_planner"] = FIXED_PLANNER
    human = {
        "name": PED_NAME,
        "model": mesh_for(profile(sample.agent_type), rng),
        "pose": _pose(placement.spawn, placement.facing),
        "agent": {"agent_type": ref, "desired_velocity": round(sample.ped_speed, 2)},
        "waypoints": [[round(placement.spawn[0], 2), round(placement.spawn[1], 2), 0.0]],
        "waypoint_mode": "once",
    }
    robot = placement.robot
    scenario: dict[str, object] = {
        "format": SCENARIO_FORMAT,
        "humans": [human],
        "robots": [{"start": _pose(robot.start, robot.heading), "phases": [{"goto": _pose(robot.end, robot.heading)}]}],
    }
    if timeline:
        scenario["timeline"] = timeline
    parse_scenario(scenario)
    provenance: dict[str, object] = {
        "sample": sample.to_dict(),
        "design": {
            "robot_start": list(_pose(robot.start, robot.heading)),
            "robot_goal": list(_pose(robot.end, robot.heading)),
            "ped_route": [[round(x, 3), round(y, 3)] for x, y in placement.ped_route()],
            "meet": [round(placement.meet[0], 3), round(placement.meet[1], 3)],
            "release_arc": None if placement.release is None else round(placement.release, 3),
            "release_time": None if placement.release is None else release_time(sample, placement),
        },
        "notes": list(notes),
    }
    return Generated(scenario=scenario, types={ref[2:]: doc for ref, doc in files.docs.items()}, provenance=provenance)
