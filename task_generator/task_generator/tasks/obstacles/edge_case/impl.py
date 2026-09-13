"""``tm_obstacles:=edge_case`` - runs the effects a scenario's `edge_case:` block declares.

The base population comes from the scenario file, untouched: everything static about a case -
who is present, their profiles, a placed formation - was written into `scenario.yaml` by the
generator. What this mode adds at reset is what depends on the live robot:

* `intercept` - one pedestrian placed so it meets the robot at a designed point and time on
  its *planned* route (`geometry.py`, `pathing.py`);
* `rally`, `shuffle`/`scatter`, `track_robot`, `hold` - route rewrites with an onset, driven
  on sim time by `waypoint_driver.py`, optionally gated on the robot's proximity (`triggers.py`);
* an object timeline (`objects.py`) for hand-authored scenarios.

Every effect is recorded as designed in `cases.jsonl`; what actually happened - which plans
fired and when, what the pedestrians did - goes to `scores.jsonl`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Container, Mapping, Sequence
from pathlib import Path
from typing import Any

import attrs
import shapely
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier

from task_generator.constants import Constants
from task_generator.shared import DynamicObstacle, Orientation, Pose, Position
from task_generator.simulators.human.arena_humansim import resolve_agent_type_path
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles
from task_generator.tasks.obstacles._validation import IsValid, in_map_bounds, make_is_traversable, make_is_valid

from .agent_types import AgentTypeError, load_raw, speed_of
from .criticality import Sample, formation_hold
from .criticality import score as criticality_score
from .effects import (
    ROUTE_TRIGGER_M,
    SCOPE_ALL,
    SCOPE_INJECTED,
    After,
    Effect,
    Hold,
    Intercept,
    Rally,
    Retune,
    Shuffle,
    TrackRobot,
    When,
)
from .geometry import GeometryError, encounter_fractions, solve_encounter
from .object_driver import DEFAULT_RATE_HZ as OBJECT_RATE_HZ
from .object_driver import ObjectDriver, submit_to
from .objects import ResolvedEvent, TimelineError
from .objects import load_timeline as load_object_timeline
from .objects import resolve as resolve_objects
from .pathing import PathError, plan_route, point_at_arc, route_length
from .provenance import CaseLog, CaseRecord
from .scenario_block import Block, BlockError, scenario_path
from .scenario_block import read as read_block
from .scoring import DEFAULT_RATE_HZ, EpisodeScorer, write_score
from .triggers import Trigger, centroid
from .waypoint_driver import DEFAULT_RATE_HZ as WAYPOINT_RATE_HZ
from .waypoint_driver import ROBOT_DEPART_M, WaypointDriver, Work
from .waypoints import Mode as WaypointMode
from .waypoints import Plan, Point, WaypointError

#: Fallback robot design speed, m/s, when `task.edge_case.robot_speed` cannot be read.
_ROBOT_SPEED = 1.0

#: Name prefix for injected agents, so scorers can isolate them from the base population.
_INJECT_PREFIX = "edge"

_SCOPE_AUTO = "auto"
_SCOPE_ALL = "all"
_SCOPE_INJECTED = "injected"

#: Profile and mesh used when the base population offers nothing to adopt.
_FALLBACK_TYPE = "adult"
_FALLBACK_MODEL = "arenian"

#: The robot mode's traversal count, as declared.
_ROBOT_TRAVERSALS_PARAM = "task.edge_case_robot.traversals"

#: One armed plan: the plan, its gate, and the fixed point the gate measures to.
ArmedPlan = tuple[Plan, Trigger | None, Point | None]


@attrs.define()
class EdgeCaseConfig:
    read_scenario_block: ROSParamT[bool]
    prompt: ROSParamT[str]
    prompt_base: ROSParamT[str]
    record_dir: ROSParamT[str]
    score: ROSParamT[bool]
    score_rate_hz: ROSParamT[float]
    score_scope: ROSParamT[str]
    objects: ROSParamT[str]
    objects_rate_hz: ROSParamT[float]


class CaseAborted(RuntimeError):
    """The case could not be built. Recorded and re-raised as an episode abort rather than
    silently degrading to an unperturbed run."""


class TM_EdgeCase(TM_Obstacles):
    _config: EdgeCaseConfig

    # Construction
    # ------------

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        self._config = EdgeCaseConfig(
            read_scenario_block=self.node.ROSParam[bool](self.namespace("read_scenario_block"), value=True),
            prompt=self.node.ROSParam[str](self.namespace("prompt"), value=""),
            prompt_base=self.node.ROSParam[str](self.namespace("prompt_base"), value=""),
            record_dir=self.node.ROSParam[str](self.namespace("record_dir"), value=""),
            score=self.node.ROSParam[bool](self.namespace("score"), value=True),
            score_rate_hz=self.node.ROSParam[float](self.namespace("score_rate_hz"), value=DEFAULT_RATE_HZ),
            score_scope=self.node.ROSParam[str](self.namespace("score_scope"), value=_SCOPE_AUTO),
            objects=self.node.ROSParam[str](self.namespace("objects"), value=""),
            objects_rate_hz=self.node.ROSParam[float](self.namespace("objects_rate_hz"), value=OBJECT_RATE_HZ),
        )

        self._block: Block | None = None
        self._last_block_path: str = ""
        #: prompt text -> scenario id it was realised as, so a repeated prompt is not regenerated.
        self._prompt_scenarios: dict[str, str] = {}

        self._scorer: EpisodeScorer | None = None
        self._score_broken = False
        self._pending_case: dict[str, Any] | None = None

        self._objects: ObjectDriver | None = None
        self._objects_broken = False
        self._timeline_cache: tuple[str, list[Any]] | None = None

        self._waypoints: WaypointDriver | None = None
        self._waypoints_broken = False

        self._base_cache: TM_Obstacles | None = None
        self._type_cache: dict[str, dict[str, Any]] = {}
        self._route_cache: list[Point] | None = None
        #: The population as placed this episode, by name: what a retune respawns from.
        self._population: dict[str, DynamicObstacle] = {}

    # Collaborators
    # -------------

    def _base(self) -> TM_Obstacles:
        """The scenario mode supplying the base population."""
        from task_generator.tasks.registry import OBSTACLES_MODES  # noqa: PLC0415

        if self._base_cache is not None:
            return self._base_cache
        key = Constants.TaskMode.TM_Obstacles.SCENARIO
        if key not in OBSTACLES_MODES:
            raise CaseAborted("the scenario obstacle mode is not registered")
        cls = OBSTACLES_MODES.get(key)
        meta = OBSTACLES_MODES.meta(key)
        self._base_cache = cls(ctx=self._ctx, namespace=meta.namespace, node=self.node)
        return self._base_cache

    def _scenario_name(self) -> str:
        return str(self.node.rosparam[str].get("task.scenario.file", "") or "").strip()

    def _scenario_dirs(self) -> list[str]:
        """Where a scenario-relative file may live: the active scenario's directory, then the
        world's `scenarios/` root. Same convention as `agent_type: ./types/x.yaml`."""
        try:
            world = Path(WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().path)
        except Exception:
            return []
        scenarios = world / "scenarios"
        scenario = self._scenario_name()
        dirs = [str(scenarios / scenario)] if scenario else []
        dirs.append(str(scenarios))
        return dirs

    # Scoring
    # -------

    def _ensure_scorer(self) -> EpisodeScorer | None:
        if not self._config.score.value:
            self._teardown_scorer()
            return None
        if self._scorer is not None or self._score_broken:
            return self._scorer
        robots = self._ctx.robots
        if not robots:
            return None
        try:
            _, manager = next(iter(robots.items()))
            # Direct in-process roster feed: the human simulator's live roster object, no middleware.
            sim = getattr(self._ctx.environment_manager, "_human_simulator", None)
            source = None
            if sim is not None and hasattr(sim, "_arena_pedestrians"):
                source = lambda: sim._arena_pedestrians  # noqa: E731 - reference swap read, GIL-atomic
            scorer = EpisodeScorer(
                manager,
                rate_hz=float(self._config.score_rate_hz.value or DEFAULT_RATE_HZ),
                peds_source=source,
            )
            self.node.executor.add_node(scorer)
        except Exception as exc:
            self._score_broken = True
            self._logger.warn(f"edge_case: scoring disabled for this run ({exc!r})")
            return None
        self._scorer = scorer
        self._logger.info(f"edge_case: scoring active at {self._config.score_rate_hz.value} Hz -> {self._case_log().path.parent / 'scores.jsonl'}")
        return scorer

    def _teardown_scorer(self) -> None:
        if self._scorer is None:
            return
        try:
            self.node.executor.remove_node(self._scorer)
            self._scorer.shutdown()
        except Exception as exc:
            self._logger.warn(f"edge_case: scorer teardown failed ({exc!r})")
        finally:
            self._scorer = None

    def _score_scope(self, case: dict[str, Any], scorer: EpisodeScorer) -> tuple[str, list[str]]:
        scope = (self._config.score_scope.value or _SCOPE_AUTO).strip().lower()
        if scope == _SCOPE_ALL:
            return _SCOPE_ALL, []
        if scope == _SCOPE_AUTO and not case.get("injected"):
            return _SCOPE_ALL, []
        seen = scorer.injected_seen()
        if not seen:
            self._logger.warn("edge_case: score_scope wanted the injected agents but none appeared on arena_peds; scoring all")
            return _SCOPE_ALL, []
        return _SCOPE_INJECTED, seen

    def flush_score(self) -> None:
        """Score the episode that just ended and append it to ``scores.jsonl``."""
        scorer, case = self._scorer, self._pending_case
        self._pending_case = None
        if scorer is None or case is None:
            return
        try:
            scope, agents = self._score_scope(case, scorer)
            point = case.get("encounter_point")
            encounter = (float(point[0]), float(point[1])) if point else None
            samples = getattr(scorer, "samples", None)
            goal_xy = None
            away = None
            route = case.get("robot_route") or case.get("robot_leg") or ()
            length = float(case.get("route_length") or 0.0)
            if length > 0 and route and samples:
                gx, gy = (float(v) for v in route[-1][:2])
                # The samples are map-frame, the case route scenario-frame: realise the goal the way the
                # formation centre is realised below.
                placed = self._ctx.environment_manager.realize(Pose(Position(x=gx, y=gy)))
                goal_xy = (float(placed.position.x), float(placed.position.y))

                def goal_dist(s: Sample) -> float:
                    return math.hypot(s.robot[0] - goal_xy[0], s.robot[1] - goal_xy[1])

                # Samples can predate the teleport back to the start; an arrival only counts after the
                # robot has once been clear of the goal.
                away = next((i for i, s in enumerate(samples) if goal_dist(s) > 1.2), None)
                # Truncate at arrival: sampling continues until the next reset. Keep 2 s past the first
                # sample inside the accept radius.
                hit = None if away is None else next(
                    (i for i in range(away, len(samples)) if goal_dist(samples[i]) <= 0.6), None)
                if hit is not None:
                    keep_t = samples[hit].t + 2.0
                    samples = samples[: hit + 1] + [s for s in samples[hit + 1:] if s.t <= keep_t]
            if goal_xy is not None and samples:
                result = criticality_score(samples, encounter=encounter, only=agents or None)
                ambient = criticality_score(samples, encounter=encounter) if agents else None
            else:
                result = scorer.result(encounter=encounter, only=agents or None)
                ambient = scorer.result(encounter=encounter) if agents else None
            if not result.observed:
                self._logger.warn("edge_case: episode produced no usable samples; no score written")
                return
            extra = {**self._object_outcomes(), **self._waypoint_outcomes()}
            # Route progress: closest approach to the (map-frame) goal over the truncated
            # samples; the robots-mode verdict (dwell timeout vs blocked abort) lives in the
            # manifest, not here.
            if goal_xy is not None and samples and away is not None:
                # Distances only from the first clear-of-goal sample: the pre-teleport prefix
                # sits at the previous goal and would fake both progress and arrival.
                nearest = min(math.hypot(s.robot[0] - goal_xy[0], s.robot[1] - goal_xy[1]) for s in samples[away:])
                extra["progress"] = round(max(0.0, min(1.0, 1.0 - nearest / length)), 4)
                extra["arrived"] = nearest <= 0.6
                extra["ended"] = "arrived" if nearest <= 0.6 else "cut"
            members = list((self._block.owned.get("D") if self._block else None) or ())
            if members:
                spec = dict(self._block.formation) if self._block else {}
                centre = None
                if spec.get("centre"):
                    # The block's centre is abstract (scenario frame); the scorer's samples are
                    # map-frame, so the centre is realised the way the placements were.
                    cx, cy = (float(v) for v in spec["centre"])
                    placed = self._ctx.environment_manager.realize(Pose(Position(x=cx, y=cy)))
                    centre = (float(placed.position.x), float(placed.position.y))
                extra["formation"] = formation_hold(scorer.samples, members, radius=float(spec.get("radius", 1.5)), centre=centre)  # type: ignore[arg-type]
            write_score(
                self._case_log().path.parent, case, result,
                scope=scope, scored_agents=agents, ambient=ambient, extra=extra,
            )
            self._logger.info(f"edge_case: scored episode {case.get('episode_id')} scope={scope} min_ttc={result.min_ttc_s} min_clearance={result.min_clearance_m}")
        except Exception as exc:
            self._logger.warn(f"edge_case: scoring this episode failed ({exc!r})")

    def _arm_score(self) -> None:
        scorer = self._ensure_scorer()
        if scorer is not None:
            scorer.reset()

    def _write_case(self, record: CaseRecord) -> None:
        self._pending_case = attrs.asdict(record)
        self._case_log().write(record)

    def _case_log(self) -> CaseLog:
        return CaseLog(self._config.record_dir.value or None)

    # Object timelines
    # ----------------

    def _timeline_source(self) -> str:
        return (self._config.objects.value or "").strip() or (self._block.objects if self._block else "")

    def _timeline(self) -> list[Any]:
        path = self._timeline_source()
        if not path:
            self._timeline_cache = None
            return []
        if self._timeline_cache is not None and self._timeline_cache[0] == path:
            return self._timeline_cache[1]
        events = load_object_timeline(path, self._scenario_dirs())
        self._timeline_cache = (path, events)
        return events

    def _ensure_object_driver(self) -> ObjectDriver | None:
        if self._objects is not None or self._objects_broken:
            return self._objects
        robots = self._ctx.robots
        if not robots:
            return None
        try:
            _, manager = next(iter(robots.items()))
            driver = ObjectDriver(
                str(manager.namespace),
                rate_hz=float(self._config.objects_rate_hz.value or OBJECT_RATE_HZ),
                spawn=self._spawn_object,
                despawn=self._despawn_object,
                submit=submit_to(self.node.event_loop),
            )
            self.node.executor.add_node(driver)
        except Exception as exc:
            self._objects_broken = True
            self._logger.warn(f"edge_case: object timelines disabled for this run ({exc!r})")
            return None
        self._objects = driver
        return driver

    def _teardown_object_driver(self) -> None:
        if self._objects is None:
            return
        try:
            self.node.executor.remove_node(self._objects)
            self._objects.shutdown()
        except Exception as exc:
            self._logger.warn(f"edge_case: object driver teardown failed ({exc!r})")
        finally:
            self._objects = None

    async def _spawn_object(self, event: ResolvedEvent) -> str:
        from task_generator.tasks.obstacles import ObstacleKind  # noqa: PLC0415

        assert event.pose is not None
        pose = Pose(Position(float(event.pose[0]), float(event.pose[1])), orientation=Orientation.from_yaw(float(event.yaw)))
        return await self.extend(ObstacleKind.STATIC, event.event.model, pose)

    async def _despawn_object(self, event: ResolvedEvent, entity_id: str) -> bool:
        del event
        return await self.retract(entity_id)

    def _arm_objects(self, route_of: Callable[[], Sequence[Point]]) -> list[dict[str, Any]]:
        """Resolve the timeline against this episode's route and arm the driver. A timeline
        that cannot be resolved aborts the case: an object that never appeared means the case
        did not run."""
        try:
            events = self._timeline()
        except TimelineError as exc:
            raise CaseAborted(str(exc)) from None
        if not events:
            if self._objects is not None:
                self._objects.arm([])
            return []
        needs_route = any(e.at_fraction is not None or (e.pose is None and e.action != "despawn") for e in events)
        route: Sequence[Point] = ()
        if needs_route:
            try:
                route = route_of()
            except (CaseAborted, GeometryError, PathError) as exc:
                raise CaseAborted(f"the object timeline is route-relative and the route could not be planned: {exc}") from None
        try:
            resolved = resolve_objects(events, route, robot_speed=self._robot_speed(), delay=0.0)
        except TimelineError as exc:
            raise CaseAborted(str(exc)) from None
        driver = self._ensure_object_driver()
        if driver is None:
            raise CaseAborted("object timeline configured but the driver could not be created")
        driver.arm(resolved)
        self._logger.info(f"edge_case: object timeline armed, {len(resolved)} event(s) over {resolved[-1].t:.1f}s")
        return [
            {"entity": r.entity, "action": r.action, "model": r.event.model, "t": r.t, "pose": r.pose,
             "reveal_distance_m": r.reveal_distance_m, "note": r.event.note}
            for r in resolved
        ]

    def _object_outcomes(self) -> dict[str, Any]:
        if self._objects is None:
            return {}
        outcomes = self._objects.outcomes()
        if not outcomes:
            return {}
        return {"object_events": [o.as_dict() for o in outcomes], "objects_pending": self._objects.pending()}

    # Runtime waypoints
    # -----------------

    def _ensure_waypoint_driver(self) -> WaypointDriver | None:
        if self._waypoints is not None or self._waypoints_broken:
            return self._waypoints
        robots = self._ctx.robots
        if not robots:
            return None
        try:
            _, manager = next(iter(robots.items()))
            driver = WaypointDriver(
                str(manager.namespace),
                reroute=self._reroute,
                robot_pose=self._robot_here,
                submit=submit_to(self.node.event_loop),
                rate_hz=WAYPOINT_RATE_HZ,
                retune=self._retune_agents,
                depart_after_m=ROBOT_DEPART_M,
            )
            self.node.executor.add_node(driver)
        except Exception as exc:
            self._waypoints_broken = True
            self._logger.warn(f"edge_case: runtime waypoints disabled for this run ({exc!r})")
            return None
        self._waypoints = driver
        return driver

    def _teardown_waypoint_driver(self) -> None:
        if self._waypoints is None:
            return
        try:
            self.node.executor.remove_node(self._waypoints)
            self._waypoints.shutdown()
        except Exception as exc:
            self._logger.warn(f"edge_case: waypoint driver teardown failed ({exc!r})")
        finally:
            self._waypoints = None

    def _reroute(self, routes: dict[str, Sequence[Point]]) -> Work:
        return self._ctx.environment_manager.set_agent_waypoints(routes)

    async def _retune_agents(self, changes: Mapping[str, Mapping[str, Any]]) -> int:
        """Change the named agents' parameters in place (`update_agents`, HumanSim's
        `UpdateAgents`): new `agent:` block, same position, velocity and route. Returns how many
        the backend changed; 0 when it refused, which the driver records as failures."""
        env = self._ctx.environment_manager
        wanted: list[DynamicObstacle] = []
        for name, block in changes.items():
            original = self._population.get(name)
            if original is None:
                self._logger.warn(f"edge_case: retune: {name!r} is not in this episode's population")
                continue
            obs = DynamicObstacle(
                name=name, model=original.model, pose=original.pose, waypoints=list(original.waypoints),
                velocity=float(block.get("desired_velocity", original.velocity or 0.0) or original.velocity or 0.0),
            )
            obs.extra = {**dict(original.extra or {}), "agent": dict(block)}
            wanted.append(obs)
        if not wanted:
            return 0
        if not await env.update_agents(wanted):
            self._logger.warn(f"edge_case: retune: the backend changed none of {[o.name for o in wanted]}")
            return 0
        for obs in wanted:
            live = self._population[obs.name]
            live.extra = dict(obs.extra)
            live.velocity = obs.velocity
        return len(wanted)

    def _robot_start(self) -> Point | None:
        """Where the robot starts this episode (abstract frame), None until it has been placed."""
        robots = self._ctx.robots
        if not robots:
            return None
        _, manager = next(iter(robots.items()))
        if not getattr(manager, "start_assigned", True):
            return None
        start = manager.start_pos
        return (float(start.position.x), float(start.position.y))

    def _robot_here(self) -> Point | None:
        """The robot's current position in the **abstract** frame, or None while respawning."""
        robots = self._ctx.robots
        if not robots:
            return None
        _, manager = next(iter(robots.items()))
        pose = manager.pose
        if pose is None:
            return None
        abstract = self._ctx.environment_manager.ezilear(pose)
        return (float(abstract.position.x), float(abstract.position.y))

    def _arm_plans(self, entries: Sequence[ArmedPlan]) -> None:
        if not entries:
            if self._waypoints is not None:
                self._waypoints.arm([])
            return
        driver = self._ensure_waypoint_driver()
        if driver is None:
            raise CaseAborted("route-rewrite effects were declared but the waypoint driver could not be created")
        driver.arm(list(entries), origin=self._robot_start())
        for plan, trigger, reference in entries:
            gated = f" once {trigger.describe(reference)}" if trigger else ""
            self._logger.warn(
                f"edge_case: armed {plan.label or plan.mode} over {len(plan.homes)} agent(s), "
                f"first rewrite at t={plan.at:.1f}s{gated}"
                + (f", every {plan.period:g}s" if plan.mode in (WaypointMode.SHUFFLE, WaypointMode.TRACK_ROBOT) and not plan.once else "")
            )

    def _waypoint_outcomes(self) -> dict[str, Any]:
        if self._waypoints is None:
            return {}
        outcomes = self._waypoints.outcomes()
        if not outcomes:
            return {}
        return {"waypoint_outcomes": [o.as_dict() for o in outcomes]}

    # World geometry
    # --------------

    def _zone_centroid(self, name: str) -> Point:
        try:
            zones = self._ctx.world_manager.world_compacted().zones or ()
        except Exception as exc:
            raise CaseAborted(f"zone {name!r}: the world has no readable zones ({exc!r})") from None
        for zone in zones:
            if str(zone.name) == name:
                polygon = shapely.Polygon([(float(c.x), float(c.y)) for c in zone.corners])
                if polygon.is_empty:
                    raise CaseAborted(f"zone {name!r} has no usable polygon")
                c = polygon.centroid
                return (float(c.x), float(c.y))
        known = sorted(str(z.name) for z in zones)
        raise CaseAborted(f"{name!r} is not a zone of this world; known zones: {known}")

    def _resolve_point(self, target: str | Point, *, where: str) -> Point:
        """A zone name -> its centroid (abstract frame, like agent homes); a pair passes through."""
        if isinstance(target, str):
            return self._zone_centroid(target)
        try:
            return (float(target[0]), float(target[1]))
        except (TypeError, ValueError, IndexError):
            raise CaseAborted(f"{where}: {target!r} is neither a zone name nor an [x, y] pair") from None

    def _robot_route(self) -> list[Point]:
        """The robot's planned route for this episode, as a polyline, or abort. Memoised per reset."""
        if self._route_cache is not None:
            return self._route_cache
        robots = self._ctx.robots
        if not robots:
            raise CaseAborted("an effect needs the robot's route, but no robot is present")
        name, manager = next(iter(robots.items()))
        if not getattr(manager, "start_assigned", True):
            raise CaseAborted(
                f"robot {name!r} has not been placed yet, so `start_pos` is still the map origin. "
                "This happens on the first episode, where obstacles can be generated before the robot is moved."
            )
        start = (float(manager.start_pos.position.x), float(manager.start_pos.position.y))
        goal = (float(manager.route_goal.position.x), float(manager.route_goal.position.y))
        if math.dist(start, goal) < 1e-6:
            raise CaseAborted(
                f"robot {name!r} has no distinct goal at obstacle-generation time, so there is no route to "
                "design against. Use tm_robots:=edge_case (or :=scenario); tm_robots:=explore assigns goals lazily."
            )
        world_map = self._ctx.world_manager.map
        if world_map is None:
            self._route_cache = [start, goal]
            return self._route_cache
        for label, pt in (("start", start), ("goal", goal)):
            if not in_map_bounds(world_map, Position(x=pt[0], y=pt[1])):
                raise CaseAborted(f"robot {name!r} {label} ({pt[0]:.2f}, {pt[1]:.2f}) is outside the map entirely")
        try:
            route = plan_route(world_map, start, goal, is_valid=self._is_traversable())
        except PathError as exc:
            raise CaseAborted(
                f"robot {name!r} has no navigable route from ({start[0]:.2f}, {start[1]:.2f}) to ({goal[0]:.2f}, {goal[1]:.2f}): {exc}"
            ) from exc
        if len(route) > 2:
            self._logger.info(f"edge_case: planned the robot's route - {len(route)} waypoint(s), {route_length(route):.2f} m")
        self._route_cache = route
        return route

    def _traversals(self) -> int | None:
        try:
            if self.node.has_parameter(_ROBOT_TRAVERSALS_PARAM):
                return int(self.node.get_parameter(_ROBOT_TRAVERSALS_PARAM).value)
        except Exception:  # pragma: no cover
            pass
        return None

    def _is_valid(self) -> IsValid | None:
        return make_is_valid(self._ctx.world_manager.map, self.node.conf.Obstacles.SAFE_DIST.value)

    def _is_traversable(self) -> IsValid | None:
        world_map = self._ctx.world_manager.map
        if world_map is None:
            return None
        return make_is_traversable(world_map, self.node.conf.Obstacles.SAFE_DIST.value)

    def _validate_poses(self, obstacles: Sequence[DynamicObstacle], fatal: Container[str]) -> tuple[str, ...]:
        """Check spawns against free space; return the names that failed. Only agents in
        ``fatal`` - the injected ones - abort the case."""
        is_valid = self._is_valid()
        if is_valid is None:
            return ()
        invalid: list[str] = []
        for obs in obstacles:
            pt = Position(x=obs.pose.position.x, y=obs.pose.position.y)
            if not is_valid(pt):
                invalid.append(obs.name)
                if obs.name in fatal:
                    raise CaseAborted(f"edge-case agent {obs.name!r} spawn ({pt.x:.2f}, {pt.y:.2f}) is not in free space")
        if invalid:
            self._logger.warn(
                f"edge_case: {len(invalid)} base-scenario agent(s) spawn outside free space ({', '.join(invalid)}); recorded, not fatal"
            )
        return tuple(invalid)

    # Agents
    # ------

    @staticmethod
    def _agent_type_of(obs: DynamicObstacle) -> str | None:
        agent_block = (obs.extra or {}).get("agent")
        if isinstance(agent_block, str):
            return agent_block
        if isinstance(agent_block, dict):
            return str(agent_block.get("agent_type", "adult"))
        return None

    def _resolved_type(self, obs: DynamicObstacle) -> dict[str, Any]:
        declared = self._agent_type_of(obs) or "adult"
        resolved = resolve_agent_type_path(declared, getattr(obs, "included_from", None))
        if resolved not in self._type_cache:
            self._type_cache[resolved] = load_raw(resolved)
        return self._type_cache[resolved]

    @staticmethod
    def _model_name(model: object) -> str:
        return str(getattr(model, "name", model))

    def _resolve_profile(self, configured: str) -> str:
        """A `./types/name.yaml` profile, found beside the active scenario; builtins pass through."""
        candidate = Path(configured)
        if candidate.is_absolute() or candidate.suffix != ".yaml" or candidate.is_file():
            return configured
        for scenario_dir in self._scenario_dirs():
            found = Path(scenario_dir) / candidate
            if found.is_file():
                return str(found)
        return configured

    def _inject_profile(self, effect: Intercept, dynamic: Sequence[DynamicObstacle]) -> tuple[str, dict[str, Any]]:
        """Agent type for an injected pedestrian: the effect's, else one adopted from the base
        population (sorted, so the pick is reproducible), else `adult`."""
        if effect.profile:
            resolved = self._resolve_profile(effect.profile)
            try:
                return resolved, load_raw(resolved)
            except AgentTypeError as exc:
                raise CaseAborted(f"intercept profile {effect.profile!r}: {exc}") from None
        seen: list[tuple[str, dict[str, Any]]] = []
        for obs in dynamic:
            declared = self._agent_type_of(obs)
            if declared is None:
                continue
            try:
                resolved = resolve_agent_type_path(declared, getattr(obs, "included_from", None))
                seen.append((resolved, self._resolved_type(obs)))
            except AgentTypeError:
                continue
        if seen:
            return min(seen, key=lambda pair: pair[0])
        return _FALLBACK_TYPE, load_raw(_FALLBACK_TYPE)

    def _inject_model(self, effect: Intercept, dynamic: Sequence[DynamicObstacle]) -> object:
        if effect.model:
            return effect.model
        models = [o.model for o in dynamic if o.model is not None]
        if not models:
            return _FALLBACK_MODEL
        return min(models, key=self._model_name)

    # Effects
    # -------

    def _when(self, when: When, *, default_reference: Point | None) -> tuple[Trigger | None, Point | None, float]:
        """Resolve an effect's onset to (gate, reference point, at)."""
        if when.route_fraction is not None:
            route = self._robot_route()
            point, _heading = point_at_arc(route, when.route_fraction * route_length(route))
            return Trigger(robot_within=ROUTE_TRIGGER_M), point, when.at
        if when.robot_within is None and when.robot_beyond is None:
            return None, None, when.at
        trigger = Trigger(robot_within=when.robot_within, robot_beyond=when.robot_beyond, of=when.of)
        reference = self._resolve_point(when.of, where="when.of") if when.of is not None else default_reference
        if reference is None:
            raise CaseAborted("a proximity gate has no reference point: the effect owns no agents and names no `of`")
        return trigger, reference, when.at

    def _owned(self, scope: str | tuple[str, ...], dynamic: Sequence[DynamicObstacle], *, where: str) -> dict[str, Point]:
        """Agent name -> placed position, for the agents an effect owns. Empty scope aborts:
        a rally that owns nobody is a case that did not run."""
        by_name = {str(a.name): (float(a.pose.position.x), float(a.pose.position.y)) for a in dynamic}
        if scope == SCOPE_ALL:
            owned = dict(by_name)
        elif scope == SCOPE_INJECTED:
            owned = {n: p for n, p in by_name.items() if n.startswith(f"{_INJECT_PREFIX}_")}
        else:
            missing = [n for n in scope if n not in by_name]
            if missing:
                raise CaseAborted(f"{where}: scope names agent(s) this episode did not place: {missing}; placed: {sorted(by_name)}")
            owned = {n: by_name[n] for n in scope}
        if not owned:
            raise CaseAborted(f"{where}: scope {scope!r} matched none of the {len(dynamic)} agent(s) this episode placed")
        return owned

    def _plan_for(self, effect: Effect, index: int, dynamic: Sequence[DynamicObstacle]) -> ArmedPlan:
        where = f"effects[{index}] ({effect.kind})"
        label = effect.label or f"{effect.kind}#{index}"
        homes = self._owned(effect.scope, dynamic, where=where)  # type: ignore[attr-defined]
        trigger, reference, at = self._when(effect.when, default_reference=centroid(list(homes.values())))
        try:
            if isinstance(effect, Rally):
                points = [self._resolve_point(t, where=f"{where}.targets") for t in effect.targets]
                targets = {name: points[i % len(points)] for i, name in enumerate(sorted(homes))}
                plan = Plan(mode=WaypointMode.RALLY, homes=homes, at=at, targets=targets, label=label)
            elif isinstance(effect, Shuffle):
                plan = Plan(mode=WaypointMode.SHUFFLE, homes=homes, at=at, period=effect.period, radius=effect.radius, once=effect.once, label=label)
            elif isinstance(effect, TrackRobot):
                plan = Plan(mode=WaypointMode.TRACK_ROBOT, homes=homes, at=at, period=effect.period, duration=effect.duration, label=label)
            elif isinstance(effect, Hold):
                plan = Plan(mode=WaypointMode.HOLD, homes=homes, at=at, duration=effect.duration, resume=self._goals_of(homes, dynamic), label=label)
            elif isinstance(effect, Retune):
                plan = Plan(mode=WaypointMode.RETUNE, homes=homes, at=at, retune=self._retune_blocks(effect, homes, dynamic), label=label)
            else:  # pragma: no cover - parse() only builds the kinds above
                raise CaseAborted(f"{where}: no runtime for this effect")
        except WaypointError as exc:
            raise CaseAborted(f"{where}: {exc}") from None
        return plan, trigger, reference

    def _retune_blocks(self, effect: Retune, homes: dict[str, Point], dynamic: Sequence[DynamicObstacle]) -> dict[str, dict[str, Any]]:
        """The new `agent:` block per owned agent: the effect's profile (resolved beside the
        scenario) or the agent's own, and its speed, scaled or replaced."""
        profile = self._resolve_profile(effect.profile) if effect.profile else ""
        if profile:
            try:
                load_raw(profile)
            except AgentTypeError as exc:
                raise CaseAborted(f"retune profile {effect.profile!r}: {exc}") from None
        out: dict[str, dict[str, Any]] = {}
        for agent in dynamic:
            name = str(agent.name)
            if name not in homes:
                continue
            block = dict((agent.extra or {}).get("agent") or {}) if isinstance((agent.extra or {}).get("agent"), dict) else {"agent_type": self._agent_type_of(agent) or "adult"}
            if profile:
                block["agent_type"] = profile
            own = block.get("desired_velocity")
            if effect.speed:
                block["desired_velocity"] = float(effect.speed)
            elif isinstance(own, (int, float)):
                block["desired_velocity"] = round(float(own) * effect.speed_scale, 3)
            elif abs(effect.speed_scale - 1.0) > 1e-9:
                block["desired_velocity"] = round(speed_of(self._resolved_type(agent)) * effect.speed_scale, 3)
            out[name] = block
        return out

    @staticmethod
    def _goals_of(homes: dict[str, Point], dynamic: Sequence[DynamicObstacle]) -> dict[str, Point]:
        """Each owned agent's first waypoint - where a hold releases it to."""
        out: dict[str, Point] = {}
        for agent in dynamic:
            if str(agent.name) in homes and agent.waypoints:
                wp = agent.waypoints[0]
                out[str(agent.name)] = (float(wp.x), float(wp.y))
        return out

    def _inject(self, effect: Intercept, index: int, ordinal: int, dynamic: Sequence[DynamicObstacle]) -> tuple[DynamicObstacle, dict[str, Any], list[ArmedPlan]]:
        """Build one intercepting pedestrian: the agent, its geometry record, and any plan the
        `after` clause needs."""
        where = f"effects[{index}] (intercept)"
        name = f"{_INJECT_PREFIX}_{ordinal}"
        profile_ref, raw = self._inject_profile(effect, dynamic)
        speed = effect.speed or speed_of(raw)
        model = self._inject_model(effect, dynamic)
        route = self._robot_route()
        is_valid = self._is_valid()
        # The solver probes the route itself for free space; that must be *physical* free
        # space, not the forbidden discs reserved around the robot's own start and goal, or
        # every route ending at a goal in a room reads as blocked within a metre of it.
        # The pedestrian's spawn is still checked against the full predicate afterwards.
        is_traversable = self._is_traversable()
        plans: list[ArmedPlan] = []
        geometry: dict[str, Any] = {
            "inject_type": profile_ref, "inject_model": self._model_name(model), "waypoint_mode": effect.waypoint_mode,
            "after": str(effect.after), "ped_speed": speed,
            "robot_leg": (route[0], route[-1]), "robot_route": tuple(route), "route_length": route_length(route),
            "route_planned": len(route) > 2, "approach_angle_requested": effect.angle,
            "encounter_at_requested": effect.route_fraction, "co_arrival_offset": effect.offset,
        }

        if effect.after is After.STAND:
            total = route_length(route)
            spot: Point | None = None
            achieved = effect.route_fraction
            for fraction in encounter_fractions(effect.route_fraction):
                point, heading = point_at_arc(route, fraction * total)
                if is_valid is None or is_valid(Position(x=point[0], y=point[1])):
                    spot, achieved = point, fraction
                    break
            if spot is None:
                raise CaseAborted(f"{where}: no free point on the robot's route to stand on")
            yaw = math.atan2(-heading[1], -heading[0])  # facing the oncoming robot
            t_arrival = achieved * total / self._robot_speed()
            release = point_at_arc(route, min(total, achieved * total + effect.lead_out))[0]
            obs = DynamicObstacle(
                name=name, model=model,
                pose=Pose(Position(x=spot[0], y=spot[1]), orientation=Orientation.from_yaw(yaw)),
                waypoints=[Position(x=spot[0], y=spot[1])], velocity=speed,
            )
            obs.extra["agent"] = {"agent_type": profile_ref, "desired_velocity": speed}
            obs.extra["waypoint_mode"] = "once"
            geometry.update({
                "place": "stand", "approach_angle_achieved": effect.angle, "encounter_at_achieved": achieved,
                "encounter_point": spot, "t_encounter": t_arrival, "designed_ttc": t_arrival,
                "spawn_pose": spot, "ped_goal": spot, "geometry_retries": 0,
            })
            if effect.duration > 0:
                plans.append((Plan(mode=WaypointMode.HOLD, homes={name: spot}, at=0.0, duration=effect.duration,
                                   resume={name: release}, label=f"{name}:stand"), None, None))
            return obs, geometry, plans

        try:
            enc = solve_encounter(
                route, robot_speed=self._robot_speed(), ped_speed=speed, approach_angle=effect.angle,
                encounter_at=effect.route_fraction, co_arrival_offset=effect.offset,
                lead_out=effect.lead_out, angle_search=effect.search, is_valid=is_traversable,
                spawn_valid=self._is_valid(),
            )
        except GeometryError as exc:
            raise CaseAborted(f"{where}: {exc}") from None
        if enc.deviated:
            self._logger.warn(
                f"edge_case: {where} requested angle={enc.angle_requested:g} at {enc.encounter_at_requested:g} was obstructed; "
                f"solved at {enc.angle_achieved:g} / {enc.encounter_at_achieved:g} after {enc.retries} retries"
            )

        goal = enc.goal
        waypoints = [Position(x=goal[0], y=goal[1])]
        mode = effect.waypoint_mode
        if effect.after is After.VEER:
            assert effect.veer_to is not None
            veer = self._resolve_point(effect.veer_to, where=f"{where}.veer_to")
            waypoints = [Position(x=enc.encounter[0], y=enc.encounter[1]), Position(x=veer[0], y=veer[1])]
            mode = "once"
        yaw = math.atan2(enc.encounter[1] - enc.spawn[1], enc.encounter[0] - enc.spawn[0])
        obs = DynamicObstacle(
            name=name, model=model,
            pose=Pose(Position(x=enc.spawn[0], y=enc.spawn[1]), orientation=Orientation.from_yaw(yaw)),
            waypoints=waypoints, velocity=speed,
        )
        obs.extra["agent"] = {"agent_type": profile_ref, "desired_velocity": speed}
        obs.extra["waypoint_mode"] = mode
        geometry.update({
            "place": "intercept", "approach_angle_achieved": enc.angle_achieved,
            "encounter_at_achieved": enc.encounter_at_achieved, "encounter_point": enc.encounter,
            "t_encounter": enc.t_encounter, "designed_ttc": enc.designed_ttc, "designed_pet": enc.designed_pet,
            "lead_out_achieved": enc.lead_out_achieved, "spawn_pose": enc.spawn, "ped_goal": goal,
            "geometry_retries": enc.retries,
        })

        # The interception is designed against a robot that leaves at t=0 - and t=0 is the
        # tick the robot is under way (the driver's clock, `ROBOT_DEPART_M`), so nav2's first
        # plan and any hold the robot mode adds (`task.edge_case_robot.hold`, recorded here)
        # shift nothing: the walker's plans count from the moment the robot actually sets off.
        geometry["robot_hold"] = self._robot_hold()
        geometry["depart_at"] = enc.depart_at
        wait = enc.depart_at
        if wait > 0.0:
            first = (float(waypoints[0].x), float(waypoints[0].y))
            plans.append((Plan(mode=WaypointMode.HOLD, homes={name: enc.spawn}, at=0.0, duration=wait,
                               points={name: enc.spawn}, resume={name: first}, label=f"{name}:wait_for_robot"), None, None))
        if effect.after is After.STOP_FACING:
            plans.append((Plan(mode=WaypointMode.HOLD, homes={name: enc.spawn}, at=enc.t_encounter,
                               duration=effect.duration, points={name: enc.encounter}, resume={name: goal},
                               label=f"{name}:stop_facing"), None, None))
        elif effect.after is After.FOLLOW:
            plans.append((Plan(mode=WaypointMode.TRACK_ROBOT, homes={name: enc.spawn}, at=enc.t_encounter,
                               period=2.0, duration=effect.duration, resume={name: goal},
                               label=f"{name}:follow"), None, None))
        return obs, geometry, plans

    def _robot_speed(self) -> float:
        """Metres per second the robot is assumed to drive (`task.edge_case.robot_speed`)."""
        try:
            value = float(self._config.robot_speed.value)
        except Exception:  # noqa: BLE001 - a test double without the parameter
            return _ROBOT_SPEED
        return value if value > 0.0 else _ROBOT_SPEED

    def _robot_hold(self) -> float:
        """Sim seconds the robot mode holds the robot at its start before its first leg (0 without it)."""
        try:
            return max(0.0, float(self.node.rosparam[float].get("task.edge_case_robot.hold", 0.0) or 0.0))
        except Exception:  # noqa: BLE001 - an undeclared or oddly typed value means no hold
            return 0.0

    # Prompt input (`new_plan_2.md` §12)
    # -----------------------------------

    def _apply_prompt(self) -> None:
        """Realise `task.edge_case.prompt` into a scenario and point `task.scenario.file` at it.

        The same generator the CLI runs, in process: the model answer is cached by content, so
        a repeated prompt is instant and identical. The directory is written into the world's
        installed `scenarios/` tree, which is what the loader reads, so no rebuild is needed.
        """
        text = " ".join(str(self._config.prompt.value or "").split())
        if not text:
            return
        current = self._scenario_name()
        wanted = self._prompt_scenarios.get(text)
        if wanted and wanted == current:
            return
        if wanted:
            self.node.rosparam[str].set("task.scenario.file", wanted)
            return

        import hashlib  # noqa: PLC0415

        base = str(self._config.prompt_base.value or "").strip() or (current.split("__", 1)[0] if "__rviz_" in current else current)
        world_root = Path(WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().path)
        base_path = world_root / "scenarios" / base / "scenario.yaml"
        if not base_path.is_file():
            raise CaseAborted(f"prompt: base scenario {base!r} not found at {base_path}")
        try:
            from arena_benchmark.promptgen import generate, write_result  # noqa: PLC0415
            from arena_benchmark.promptgen.prompts import Prompt  # noqa: PLC0415
            from arena_benchmark.promptgen.world import WorldContext  # noqa: PLC0415
            from arena_benchmark.propose import load_overrides  # noqa: PLC0415
        except ImportError as exc:
            raise CaseAborted(f"prompt: the generator (arena_benchmark.promptgen) is not importable: {exc}") from None
        import yaml  # noqa: PLC0415

        base_doc = yaml.safe_load(base_path.read_text()) or {}
        digest = hashlib.sha256(f"{base}\0{text}".encode()).hexdigest()[:8]
        prompt = Prompt(id=f"rviz_{digest}", text=text, targets=(), oracle={}, scope="rviz", world=str(self._ctx.world_manager.loaded_world))
        self._logger.warn(f"edge_case: realising prompt {prompt.id} on {base!r}: {text}")
        try:
            world = WorldContext.load(world_root.parent, world_root.name, base_doc, load_overrides())
            result = generate(prompt, world=world, base_id=base, base_doc=base_doc, base_dir=base_path.parent)
        except Exception as exc:  # noqa: BLE001 - any generator failure is one abort with the reason
            raise CaseAborted(f"prompt: generation failed: {exc!r}") from None
        if result.record.status == "infeasible":
            raise CaseAborted("prompt: " + "; ".join(f"{f.step}: {f.reason}" for f in result.record.failures))
        out = write_result(result, world_root)
        if out is None:
            raise CaseAborted("prompt: nothing to write")
        for note in result.record.warnings:
            self._logger.info(f"edge_case: prompt - {note}")
        for failure in result.record.failures:
            self._logger.warn(f"edge_case: prompt - {failure.step}: {failure.reason}")
        self._prompt_scenarios[text] = out.name
        self.node.rosparam[str].set("task.scenario.file", out.name)
        self._logger.warn(f"edge_case: prompt realised as {out.name} [{', '.join(result.record.steps) or 'no steps'}] -> task.scenario.file")

    # Entry point
    # -----------

    def _read_block(self) -> Block | None:
        if not bool(self._config.read_scenario_block.value):
            return None
        world = str(getattr(self._ctx.world_manager, "loaded_world", "") or "")
        scenario = self._scenario_name()
        path = scenario_path(world, scenario)
        if path is None:
            if scenario:
                self._logger.debug(f"edge_case: no scenario file for {world}/{scenario}; block not read")
            return None
        try:
            block = read_block(path)
        except BlockError as exc:
            self._logger.error(f"edge_case: {exc}")
            raise
        first_time = str(path) != self._last_block_path
        self._last_block_path = str(path)
        if block is None:
            if first_time:
                self._logger.info(f"edge_case: {path} carries no 'edge_case' block - base population only")
        elif first_time:
            self._logger.warn(f"edge_case: {path} -> {block.describe()}")
        return block

    async def reset(self, **kwargs: object) -> Obstacles:
        # The episode that just ended is scored here, because TM_Obstacles has no episode-end hook.
        self.flush_score()
        self._route_cache = None
        try:
            self._apply_prompt()
        except CaseAborted as exc:
            self._logger.error(f"edge_case: {exc}")
            self._ctx.abort_episode(f"edge_case: {exc}")
            self._block = None
            return [], []
        self._block = self._read_block()
        try:
            return await self._reset(**kwargs)
        finally:
            self._arm_score()

    async def teardown(self) -> None:
        self.flush_score()
        self._teardown_scorer()
        self._teardown_object_driver()
        self._teardown_waypoint_driver()
        if self._base_cache is not None:
            base = self._base_cache
            self._base_cache = None
            await base.teardown()
        await super().teardown()

    async def _reset(self, **kwargs: object) -> Obstacles:
        base = self._base()
        static, dynamic = await base.reset(**kwargs)
        static, dynamic = list(static), list(dynamic)
        self.pending_regions[:] = base.pending_regions

        episode = self.node._episodes
        block = self._block or Block()
        base_record: dict[str, Any] = {
            "run_seed": episode.run_seed, "episode_id": episode.current.episode_id,
            "world": self._ctx.world_manager.loaded_world, "seed": episode.current.seed,
            "scenario": self._scenario_name(), "case_id": block.id, "base": block.base,
            "prompt_id": block.prompt_id, "steps": tuple(block.steps),
            "effects": tuple(e.as_dict() for e in block.effects),
        }

        try:
            object_events = self._arm_objects(self._robot_route)
            if object_events:
                base_record["object_timeline"] = self._timeline_source()
                base_record["object_events"] = tuple(object_events)

            injected: list[DynamicObstacle] = []
            geometry: dict[str, Any] = {}
            plans: list[ArmedPlan] = []
            for index, effect in enumerate(block.effects):
                if isinstance(effect, Intercept):
                    obs, geo, extra = self._inject(effect, index, len(injected), dynamic)
                    injected.append(obs)
                    plans.extend(extra)
                    if not geometry:
                        geometry = {**geo, "target_agent": obs.name}
            if not geometry.get("robot_route"):
                # Record the route for every case that can plan one, not only intercept cases; the first
                # boot episode may not.
                try:
                    route = self._robot_route()
                    geometry.update({"robot_leg": (route[0], route[-1]), "robot_route": tuple(route), "route_length": route_length(route)})
                except CaseAborted as exc:
                    self._logger.info(f"edge_case: no route recorded for scoring ({exc})")
            combined = [*dynamic, *injected]
            self._population = {str(a.name): a for a in combined}
            for index, effect in enumerate(block.effects):
                if not isinstance(effect, Intercept):
                    plans.append(self._plan_for(effect, index, combined))

            invalid_spawns = self._validate_poses(combined, fatal={o.name for o in injected})
            self._arm_plans(plans)

            status = "ok" if not block.empty else "baseline"
            self._write_case(CaseRecord(
                **base_record, **geometry, injected=len(injected), invalid_spawns=invalid_spawns,
                plans=tuple(p.as_dict() for p, _t, _r in plans), traversals_seen=self._traversals(), status=status,
            ))
            if block.empty:
                self._logger.info("edge_case: base population only, nothing added")
            else:
                self._logger.warn(
                    f"edge_case: {block.id or 'case'} built - {len(injected)} injected, {len(plans)} plan(s), "
                    f"{len(dynamic)} base agent(s)"
                )
            return static, combined

        except (CaseAborted, GeometryError, WaypointError, AgentTypeError) as exc:
            self._write_case(CaseRecord(**base_record, status="aborted", reason=str(exc)))
            self._logger.error(f"edge_case: {exc}")
            self._ctx.abort_episode(f"edge_case: {exc}")
            return static, dynamic
