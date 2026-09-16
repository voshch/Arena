"""Patrol-with-service: the robot loops the scenario's checkpoints until a pedestrian's call reaches it.

The call only reaches the robot if the robot could perceive it: the pedestrian publishes the call clip,
stands within sensor range, and is not behind a wall. Nothing about the call is handed to the robot out of
band, so the gesture-off control arm (`tm_obstacles.scenario.gesture_mode=disabled`) is simply a scenario in
which no call is ever published and the pedestrian is served only if the patrol happens to pass them.
"""

from __future__ import annotations

import math

import numpy as np
from arena_people_msgs.msg import Pedestrians
from arena_rclpy_mixins.shared import Namespace
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import ScenarioGotoPhase
from rclpy.qos import qos_profile_sensor_data

from task_generator.manager.world_manager.utils import WorldOccupancy
from task_generator.shared import Orientation, Pose, Position
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, default_scenario
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, TaskRequest

_SCENARIO_NS = _REGISTRY_NAMESPACE("scenario")


def _short_name(name: str) -> str:
    """Pedestrians reach the topic namespaced by their environment (`env_0/caller`)."""
    return name.rsplit("/", 1)[-1]


class TM_Patrol(TM_Robots):
    """Loops the scenario's checkpoints; diverts to a pedestrian whose call it can perceive."""

    async def reset(self) -> None:
        await super().reset()

        world = WorldIdentifier(self._ctx.world_manager.loaded_world)
        zone_conv = self._ctx.world_manager.world_compacted().zone_converter(self.node.conf.General.RNG.stream("robots", "patrol"))
        name = self._scenario_file.value or default_scenario(self._ctx.world_manager.loaded_world)
        scenario = world.resolve_sync().scenario(name).resolve_sync().load(converter=zone_conv)

        self._route = []
        self._route_local = []
        self._index = 0
        self._current_goal = None
        self._walls_cache = None
        self._last_look = -1e9
        self._served_since = None
        self._visible_since = None
        self._diverted = False
        self._divert_pending = False
        self._last_pose = None
        self._service_goal = None
        self._peds = None

        robots = list(self._ctx.robots.values())
        for robot, config in zip(robots, scenario.robots, strict=False):
            self._start_poses[robot.name] = config.start
            self._scenario_start = config.start
            self._route_local = [phase.goto for phase in config.phase_list() if isinstance(phase, ScenarioGotoPhase)]
            # place the robot where the scenario says: without this it keeps whatever pose it was left in, and
            # every route and sightline in the scenario is measured from somewhere else
            await robot.move(config.start)
            break  # one robot patrols; any others are left to their own mode

        if self._subscription is None:
            self._subscription = self.node.create_subscription(
                Pedestrians,
                Namespace(self.node.get_namespace())("arena_peds"),
                self._on_peds,
                qos_profile_sensor_data,
            )

    def _adopt_frame(self, spawn_pose: Pose) -> None:
        """Take the route as the scenario wrote it, correcting only a genuine frame offset.

        The robot is moved to the scenario's start pose at reset, so the two should agree; a residual of more
        than a metre means the world reports poses in a different frame than its scenarios are written in, and
        the route is shifted by that much rather than left broken.
        """
        ox = spawn_pose.position.x - self._scenario_start.position.x
        oy = spawn_pose.position.y - self._scenario_start.position.y
        if math.hypot(ox, oy) < 1.0:  # the robot is where the scenario put it
            ox = oy = 0.0
        self._route = [
            Pose(position=Position(pose.position.x + ox, pose.position.y + oy, pose.position.z), orientation=pose.orientation)
            for pose in self._route_local
        ]
        self._logger.info(f"patrol route in the robot's frame, offset ({ox:.2f}, {oy:.2f}): {[(round(p.position.x, 1), round(p.position.y, 1)) for p in self._route]}")

    ARRIVED_M = 1.2
    LOOK_INTERVAL_S = 0.2

    def _next_checkpoint(self) -> TaskRequest:
        """One checkpoint per request, advanced on arrival rather than on the adapter's verdict.

        Re-submitting a goal makes the adapter re-dispatch it, which stops the robot for a moment; driving the
        loop off ``is_done`` cost roughly half the robot's average speed, so arrival is measured here instead.
        """
        self._current_goal = self._route[self._index % len(self._route)]
        self._index += 1
        return TaskRequest(phases=[GoToPhase(pose=self._current_goal)])

    def _on_peds(self, msg: Pedestrians) -> None:
        """Keep the latest pedestrians; everything else happens on the episode tick.

        The callback stays a pure assignment on purpose: doing the sightline and arrival work here left the
        robot without a dispatched goal for the whole episode.
        """
        self._peds = msg

    def _look(self, pose: Pose) -> None:
        """Latch arrival and a perceived call, from the tick."""
        here = (pose.position.x, pose.position.y)
        wanted = self._service_agent.value

        target = self._agent_position(wanted)
        if target is not None and math.dist(here, (target.x, target.y)) <= self._service_radius.value:
            if self._served_since is None:
                self._served_since = self._now()
                self._logger.info(f"reached {wanted or 'the caller'} at {math.dist(here, (target.x, target.y)):.2f} m")
        elif self._service_dwell.value > 0.0:
            self._served_since = None

        if self._diverted or not self._route:
            return
        calling = self._calling()
        if calling is None or not self._perceives(pose, calling[1]):
            self._visible_since = None
            return
        now = self._now()
        if self._visible_since is None:
            self._visible_since = now
        if now - self._visible_since >= self._recognition_dwell.value:
            self._service_goal = self._service_pose(pose, calling[1])
            self._divert_pending = True
            self._logger.info(f"call from {calling[0]} perceived at {math.dist(here, (calling[1].x, calling[1].y)):.1f} m")

    def _now(self) -> float:
        return self.node.sim_time.sec + self.node.sim_time.nanosec * 1e-9

    # -- perception ----------------------------------------------------------------------------

    def _calling(self) -> tuple[str, Position] | None:
        """The pedestrian publishing the call clip right now, if any."""
        msg = self._peds
        if msg is None:
            return None
        wanted = self._service_agent.value
        for ped in msg.pedestrians:
            if wanted and _short_name(ped.name) != wanted:
                continue
            if any(g.clip == self._clip.value for g in ped.gestures):
                return ped.name, Position(ped.pose.position.x, ped.pose.position.y, ped.pose.position.z)
        return None

    def _agent_position(self, name: str) -> Position | None:
        msg = self._peds
        if msg is None:
            return None
        for ped in msg.pedestrians:
            if not name or _short_name(ped.name) == name:
                return Position(ped.pose.position.x, ped.pose.position.y, ped.pose.position.z)
        return None

    def _walls(self) -> np.ndarray:
        """The wall layer as a boolean grid, resolved once.

        Rasterising the whole map per call is what this replaces: the check runs on every pedestrian message,
        and rebuilding a 694x510 mask at that rate starved the executor badly enough that the robot never
        received a goal at all.
        """
        if self._walls_cache is None:
            self._walls_cache = WorldOccupancy.fullish(self._ctx.world_manager.map.occupancy.walls)
        return self._walls_cache

    def _line_of_sight(self, a: Position, b: Position) -> bool:
        """False when a wall cell lies on the segment, so a call from inside a room does not carry through it."""
        world_map = self._ctx.world_manager.map
        walls = self._walls()
        steps = max(2, int(math.dist((a.x, a.y), (b.x, b.y)) / max(world_map.resolution, 1e-3)) + 1)
        xs = np.linspace(a.x, b.x, steps)
        ys = np.linspace(a.y, b.y, steps)
        rows, cols = [], []
        for x, y in zip(xs, ys, strict=True):
            r, c = world_map.tf_pos2grid(Position(float(x), float(y), 0.0))
            rows.append(int(r))
            cols.append(int(c))
        rows = np.clip(np.array(rows), 0, walls.shape[0] - 1)
        cols = np.clip(np.array(cols), 0, walls.shape[1] - 1)
        return not bool(walls[rows, cols].any())

    def _perceives(self, robot_pose: Pose, ped: Position) -> bool:
        if math.dist((robot_pose.position.x, robot_pose.position.y), (ped.x, ped.y)) > self._sensor_range.value:
            return False
        return not self._require_los.value or self._line_of_sight(robot_pose.position, ped)

    # -- episode -------------------------------------------------------------------------------

    @property
    async def done(self) -> bool:
        """Answer a perceived call, keep the loop running, and end the episode once the caller is served."""
        robots = list(self._ctx.robots.values())
        if not robots or not self._route_local:
            return False
        robot = robots[0]
        pose = robot.pose
        if pose is None:
            return False

        if not self._route:  # the first pose the robot reports fixes the frame, then the patrol starts
            self._adopt_frame(pose)
            await robot.submit_task(self._next_checkpoint())
            return False

        self._last_pose = pose
        self._look(pose)
        now = self._now()

        # the caller was reached, whatever brought the robot there
        if self._served_since is not None and now - self._served_since >= self._service_dwell.value:
            return True

        # a call the robot could perceive overrides the patrol, once
        if self._divert_pending and not self._diverted and self._service_goal is not None:
            await robot.submit_task(TaskRequest(phases=[GoToPhase(pose=self._service_goal)]))
            self._diverted = True
            self._divert_pending = False
            self._logger.info(f"goal overridden by the caller at ({self._service_goal.position.x:.1f}, {self._service_goal.position.y:.1f})")

        if not self._diverted and self._current_goal is not None:
            here = (pose.position.x, pose.position.y)
            goal = (self._current_goal.position.x, self._current_goal.position.y)
            if math.dist(here, goal) <= self.ARRIVED_M:
                await robot.submit_task(self._next_checkpoint())

        return False

    def _service_pose(self, robot_pose: Pose, ped: Position) -> Pose:
        """A pose service_standoff short of the pedestrian, on the robot's side, facing them."""
        dx, dy = ped.x - robot_pose.position.x, ped.y - robot_pose.position.y
        distance = math.hypot(dx, dy)
        yaw = math.atan2(dy, dx)
        standoff = min(self._service_standoff.value, distance)
        return Pose(
            position=Position(ped.x - math.cos(yaw) * standoff, ped.y - math.sin(yaw) * standoff, 0.0),
            orientation=Orientation.from_yaw(yaw),
        )

    def __init__(self, **kwargs: object) -> None:
        TM_Robots.__init__(self, **kwargs)
        self._subscription = None
        self._peds: Pedestrians | None = None
        self._route: list[Pose] = []
        self._route_local: list[Pose] = []
        self._index = 0
        self._current_goal: Pose | None = None
        self._walls_cache: np.ndarray | None = None
        self._last_look = -1e9
        self._scenario_start = Pose()
        self._service_goal: Pose | None = None
        self._divert_pending = False
        self._last_pose: Pose | None = None
        self._served_since: float | None = None
        self._visible_since: float | None = None
        self._diverted = False

        # resolved at reset, not here: the mode is built before a world is loaded, and there is no default then
        self._scenario_file = self.node.ROSParam[str](_SCENARIO_NS("file"), "")
        self._service_agent = self.node.ROSParam[str](self.namespace("service_agent"), "")
        self._clip = self.node.ROSParam[str](self.namespace("clip"), "beckon")
        self._service_radius = self.node.ROSParam[float](self.namespace("service_radius"), 1.5)
        self._service_dwell = self.node.ROSParam[float](self.namespace("service_dwell"), 2.0)
        self._service_standoff = self.node.ROSParam[float](self.namespace("service_standoff"), 1.0)
        self._sensor_range = self.node.ROSParam[float](self.namespace("sensor_range"), 12.0)
        self._require_los = self.node.ROSParam[bool](self.namespace("require_line_of_sight"), True)
        self._recognition_dwell = self.node.ROSParam[float](self.namespace("recognition_dwell"), 0.5)
