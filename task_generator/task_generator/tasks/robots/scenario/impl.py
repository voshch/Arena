from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import ScenarioGesturePhase, ScenarioGotoPhase

from task_generator.shared import Pose, Position, PositionRadius
from task_generator.tasks.obstacles._validation import make_is_valid
from task_generator.tasks.registry import default_scenario
from task_generator.tasks.robots import TM_Robots
from task_generator.tasks.robots.request import GoToPhase, PlayGesturePhase, TaskRequest


class TM_Scenario(TM_Robots):
    _config: ROSParamT[str]

    def _spawn_clearance(self) -> float:
        """Free radius a robot start/goal must have, in metres.

        Not `Obstacles.SAFE_DIST`, which is the clearance a **pedestrian** needs: nav2's collision
        monitor stops on anything inside its stop polygon and its costmap inflates obstacles further,
        so a start placed to pedestrian tolerances is a start the robot cannot drive out of.
        `Robot.SPAWN_ROBOT_SAFE_DIST` plus the robot's radius (`RobotManager.safe_distance`) is the
        constant for this; the largest managed robot's value when robots exist, a conservative
        nominal radius otherwise.
        """
        managed = [r.safe_distance for r in self._ctx.robots.values()] if self._ctx.robots else []
        if managed:
            return max(managed)
        nominal_radius = 0.3
        return nominal_radius + self.node.conf.Robot.SPAWN_ROBOT_SAFE_DIST.value

    def _crowd_clearance(self, safe_distance: float) -> float:
        """Radius to forbid around a robot start/goal, in metres.

        `safe_distance` is what the ROBOT needs around its own centre. Placement only requires a
        pedestrian's **centre** to clear the disc, and the sampler erodes the occupancy by
        `Obstacles.SAFE_DIST`, so an agent can legally land at `safe_distance + SAFE_DIST` and put
        its body radius back toward the robot. Adding the pedestrian's extent makes the disc mean
        what `_spawn_clearance` claims: a start the robot can drive out of. This bounds the crowd at
        spawn only; pedestrians walk.
        """
        return safe_distance + self.node.conf.Obstacles.PEDESTRIAN_BODY_RADIUS.value

    async def reset(self) -> None:
        await super().reset()

        world_description = self._ctx.world_manager.world_compacted()
        is_valid = make_is_valid(self._ctx.world_manager.map, self._spawn_clearance())
        zone_conv = world_description.zone_converter(
            self.node.conf.General.RNG.stream("robots", "scenario"),
            is_valid=is_valid,
        )
        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(self._config.value).resolve_sync()
        SCENARIO_ROBOTS = scenario_view.load(converter=zone_conv).robots

        managed_robots = list(self._ctx.robots.values())

        scenario_robots_length = len(SCENARIO_ROBOTS)
        setup_robot_length = len(managed_robots)

        if setup_robot_length > scenario_robots_length:
            managed_robots = managed_robots[:scenario_robots_length]
            self._logger.warn("Robot setup contains more robots than the scenario file.", once=True)

        if scenario_robots_length > setup_robot_length:
            SCENARIO_ROBOTS = SCENARIO_ROBOTS[:setup_robot_length]
            self._logger.warn("Scenario file contains more robots than setup.", once=True)

        # Shift floor-tagged poses from level-local to the flattened map frame.
        level_origins = self._ctx.world_manager.map.level_origins

        def _to_map_frame(pose: Pose, floor: str) -> Pose:
            if not floor:
                return pose
            ox, oy = level_origins.get(floor, (0.0, 0.0))
            return Pose(
                position=Position(pose.position.x + ox, pose.position.y + oy, pose.position.z),
                orientation=pose.orientation,
            )

        for robot, config in zip(managed_robots, SCENARIO_ROBOTS, strict=False):
            start_pose = _to_map_frame(config.start, config.start_floor)
            self._start_poses[robot.name] = start_pose

            phases: list[GoToPhase | PlayGesturePhase] = []
            crowd_clearance = self._crowd_clearance(robot.safe_distance)
            forbidden: list[PositionRadius] = [
                PositionRadius(x=start_pose.position.x, y=start_pose.position.y, radius=crowd_clearance),
            ]
            for phase in config.phase_list():
                if isinstance(phase, ScenarioGotoPhase):
                    goto_pose = _to_map_frame(phase.goto, config.goal_floor)
                    phases.append(GoToPhase(pose=goto_pose))
                    forbidden.append(PositionRadius(x=goto_pose.position.x, y=goto_pose.position.y, radius=crowd_clearance))
                elif isinstance(phase, ScenarioGesturePhase):
                    phases.append(PlayGesturePhase(gesture=None if phase.gesture in ("", "random") else phase.gesture, instance=phase.instance))

            await robot.submit_task(TaskRequest(phases=phases))
            self._ctx.world_manager.forbid(forbidden)

    def __init__(self, **kwargs: object) -> None:
        TM_Robots.__init__(self, **kwargs)

        self._config = self.node.ROSParam[str](
            self.namespace('file'),
            default_scenario(self._ctx.world_manager.loaded_world),
        )
