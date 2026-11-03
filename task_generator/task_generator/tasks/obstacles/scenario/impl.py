import asyncio
import typing

import attrs
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_runtime.env.task_api import Stop
from arena_runtime.shared import WORLD_ENTITY_PREFIX, DynamicObstacle, Position, Region
from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier
from arena_simulation_setup.tree.World import WorldIdentifier, anchor_light, point_scenario_assets
from arena_simulation_setup.tree.World.Scenario import Scenario, ScenarioView, articulation_problems

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles
from task_generator.tasks.obstacles._itinerary import attach_itinerary
from task_generator.tasks.obstacles._validation import make_is_valid
from task_generator.tasks.registry import default_scenario


class TM_Scenario(TM_Obstacles):
    _config: ROSParamT[str]

    async def reset(self, *, seed: int) -> Obstacles:
        scenario_name = self._config.value
        await self._ctx.environment_manager.configure_contact(self._contact_mode.value, self._standing_distance.value)
        await self._ctx.environment_manager.configure_gestures(self._gesture_mode.value)
        world_description = self._ctx.world_manager.world_compacted()

        is_valid = make_is_valid(self._ctx.world_manager.map, self._ctx.conf.SAFE_DIST.value)

        zone_conv = world_description.zone_converter(
            self._ctx.conf.RNG.stream("obstacles", "scenario"),
            is_valid=is_valid,
        )

        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(scenario_name).resolve_sync()
        point_scenario_assets(scenario_view.path)
        scenario = scenario_view.load(converter=zone_conv)
        self.scenario = scenario
        self.lights = [anchor_light(light, scenario.objects) for light in scenario.lights]
        await self._check_clips(scenario_view, scenario)
        self._ctx.environment_manager.check_stimuli(scenario_view.path.name, scenario.humans, scenario.timeline)
        await self._check_articulations(scenario_view, scenario)

        regions = [
            Region(
                name=name,
                type=r.type,
                polygon=list(r.polygon),
                config=r.config,
                included_from=scenario_view.path,
            )
            for name, r in scenario.regions.items()
        ]
        await self._ctx.environment_manager.setup_regions(regions)

        self._ctx.episode.register_timeline(scenario.timeline, seed)
        self._ctx.episode.register_conditions(scenario.conditions)

        return scenario.objects, [self._with_itinerary(obstacle) for obstacle in scenario.humans]

    def _with_itinerary(self, obstacle: DynamicObstacle) -> DynamicObstacle:
        """Attach waypoint dwell and explicit elevator rides."""
        wm = self._ctx.world_manager
        realizer = self._ctx.realizer

        def elevator_ride(name: str, toward: Position | None) -> Stop | None:
            try:
                hop = wm.elevator_hop(name, toward=toward)
            except ValueError as e:
                self._logger.warning(f"{obstacle.name}: {e}")
                return None
            return Stop(x=hop.board_position.x, y=hop.board_position.y, ride=True, elevator=hop.board_name, target=hop.target)

        try:
            return attach_itinerary(
                obstacle,
                level_origin=realizer.get_level_origin,
                placement=lambda zone, level: realizer.placement_level(zone, level, obstacle.level),
                elevator_ride=elevator_ride,
                warn=self._logger.warning,
            )
        except (KeyError, ValueError) as e:
            self._logger.error(f"{obstacle.name}: itinerary ignored: {e}")
            return obstacle

    async def _check_clips(self, scenario_view: ScenarioView, scenario: Scenario) -> None:
        """Resolve the clips the scenario's agents play and warn about the ones that will not."""
        try:
            names = sorted({name for obstacle in scenario.humans for name in scenario_view.agent_clips(obstacle.agent)})
        except OSError as e:
            self._logger.warning(f"scenario {scenario_view.path.name}: could not read an agent file to check its clips: {e}")
            return
        found = await asyncio.gather(*(AnimationIdentifier.parse(name).resolve_path() for name in names), return_exceptions=True)
        missing = [name for name, path in zip(names, found, strict=True) if isinstance(path, Exception)]
        if missing:
            self._logger.warning(f"scenario {scenario_view.path.name}: animation clips not found, agents will skip them: {missing} (arena asset find animation <name>)")

    async def _check_articulations(self, scenario_view: ScenarioView, scenario: Scenario) -> None:
        """Raise ValueError naming the scenario for `joints`, condition fields and timeline sets its or the world's articulated objects refuse, once per world and scenario."""
        key = (str(self._ctx.world_manager.loaded_world), str(scenario_view.path))
        if key in self._articulations_checked:
            return
        world_objects = [attrs.evolve(obj, name=obj.name.removeprefix(WORLD_ENTITY_PREFIX)) for level in self._ctx.world_manager.world.levels.values() for obj in level.objects]
        problems = await asyncio.to_thread(articulation_problems, scenario, world_objects)
        if problems:
            raise ValueError(f"scenario {scenario_view.path.name!r}: {'; '.join(problems)}")
        self._articulations_checked.add(key)

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
        TM_Obstacles.__init__(self, **kwargs)
        self.scenario: Scenario | None = None
        self._articulations_checked: set[tuple[str, str]] = set()

        self._config = self.node.ROSParam[str](
            self.namespace("file"),
            default_scenario(self._ctx.world_manager.loaded_world),
        )
        self._contact_mode = self.node.ROSParam[str](self.namespace("contact_mode"), "enabled")
        self._standing_distance = self.node.ROSParam[float](self.namespace("standing_distance"), 1.2)
        self._gesture_mode = self.node.ROSParam[str](self.namespace("gesture_mode"), "enabled")
