import asyncio

from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.assets.Animation import AnimationIdentifier
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import Scenario, ScenarioView
from arena_simulation_setup.utils.geometry import Position

from task_generator.manager.world_manager.utils import WorldOccupancy
from task_generator.shared import PositionRadius, Region
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles
from task_generator.tasks.registry import default_scenario


class TM_Scenario(TM_Obstacles):
    _config: ROSParamT[str]

    async def reset(self, *, seed: int) -> Obstacles:
        scenario_name = self._config.value
        # before any ped spawns, contact interactions read the mode when they form
        await self._ctx.environment_manager.configure_contact(self._contact_mode.value, self._standing_distance.value)
        await self._ctx.environment_manager.configure_gestures(self._gesture_mode.value)
        world_description = self._ctx.world_manager.world_compacted()

        safe_dist = self.node.conf.Obstacles.SAFE_DIST.value
        if safe_dist > 0:
            world_map = self._ctx.world_manager.map
            occupancy_grid = world_map.occupancy.grid
            rows, cols = occupancy_grid.shape

            def is_valid(pt: Position) -> bool:
                (lo_r, lo_c), (hi_r, hi_c) = world_map.tf_posr2rect(
                    PositionRadius(x=pt.x, y=pt.y, radius=safe_dist),
                )
                r0 = max(0, int(min(lo_r, hi_r)))
                r1 = min(rows, int(max(lo_r, hi_r)) + 1)
                c0 = max(0, int(min(lo_c, hi_c)))
                c1 = min(cols, int(max(lo_c, hi_c)) + 1)
                if r0 >= r1 or c0 >= c1:
                    return False
                return bool(WorldOccupancy.empty(occupancy_grid[r0:r1, c0:c1]).all())
        else:
            is_valid = None

        zone_conv = world_description.zone_converter(
            self.node.conf.General.RNG.stream("obstacles", "scenario"),
            is_valid=is_valid,
        )

        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(scenario_name).resolve_sync()
        scenario = scenario_view.load(converter=zone_conv)
        self.scenario = scenario
        await self._check_clips(scenario_view, scenario)

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

        self.node.register_timeline(scenario.timeline, seed)
        self.node.register_conditions(scenario.conditions)

        return scenario.static, scenario.dynamic

    async def _check_clips(self, scenario_view: ScenarioView, scenario: Scenario) -> None:
        """Resolve the clips the scenario's agents play, fetching bucket clips before the first agent needs one, and name any that will not play."""
        try:
            names = sorted({name for obstacle in scenario.dynamic for name in scenario_view.agent_clips(obstacle.extra.get("agent"))})
        except OSError as e:
            self._logger.warning(f"scenario {scenario_view.path.name}: could not read an agent file to check its clips: {e}")
            return
        found = await asyncio.gather(*(AnimationIdentifier.parse(name).resolve_path() for name in names), return_exceptions=True)
        missing = [name for name, path in zip(names, found, strict=True) if isinstance(path, Exception)]
        if missing:
            self._logger.warning(f"scenario {scenario_view.path.name}: animation clips not found, agents will skip them: {missing} (arena asset find animation <name>)")

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)
        self.scenario: Scenario | None = None

        self._config = self.node.ROSParam[str](
            self.namespace("file"),
            default_scenario(self._ctx.world_manager.loaded_world),
        )
        self._contact_mode = self.node.ROSParam[str](self.namespace("contact_mode"), "enabled")
        self._standing_distance = self.node.ROSParam[float](self.namespace("standing_distance"), 1.2)
        self._gesture_mode = self.node.ROSParam[str](self.namespace("gesture_mode"), "enabled")
