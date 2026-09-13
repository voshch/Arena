from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.tree.World.Scenario import Scenario

from task_generator.shared import Region
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles
from task_generator.tasks.obstacles._validation import make_is_valid
from task_generator.tasks.registry import default_scenario


class TM_Scenario(TM_Obstacles):
    _config: ROSParamT[str]

    async def reset(self, *, seed: int) -> Obstacles:
        scenario_name = self._config.value
        world_description = self._ctx.world_manager.world_compacted()

        # A zone reference resolves to a point a pedestrian STANDS at, so the disc that must be clear
        # is its body, not its centre; SAFE_DIST alone is a sampling margin.
        safe_dist = (
            self.node.conf.Obstacles.SAFE_DIST.value
            + self.node.conf.Obstacles.PEDESTRIAN_BODY_RADIUS.value
        )
        is_valid = make_is_valid(self._ctx.world_manager.map, safe_dist)

        zone_conv = world_description.zone_converter(
            self.node.conf.General.RNG.stream("obstacles", "scenario"),
            is_valid=is_valid,
        )

        scenario_view = WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario(scenario_name).resolve_sync()
        scenario = scenario_view.load(converter=zone_conv)
        self.scenario = scenario

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
        # Recorded, not pushed: `Task._reset_episode` applies these once, after every mode
        # that wraps this one has had a chance to see them. See TM_Obstacles.pending_regions.
        self.pending_regions[:] = regions

        self.node.register_timeline(scenario.timeline, seed)
        self.node.register_conditions(scenario.conditions)

        return scenario.static, scenario.dynamic

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)
        self.scenario: Scenario | None = None

        self._config = self.node.ROSParam[str](
            self.namespace("file"),
            default_scenario(self._ctx.world_manager.loaded_world),
        )
