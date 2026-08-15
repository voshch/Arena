from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import WorldIdentifier
from arena_simulation_setup.utils.geometry import Position

from task_generator.manager.world_manager.utils import WorldOccupancy
from task_generator.shared import PositionRadius, Region
from task_generator.tasks import identifier_to_available
from task_generator.tasks.obstacles import Obstacles, TM_Obstacles


class TM_Scenario(TM_Obstacles):
    _config: ROSParamT[str]

    async def reset(self, **kwargs: object) -> Obstacles:
        scenario_name = self._config.value
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

        self.node.register_timeline(scenario.timeline, int(kwargs.get("seed", -1)))
        self.node.register_conditions(scenario.conditions)

        return scenario.static, scenario.dynamic

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        default_scenario: str | None = 'default'
        if default_scenario not in (scenarios := list(identifier_to_available(WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().scenario))):
            default_scenario = next(iter(scenarios), None)
        if default_scenario is None:
            raise ValueError(f"No scenarios found in world {self._ctx.world_manager.loaded_world}")

        self._config = self.node.ROSParam[str](
            self.namespace("file"),
            default_scenario,
        )
