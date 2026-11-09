"""Parameters of how a world is loaded, shared by every node and launch file that loads one."""

from __future__ import annotations

from arena_rclpy_mixins.params import Choices, Param

WORLD_LIGHTING_MODES = ('authored', 'auto')
WORLD_LEVELS_MODES = ('auto', 'stack', 'spread')
DERIVE_MODES = ('auto', 'all', 'off')

WORLD_LIGHTING = Param[str]('world.lighting', 'authored', 'Lighting mode of the world.', constraint=Choices(*WORLD_LIGHTING_MODES))
WORLD_LEVELS = Param[str]('world.levels', 'auto', 'Level layout of the world in the simulator.', constraint=Choices(*WORLD_LEVELS_MODES))
KG_DERIVE = Param[str]('kg.derive', 'auto', 'Which knowledge graph facts are derived from geometry.', constraint=Choices(*DERIVE_MODES))
