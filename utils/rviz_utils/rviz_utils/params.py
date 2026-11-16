"""Node parameters of the rviz config generator."""

from __future__ import annotations

from arena_rclpy_mixins.params import Choices, Param

VIEW = Param[str]('view', 'map', 'Camera view of the generated config.', constraint=Choices('map', 'robot', 'robot3p'))
ROBOT = Param[int]('robot', 0, 'Index of the robot in the fleet that the robot views follow (negative counts from the end).')
GENERATOR_PARAMS = (VIEW, ROBOT)
