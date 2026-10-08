"""Backend for robot.hearing:=bus|srp|seld, the arena_hearing belief grid, speed-filter policy and front-ends."""

from __future__ import annotations

from arena_viz.kinds import DisplayKind
from arena_viz.style import StyleSpec
from task_generator_msgs.msg import AdapterDisplay

from task_generator.simulators.hearing import BaseHearing


class ArenaHearing(BaseHearing):
    def robot_displays(self, robot: str) -> tuple[AdapterDisplay, ...]:
        from arena_hearing.constants import BELIEF_GRID, BELIEF_WEDGES, SPEED_FILTER_MASK

        tg_ns = self.node.get_fully_qualified_name()
        return tuple(
            AdapterDisplay(
                name=name,
                topic=f"{tg_ns}/{robot}/{topic}",
                topic_type=topic_type,
                kind=kind,
                style_json=style,
                topic_must_exist=False,
                group="Hearing",
            )
            for name, topic, topic_type, kind, style in (
                ("Belief", BELIEF_GRID, "nav_msgs/OccupancyGrid", DisplayKind.MAP, StyleSpec(alpha=0.6, extra={"rviz": {"Color Scheme": "costmap", "Durability Policy": "Volatile"}}).to_json()),
                ("Speed Mask", SPEED_FILTER_MASK, "nav_msgs/OccupancyGrid", DisplayKind.MAP, StyleSpec(alpha=0.4, enabled=False, extra={"rviz": {"Color Scheme": "costmap"}}).to_json()),
                ("Wedges", BELIEF_WEDGES, "visualization_msgs/MarkerArray", DisplayKind.MARKER_ARRAY, StyleSpec(enabled=True).to_json()),
            )
        )
