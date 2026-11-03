"""DisplayKind.PEDESTRIANS renderer: stub pending rerun HRI integration.

TODO: subscribe to {env}/humans/bodies/tracked (hri_msgs/IdsList) and log
per-body poses from TF; rerun has no hri plugin equivalent of hri_rviz/Skeletons3D.
"""

from __future__ import annotations

from arena_runtime_msgs.msg import AdapterDisplay, RobotDescriptor
from arena_viz import DisplayKind

from rerun_utils.renderers._registry import RendererCtx, register


@register(DisplayKind.PEDESTRIANS)
def render_pedestrians(d: AdapterDisplay, robot: RobotDescriptor | None, ctx: RendererCtx) -> None:
    pass
