"""The viewer layout of an env: the 3D scene, a 2D view per camera, a plot per scalar series."""

from __future__ import annotations

import rerun.blueprint as rrb
from arena_viz import DisplayKind

from rerun_utils.stream import TIMELINE

PLOT_WINDOW_S = 10.0
SIDE_KINDS = (DisplayKind.IMAGE, DisplayKind.IMU)


def side_view(kind: DisplayKind, entity: str, name: str) -> rrb.View:
    if kind == DisplayKind.IMAGE:
        return rrb.Spatial2DView(origin=entity, name=name)
    window = rrb.VisibleTimeRange(
        TIMELINE,
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-PLOT_WINDOW_S),
        end=rrb.TimeRangeBoundary.cursor_relative(),
    )
    return rrb.TimeSeriesView(origin=entity, name=name, time_ranges=window)


def blueprint(displays: list[tuple[DisplayKind, str, str]]) -> rrb.Blueprint:
    """The layout for the dispatched `(kind, entity, name)` displays, side panels collapsed."""
    scene = rrb.Spatial3DView(origin="/", name="scene")
    side = [side_view(kind, entity, name) for kind, entity, name in displays if kind in SIDE_KINDS]
    root = rrb.Horizontal(scene, rrb.Vertical(*side), column_shares=[3, 2]) if side else scene
    return rrb.Blueprint(root, collapse_panels=True)
