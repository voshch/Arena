"""rqt plugin shell: parses the target flags, ticks the panel at the stream rate."""

from __future__ import annotations

import argparse

from python_qt_binding.QtCore import QTimer
from rqt_gui_py.plugin import Plugin

from arena_cam.drive import TICK_HZ
from arena_cam.panel import Panel
from arena_cam.surfaces import TargetSelection

TICK_MS = int(1000.0 / TICK_HZ)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="arena_cam")
    parser.add_argument("--sim", action="store_true", help="drive the sim GUI camera")
    parser.add_argument("--viz", nargs="?", const="all", default=None, metavar="ENV_ID", help="drive rviz cameras: bare for all, or an env id for one")
    return parser.parse_args(argv)


def selection_from_args(args: argparse.Namespace) -> TargetSelection:
    """No flag drives everything, matching the CLI."""
    if not args.sim and args.viz is None:
        return TargetSelection(include_sim=True, viz_all=True, viz_env=None)
    if args.viz in (None, "all"):
        return TargetSelection(include_sim=args.sim, viz_all=args.viz == "all", viz_env=None)
    return TargetSelection(include_sim=args.sim, viz_all=False, viz_env=int(args.viz))


class CamSteering(Plugin):
    """rqt_gui_py plugin entry point, registered in plugin.xml."""

    def __init__(self, context: object) -> None:
        super().__init__(context)
        self.setObjectName("CamSteering")

        self._panel = Panel(selection_from_args(_parse_args(context.argv())))
        self._panel.setObjectName("CamSteeringUi")
        context.add_widget(self._panel)
        self._panel.attach(context.node)

        self._timer = QTimer()
        self._timer.timeout.connect(self._panel.tick)
        self._timer.start(TICK_MS)

    def shutdown_plugin(self) -> None:
        self._timer.stop()
        self._panel.detach()
