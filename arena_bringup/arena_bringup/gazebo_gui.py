"""Gazebo GUI layout for a fixed-size viewport."""

from __future__ import annotations

import re

TOOLBAR_HEIGHT = 48
SPLIT_HANDLE_WIDTH = 1

_PANEL_NAMES = ("Component inspector", "Entity tree", "Arena viewport camera")
_FLOATING_VIEWPORT_PLUGIN = """\
<plugin filename="ViewportCamera" name="Arena viewport camera">
  <gz-gui>
    <property type="bool" key="showTitleBar">false</property>
    <property type="string" key="state">floating</property>
    <property type="double" key="width">5</property>
    <property type="double" key="height">5</property>
  </gz-gui>
</plugin>
"""


def parse_resolution(value: str) -> tuple[int, int]:
    """Parse WxH into positive integers."""
    match = re.fullmatch(r"(\d+)x(\d+)", value.strip().lower())
    if match is None or not all(int(g) > 0 for g in match.groups()):
        raise ValueError(f"viewport.resolution: expected WxH, got {value!r}")
    return int(match.group(1)), int(match.group(2))


def fixed_viewport_config(content: str, resolution: tuple[int, int]) -> str:
    """Resize the window and drop the docked panels so the 3D view is exactly the resolution."""
    width, height = resolution
    for name in _PANEL_NAMES:
        content = re.sub(rf'<plugin\b[^>]*\bname="{re.escape(name)}"[^>]*>.*?</plugin>\s*', "", content, flags=re.DOTALL)
    content = re.sub(r"<width>\d+</width>", f"<width>{width + SPLIT_HANDLE_WIDTH}</width>", content, count=1)
    content = re.sub(r"<height>\d+</height>", f"<height>{height + TOOLBAR_HEIGHT}</height>", content, count=1)
    return content.rstrip() + "\n\n" + _FLOATING_VIEWPORT_PLUGIN
