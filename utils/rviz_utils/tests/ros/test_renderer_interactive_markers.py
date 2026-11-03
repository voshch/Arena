from __future__ import annotations

from arena_viz import DisplayKind, StyleSpec
from arena_runtime_msgs.msg import AdapterDisplay

from rviz_utils.renderers import REGISTRY


def _display(enabled: bool) -> AdapterDisplay:
    return AdapterDisplay(
        name="Handles",
        topic="/env_0/arena_env_node/markers",
        topic_type="visualization_msgs/InteractiveMarkerUpdate",
        kind=DisplayKind.INTERACTIVE_MARKERS.value,
        style_json=StyleSpec(enabled=enabled).to_json(),
        topic_must_exist=False,
        group="Interaction",
    )


def test_interactive_markers_display_points_at_server_namespace() -> None:
    result = REGISTRY[DisplayKind.INTERACTIVE_MARKERS](_display(True), None)
    assert result is not None
    assert result["Class"] == "rviz_default_plugins/InteractiveMarkers"
    assert result["Interactive Markers Namespace"] == "/env_0/arena_env_node/markers"
    assert result["Enabled"] is True


def test_interactive_markers_display_honors_disabled_style() -> None:
    result = REGISTRY[DisplayKind.INTERACTIVE_MARKERS](_display(False), None)
    assert result is not None
    assert result["Enabled"] is False
