"""Gazebo GUI config for a fixed-size viewport."""

from __future__ import annotations

from pathlib import Path

import pytest
from arena_bringup.gazebo_gui import SPLIT_HANDLE_WIDTH, TOOLBAR_HEIGHT, fixed_viewport_config, parse_resolution

CHECKED_IN_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "gazebo" / "gui.config"


def test_parse_resolution_reads_width_and_height():
    assert parse_resolution("1920x1080") == (1920, 1080)
    assert parse_resolution(" 1280X720 ") == (1280, 720)


@pytest.mark.parametrize("value", ["", "1920", "1920x", "0x1080", "axb", "1920*1080"])
def test_parse_resolution_rejects_malformed_values(value):
    with pytest.raises(ValueError, match="expected WxH"):
        parse_resolution(value)


def test_fixed_viewport_config_sizes_window_for_the_viewport_plus_split_handle_and_toolbar():
    config = fixed_viewport_config(CHECKED_IN_CONFIG.read_text(), (1920, 1080))
    assert f"<width>{1920 + SPLIT_HANDLE_WIDTH}</width>" in config
    assert f"<height>{1080 + TOOLBAR_HEIGHT}</height>" in config


def test_fixed_viewport_config_drops_docked_panels_and_keeps_one_floating_viewport_plugin():
    config = fixed_viewport_config(CHECKED_IN_CONFIG.read_text(), (1920, 1080))
    assert "ComponentInspector" not in config
    assert "EntityTree" not in config
    assert config.count('name="Arena viewport camera"') == 1
    assert "docked" not in config.split('name="Arena viewport camera"')[1]
    assert config.count('key="state">docked') == 1


def test_fixed_viewport_config_keeps_the_scene_and_scene_plugins():
    config = fixed_viewport_config(CHECKED_IN_CONFIG.read_text(), (1920, 1080))
    assert 'name="3D View"' in config
    assert 'name="Scene Manager"' in config
