"""Pedestrian skeleton pose data shared by Arena simulators."""

from pathlib import Path

BONE_MAP_PATH = Path(__file__).with_name("bone_map.json")


def bone_map_path() -> str:
    return str(BONE_MAP_PATH)
