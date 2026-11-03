"""The task-mode vocabulary, importable without loading a mode."""

import enum


class Modes:
    @enum.unique
    class TM_Obstacles(enum.Enum):
        PARAMETRIZED = "parametrized"
        RANDOM = "random"
        SCENARIO = "scenario"
        ENVIRONMENT = "environment"
        PROMPT = "prompt"

        @classmethod
        def default(cls) -> "Modes.TM_Obstacles":
            return cls.RANDOM

    @enum.unique
    class TM_Robots(enum.Enum):
        GUIDED = "guided"
        EXPLORE = "explore"
        RANDOM = "random"
        SCENARIO = "scenario"
        DEMO = "demo"
        STATIONARY = "stationary"
        CHARACTERIZATION = "characterization"

        @classmethod
        def default(cls) -> "Modes.TM_Robots":
            return cls.EXPLORE

    @enum.unique
    class TM_Module(enum.Enum):
        STAGED = "staged"
        DYNAMIC_MAP = "dynamic_map"
        CLEAR_FORBIDDEN_ZONES = "clear_forbidden_zones"
        RVIZ_UI = "rviz_ui"
        SOUNDS = "sounds"
        ZONE_EDIT = "zone_edit"

        @classmethod
        def default(cls) -> set["Modes.TM_Module"]:
            return set()
