from __future__ import annotations

import importlib

from arena_rclpy_mixins.params import Choices, Param, ParamGroup
from arena_runtime.env.task_api import Constants

from task_generator.modes import Modes
from task_generator.tasks.registry import _REGISTRY_NAMESPACE, OBSTACLES_MODES

NS = _REGISTRY_NAMESPACE("prompt")
_PROMPT_MODULES = {Constants.HumanSimulator.ARENA: "arena", Constants.HumanSimulator.HUNAV: "hunav"}


class PromptParams(ParamGroup):
    USER_PROMPT = Param[str]("user_prompt", "An empty space with no pedestrian.", "Natural-language prompt describing the desired crowd.")
    TOP_P = Param[float]("top_p", 0.3, "Nucleus sampling top-p for LLM generation.")
    GENERATION_MODE = Param[str](
        "generation_mode",
        "arena",
        "Generation mode. Under human:=arena: 'arena' generates waypoint-following agents, 'behavior' additionally generates arena_humansim agent-type behavior definitions. Under human:=hunav: 'behavior_tree' and 'crowded_behavior_tree' generate HuNav behaviour trees.",
        constraint=Choices("arena", "behavior", "behavior_tree", "crowded_behavior_tree"),
    )


def register_prompt_mode(human: Constants.HumanSimulator) -> None:
    """Back TM_Obstacles.PROMPT with the prompt implementation of the human simulator."""
    mode = _PROMPT_MODULES.get(human)
    if mode is None or Modes.TM_Obstacles.PROMPT in OBSTACLES_MODES:
        return

    @OBSTACLES_MODES.register(Modes.TM_Obstacles.PROMPT, namespace=NS, schema=PromptParams)
    def _prompt() -> type:
        return importlib.import_module(f"{__name__}.{mode}").TM_Prompt
