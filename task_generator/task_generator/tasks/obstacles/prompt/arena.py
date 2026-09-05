"""Prompt-based (LLM) obstacle generation targeting arena_humansim."""

import json
import math
import os
import re
import tempfile
import time
from enum import Enum

import attrs
import yaml
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import LevelDescription, WorldIdentifier
from arena_simulation_setup.utils.cattrs import converter

from task_generator.tasks.obstacles import (
    DynamicObstacle,
    Obstacle,
    TM_Obstacles,
)
from task_generator.utils.gpt import genai

from .prompt_utils import (
    ARENA_FORMAT,
    BEHAVIOR_FORMAT,
    REMOTE_LM,
    SYSTEM_INSTRUCTION,
)

DEBUG: bool = os.environ.get("ARENA_DEBUG", "0").lower() in ("1", "true")

_BUILTIN_AGENT_TYPES = ("adult", "elder")

_WAYPOINT_MODES = ("repeat", "reverse", "once", "random")


@attrs.define()
class _ParsedConfig:
    static: list[Obstacle]
    dynamic: list[DynamicObstacle]


class GenerationMode(Enum):
    ARENA = "arena"
    BEHAVIOR = "behavior"

    @classmethod
    def has_value(cls, value: object) -> bool:
        return value in cls._value2member_map_


@attrs.define()
class PromptConfig:
    user_prompt: ROSParamT[str]
    top_p: ROSParamT[float]
    generation_mode: ROSParamT[str]


class TM_Prompt(TM_Obstacles):
    """
    Prompt task generator for obstacles, backed by arena_humansim.

    Generates pedestrian agents from a natural-language prompt. Two modes:

    - ``arena``: agents are plain waypoint followers (``agent_type`` is one of
      the built-in arena_humansim types).
    - ``behavior``: the LLM additionally emits arena_humansim agent-type
      definitions (behavior-tree sequences: go_to steps, interactions such as
      TALK_TO / QUEUE_USE / SERVICE, needs). Definitions are written to
      scenario-local YAML files, validated with the arena_humansim loader, and
      referenced per-agent.
    """

    _config: PromptConfig

    def preprocess_world_description(self, world_description: LevelDescription) -> str:
        """
        Preprocesses the floor description, keeps corners and walls only and converts them to 2D format.

        Args:
            world_description : LevelDescription
                The level description to preprocess.

        Returns:
            parsed : str
                The preprocessed JSON formatted str world description.
        """
        parsed = {}

        parsed["zones"] = []
        for zone in world_description.zones:
            parsed_zone = {
                "name": zone.name,
                "corners": [[corner.x, corner.y] for corner in zone.corners],
                "walls": [[[wall.start.x, wall.start.y], [wall.end.x, wall.end.y]] for wall in zone.walls],
                "entities": [
                    {
                        "name": entity.name,
                        "model": entity.model.serialize(),
                        "pose": [
                            entity.pose.position.x,
                            entity.pose.position.y,
                            math.degrees(entity.pose.orientation.to_yaw()),  # LLM context uses degrees (see context.py)
                        ],
                    }
                    for entity in zone.entities.static
                ],
            }
            parsed["zones"].append(parsed_zone)

        return json.dumps(parsed)

    # LLM output post-processing
    # --------------------------

    @staticmethod
    def _pose_deg_to_rad(pose: object) -> list[float]:
        values = [float(v) for v in pose] if isinstance(pose, (list, tuple)) else [0.0, 0.0, 0.0]
        while len(values) < 3:
            values.append(0.0)
        values[2] = math.radians(values[2])
        return values[:3]

    @classmethod
    def _normalize_waypoint_mode(cls, raw: object) -> str:
        if isinstance(raw, int) and 0 <= raw < len(_WAYPOINT_MODES):
            return _WAYPOINT_MODES[raw]
        if isinstance(raw, str) and raw.lower() in _WAYPOINT_MODES:
            return raw.lower()
        return "repeat"

    def _materialize_agent_types(self, agent_types: dict) -> dict[str, str]:
        """
        Write LLM-generated arena_humansim agent-type definitions to
        scenario-local YAML files and validate them with the arena_humansim
        loader. Returns a mapping of generated type name -> absolute file path
        for the definitions that validated; invalid ones are dropped (agents
        referencing them fall back to the built-in default).
        """
        from arena_humansim.core.agents.loader import load_agent_type_from_file

        paths: dict[str, str] = {}
        for name, definition in agent_types.items():
            if not isinstance(definition, dict):
                self._logger.warn(f"Generated agent type {name!r} is not a mapping, skipping.")
                continue

            safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", str(name))
            definition = dict(definition)
            definition["name"] = safe_name
            definition.setdefault("extends", "adult")
            if "sequences" in definition:
                definition.setdefault("mode", "behavior_tree")

            path = os.path.join(self.tmp_dir.name, f"{safe_name}.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(definition, f)

            try:
                load_agent_type_from_file(path)
            except Exception as e:
                self._logger.warn(f"Generated agent type {name!r} failed validation and is dropped: {e}")
                os.unlink(path)
                continue

            paths[str(name)] = path

        return paths

    def llm_output_to_config(self, llm_output: dict, generation_mode: str) -> dict:
        """
        Convert raw LLM output into an obstacle config consumable by
        ``ArenaHumanDynamicObstacle`` (yaw converted to radians, ``agent``
        block and ``waypoint_mode`` extras attached).
        """
        try:
            agent_type_paths: dict[str, str] = {}
            if generation_mode == GenerationMode.BEHAVIOR.value:
                generated_types = llm_output.get("agent_types") or {}
                if not generated_types:
                    self._logger.warn("behavior mode: LLM output contained no agent_types; agents fall back to built-in types.")
                agent_type_paths = self._materialize_agent_types(generated_types)

            config = {"obstacles": {"static": [], "dynamic": []}}

            obstacles = llm_output.get("obstacles") or {}
            for raw_agent in obstacles.get("dynamic") or []:
                agent = dict(raw_agent)

                agent["pos"] = self._pose_deg_to_rad(agent.get("pos", agent.get("pose")))
                agent.pop("pose", None)
                agent["waypoints"] = [self._pose_deg_to_rad(wp) for wp in agent.get("waypoints") or []]
                agent["waypoint_mode"] = self._normalize_waypoint_mode(agent.get("waypoint_mode"))

                agent_type = str(agent.pop("agent_type", agent.pop("type", "adult")) or "adult")
                agent_type = agent_type_paths.get(agent_type, agent_type)
                if os.sep not in agent_type and agent_type not in _BUILTIN_AGENT_TYPES:
                    self._logger.warn(f"Unknown agent type {agent_type!r} for agent {agent.get('name')!r}, falling back to 'adult'.")
                    agent_type = "adult"

                agent_block: dict = {"agent_type": agent_type}
                desired_velocity = agent.pop("desired_velocity", None)
                if desired_velocity is not None:
                    agent_block["desired_velocity"] = float(desired_velocity)
                agent["agent"] = agent_block

                config["obstacles"]["dynamic"].append(agent)

        except Exception as e:
            self._logger.error(f"Failed to parse agents from LLM response: {e}")
            self._logger.error("Returning empty config!")
            config = {}

        return config

    # LLM inference
    # -------------

    def _context_for_mode(self, generation_mode: str) -> str:
        if generation_mode == GenerationMode.BEHAVIOR.value:
            return BEHAVIOR_FORMAT
        return ARENA_FORMAT

    async def _prompt_to_config(self, prompt: str, top_p: float, generation_mode: str) -> dict:
        world_info = self.preprocess_world_description(self._ctx.world_manager.world_compacted())

        pipeline_start = time.time()

        if generation_mode not in self.cached_context_name:
            cache = self.inference_client.caches.create(
                model=REMOTE_LM,
                config=genai.types.CreateCachedContentConfig(
                    display_name=generation_mode + "_context",
                    system_instruction=SYSTEM_INSTRUCTION,
                    contents=self._context_for_mode(generation_mode),
                ),
            )
            if cache.name is not None:
                self.cached_context_name.update({generation_mode: cache.name})

        message = f"Generate pedestrian agents data for a simulation where: {prompt}. Generate data base on this world data as below <WORLD_DESCRIPTION>: {world_info}. Only return valid JSON using the format declared in the system context, with no explanation, thoughts, or extra text."
        if generation_mode == GenerationMode.BEHAVIOR.value:
            message += ' Define every non-built-in agent_type referenced by the agents in the top-level "agent_types" object.'

        self._logger.info("Start inference...")
        start = time.time()
        response = await self.inference_client.aio.models.generate_content(
            model=REMOTE_LM,
            contents=[message],
            config=genai.types.GenerateContentConfig(
                cached_content=self.cached_context_name[generation_mode],
                top_p=top_p,
                temperature=0.2,
                top_k=40,
                thinking_config=genai.types.ThinkingConfig(thinking_level="low"),
            ),
        )

        answer = response.text
        self._logger.debug(f"LLM raw output for the prompt: {prompt}")
        self._logger.debug(answer)
        end = time.time()
        self._logger.info(f"Inference done, took: {end - start:.1f}s")

        assert answer is not None
        answer = answer.strip()
        if answer.startswith("```"):
            answer = answer.split("\n", 1)[1] if "\n" in answer else answer[3:]
            if answer.endswith("```"):
                answer = answer[:-3]
            answer = answer.strip()

        config = self.llm_output_to_config(json.loads(answer), generation_mode)

        pipeline_end = time.time()
        self._logger.info(f"Generation pipeline took: {pipeline_end - pipeline_start:.1f}s")

        if DEBUG:
            with tempfile.NamedTemporaryFile(
                delete=False,
                prefix="scenario_",
                suffix=".json",
                dir=self.tmp_dir.name,
                mode="w",
            ) as file:
                json.dump(config, file, indent=2)
                self._logger.debug(f"Saved parsed prompt result to {file.name}")

        return config

    async def _parse_prompt(self, prompt: str, top_p: float, generation_mode: str) -> _ParsedConfig:
        """
        Parses the prompt to generate obstacles config.

        Args:
            prompt (str): The prompt for generating obstacles config.

        Returns:
            _ParsedConfig: Parsed configuration containing static and dynamic obstacles.
        """
        assert GenerationMode.has_value(generation_mode)
        config = await self._prompt_to_config(prompt, top_p, generation_mode)

        static_obstacles: list[Obstacle] = []
        dynamic_obstacles = [obs for obs in config.get("obstacles", {}).get("dynamic", [])]

        result = converter.structure(dict(static=static_obstacles, dynamic=dynamic_obstacles), _ParsedConfig)

        return result

    async def reset(self, *, seed: int) -> tuple[list[Obstacle], list[DynamicObstacle]]:
        parsed_config = await self._parse_prompt(
            self._config.user_prompt.value,
            self._config.top_p.value,
            self._config.generation_mode.value,
        )

        return parsed_config.static, parsed_config.dynamic

    def __init__(self, **kwargs: object) -> None:
        TM_Obstacles.__init__(self, **kwargs)

        self._config = PromptConfig(
            user_prompt=self.node.ROSParam[str](
                self.namespace("user_prompt"),
                value="An empty space with no pedestrian.",
            ),
            top_p=self.node.ROSParam[float](
                self.namespace("top_p"),
                value=0.3,
            ),
            generation_mode=self.node.ROSParam[str](
                self.namespace("generation_mode"),
                value=GenerationMode.ARENA.value,
            ),
        )

        self.inference_client = genai.Client()

        try:
            caches = self.inference_client.caches.list()
            if caches:
                for cache in caches:
                    self.inference_client.caches.delete(name=cache.name)
        except Exception as e:
            print(e)

        self.cached_context_name: dict[str, str] = {}  # Whether the prompt context need to be changed and fed into LLM model

        self.tmp_dir = tempfile.TemporaryDirectory(
            dir=os.path.join(
                WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().path,
                "scenarios",
            )
        )  # Temporary directory to store generated agent-type YAML files

    def __del__(self):
        # May run on a half-constructed instance if __init__ raised (e.g. missing API key)
        try:
            for cache_name in getattr(self, "cached_context_name", {}).values():
                self.inference_client.caches.delete(name=cache_name)
            self.cached_context_name = {}
        except Exception as e:
            logger = getattr(self, "_logger", None)
            if logger is not None:
                logger.error(str(e))
                logger.error("Can not delete cache! Maybe it was deleted earlier.")
