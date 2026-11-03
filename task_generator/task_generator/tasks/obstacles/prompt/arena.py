"""Prompt-based (LLM) obstacle generation targeting arena_humansim."""

import json
import math
import os
import re
import tempfile
import time
import typing
from enum import Enum

import attrs
import yaml
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_runtime.shared import WORLD_ENTITY_PREFIX
from arena_simulation_setup.kg import Scene
from arena_simulation_setup.shared.entities import WAYPOINT_MODES
from arena_simulation_setup.tree.World import Level, WorldIdentifier, describe, zones_of
from arena_simulation_setup.tree.World.Scenario import SCENARIO_FORMAT, Scenario, parse_scenario
from arena_simulation_setup.utils.llm import genai

from task_generator.tasks.mode import TaskModeKwargs
from task_generator.tasks.obstacles import (
    DynamicObstacle,
    Obstacle,
    TM_Obstacles,
)

from .prompt_utils import (
    ARENA_FORMAT,
    BEHAVIOR_FORMAT,
    REMOTE_LM,
    SYSTEM_INSTRUCTION,
)

DEBUG: bool = os.environ.get("ARENA_DEBUG", "0").lower() in ("1", "true")

_BUILTIN_AGENT_TYPES = ("adult", "elder")


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

    def preprocess_world_description(self, level: Level, semantics: Scene) -> str:
        """The compacted level as JSON for the LLM: zones with corners, bordering walls, description and contained entities, then the entities outside every zone."""

        def entity(obstacle: Obstacle) -> dict:
            return {
                "name": obstacle.name,
                "model": obstacle.model.serialize(),
                "pose": [
                    obstacle.pose.position.x,
                    obstacle.pose.position.y,
                    math.degrees(obstacle.pose.orientation.to_yaw()),  # LLM context uses degrees (see context.py)
                ],
            }

        zone_walls: dict[str, list] = {zone.name: [] for zone in level.zones}
        for wall in level.all_walls:
            for zone in level.adjacent_zones(wall.start, wall.end):
                zone_walls[zone.name].append([[wall.start.x, wall.start.y], [wall.end.x, wall.end.y]])

        zone_entities: dict[str, list[dict]] = {zone.name: [] for zone in level.zones}
        unzoned: list[dict] = []
        for obstacle in level.objects:
            rooms = [name for name in zones_of(semantics, f"obj:{obstacle.name.removeprefix(WORLD_ENTITY_PREFIX)}") if name in zone_entities]
            for room in rooms:
                zone_entities[room].append(entity(obstacle))
            if not rooms:
                unzoned.append(entity(obstacle))

        parsed: dict = {"zones": []}
        for zone in level.zones:
            parsed_zone = {
                "name": zone.name,
                "corners": [[corner.x, corner.y] for corner in zone.polygon],
                "walls": zone_walls[zone.name],
                "entities": zone_entities[zone.name],
            }
            if description := describe(semantics, zone.name):
                parsed_zone["description"] = description
            parsed["zones"].append(parsed_zone)
        parsed["entities"] = unzoned

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
        if isinstance(raw, int) and 0 <= raw < len(WAYPOINT_MODES):
            return WAYPOINT_MODES[raw]
        if isinstance(raw, str) and raw.lower() in WAYPOINT_MODES:
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
        Convert raw LLM output into a format 2 scenario whose humans are
        consumable by ``ArenaHumanDynamicObstacle`` (yaw converted to radians,
        ``agent`` block and ``waypoint_mode`` set).
        """
        try:
            agent_type_paths: dict[str, str] = {}
            if generation_mode == GenerationMode.BEHAVIOR.value:
                generated_types = llm_output.get("agent_types") or {}
                if not generated_types:
                    self._logger.warn("behavior mode: LLM output contained no agent_types; agents fall back to built-in types.")
                agent_type_paths = self._materialize_agent_types(generated_types)

            humans: list[dict] = []
            config = {"format": SCENARIO_FORMAT, "humans": humans}

            obstacles = llm_output.get("obstacles") or {}
            for raw_agent in obstacles.get("dynamic") or []:
                agent = dict(raw_agent)
                human = {key: agent[key] for key in ("name", "model", "velocity") if key in agent}

                human["pose"] = self._pose_deg_to_rad(agent.get("pos", agent.get("pose")))
                human["waypoints"] = [self._pose_deg_to_rad(wp) for wp in agent.get("waypoints") or []]
                human["waypoint_mode"] = self._normalize_waypoint_mode(agent.get("waypoint_mode"))

                agent_type = str(agent.pop("agent_type", agent.pop("type", "adult")) or "adult")
                agent_type = agent_type_paths.get(agent_type, agent_type)
                if os.sep not in agent_type and agent_type not in _BUILTIN_AGENT_TYPES:
                    self._logger.warn(f"Unknown agent type {agent_type!r} for agent {agent.get('name')!r}, falling back to 'adult'.")
                    agent_type = "adult"

                agent_block: dict = {"agent_type": agent_type}
                desired_velocity = agent.pop("desired_velocity", None)
                if desired_velocity is not None:
                    agent_block["desired_velocity"] = float(desired_velocity)
                human["agent"] = agent_block

                humans.append(human)

        except Exception as e:
            self._logger.error(f"Failed to parse agents from LLM response: {e}")
            self._logger.error("Returning empty config!")
            config = {"format": SCENARIO_FORMAT}

        return config

    # LLM inference
    # -------------

    def _context_for_mode(self, generation_mode: str) -> str:
        if generation_mode == GenerationMode.BEHAVIOR.value:
            return BEHAVIOR_FORMAT
        return ARENA_FORMAT

    async def _prompt_to_config(self, prompt: str, top_p: float, generation_mode: str) -> dict:
        world = self._ctx.world_manager.world
        world_info = self.preprocess_world_description(self._ctx.world_manager.world_compacted(), world.kg if world.kg is not None else Scene())

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
                thinking_config=genai.types.ThinkingConfig(thinking_level=genai.types.ThinkingLevel.LOW),
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

    async def _parse_prompt(self, prompt: str, top_p: float, generation_mode: str) -> Scenario:
        """
        Parses the prompt to generate obstacles config.

        Args:
            prompt (str): The prompt for generating obstacles config.

        Returns:
            Scenario: Parsed scenario holding the generated humans.
        """
        assert GenerationMode.has_value(generation_mode)
        config = await self._prompt_to_config(prompt, top_p, generation_mode)

        return parse_scenario(config)

    async def reset(self, *, seed: int) -> tuple[list[Obstacle], list[DynamicObstacle]]:
        scenario = await self._parse_prompt(
            self._config.user_prompt.value,
            self._config.top_p.value,
            self._config.generation_mode.value,
        )

        return scenario.objects, scenario.humans

    def __init__(self, **kwargs: typing.Unpack[TaskModeKwargs]) -> None:
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
                    if cache.name is not None:
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
