import json
import math
import os
import pickle
import tempfile
import time
import xml.etree.ElementTree as ET
from enum import Enum

import attrs
import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from arena_hunav_sim_bridge.agent.llm_parser import Parser
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from task_generator.utils.gpt import genai

# Avoids errors related to cv2 + pyglet + X11 with arena_text_crowd
os.environ["PYGLET_HEADLESS"] = "true"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from arena_rclpy_mixins.ROSParamServer import ROSParamT
from arena_simulation_setup.tree.World import LevelDescription, WorldIdentifier
from arena_simulation_setup.utils.cattrs import converter
from arena_text_crowd.converters.arena_world_to_text_crowd_scenario import (
    arena_world_to_text_crowd_scenario,
)
from arena_text_crowd.crowd_generation_pipeline.arena_text_crowd_generation_pipeline import (
    ArenaTextCrowdGenerationPipelineConfig as ATCPConfig,
)
from arena_text_crowd.crowd_generation_pipeline.arena_text_crowd_generation_pipeline import (
    CrowdGenerationPipeline,
)
from arena_text_crowd.crowd_generation_pipeline.velocity_field_generation.velocity_field_generation_pipeline import (
    VelocityFieldGenerationPipelineConfig as VFGPConfig,
)
from hunav_msgs.srv import SetArenaWorldBounds, SetVelocityField

from task_generator.simulators.human.hunav.hunav import HunavDynamicObstacle
from task_generator.tasks.obstacles import (
    DynamicObstacle,
    Obstacle,
    TM_Obstacles,
)
from task_generator.tasks.obstacles.prompt.velocity_field_marker import (
    VelocityFieldVisualizer,
)

from .prompt_utils import (
    ARENA_FORMAT,
    BEHAVIOR_TREE_FORMAT,
    BT_REF_DOC_PATH,
    CHROMA_DB_PATH,
    LOCAL_LM,
    REMOTE_LM,
    SPLIT_PROMPT_INSTRUCTION,
    SYSTEM_INSTRUCTION,
    create_chroma_db,
    get_chroma_collection,
    get_relevant_bt_nodes,
    process_json_doc,
)

DEBUG: bool = os.environ.get("ARENA_DEBUG", "0").lower() in ("1", "true")


@attrs.define()
class _ParsedConfig:
    static: list[Obstacle]
    dynamic: list[DynamicObstacle]


class GenerationMode(Enum):
    ARENA = "arena"
    BEHAVIOR_TREE = "behavior_tree"
    CROWDED_BT = "crowded_behavior_tree"

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
    Prompt task generator for obstacles.

    This class generates obstacles based on a prompt configuration.

    Attributes:
        _config (Config): Configuration object for obstacle generation.
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
                            math.degrees(entity.pose.orientation.to_yaw()),  # I use degree for yaw for now (look at `context.py``)
                        ],
                    }
                    for entity in zone.entities.static
                ],
            }
            parsed["zones"].append(parsed_zone)

        return json.dumps(parsed)

    def llm_bt_output_to_config(
        self,
        llm_output: dict,
        generation_mode: str,
        *,
        crowd_pedestrians: None | list[dict],
    ) -> dict:
        if generation_mode == GenerationMode.ARENA.value:
            return llm_output

        if generation_mode == GenerationMode.CROWDED_BT.value and crowd_pedestrians is not None:  # TODO: Use scheme/class instead of Dict
            for ped in crowd_pedestrians:
                llm_output["hunav_agents"].append(
                    {
                        "name": ped["name"],
                        "pos": ped["pos"],
                        "model": ped["model"],
                    }
                )
                llm_output["single_agent_nodes"].append(
                    {
                        "name": "FollowVelocityField",
                        "attributes": {
                            "agent_name": ped["name"],
                            "velocity_field_group_id": ped["group_id"],
                            "time_step": 0.1,
                            "tolerance": 0.2,
                        },
                        "order": 0,
                    }
                )
        try:
            config = {"obstacles": {"static": [], "dynamic": []}}

            parser = Parser(llm_output)
            parser.parse()
            for hunav_agent in parser.agents.values():
                hunav_config = {
                    "id": hunav_agent.id,
                    "name": hunav_agent.name,
                    "pos": hunav_agent.pos,
                    "model": hunav_agent.model,
                    "waypoints": hunav_agent.waypoints,
                }

                behavior_tree_xml = hunav_agent.to_xml()

                tmp_xml_file = tempfile.NamedTemporaryFile(mode="w+t", suffix=".xml", dir=self.tmp_dir.name, delete=False)

                tmp_xml_file.write(
                    ET.tostring(
                        behavior_tree_xml,
                        encoding="UTF-8",
                        method="xml",
                        xml_declaration=True,
                    ).decode("utf-8")
                )

                hunav_config.update({"behavior_tree": tmp_xml_file.name})

                config["obstacles"]["dynamic"].append(hunav_config)

        except Exception as e:
            self._logger.error(f"Failed to parse Behavior tree from LLM response: {e}")
            self._logger.error("Returning empty config!")
            config = {}

        return config

    def setup_chroma(self):
        if os.path.isdir(CHROMA_DB_PATH):
            self.chroma_collection = get_chroma_collection(CHROMA_DB_PATH, self.inference_client)
        else:
            processed_doc = process_json_doc(BT_REF_DOC_PATH)
            self._logger.info("Creating Chroma DB from Behavior Tree Nodes Reference...")
            self.chroma_collection = create_chroma_db(
                documents=processed_doc,
                db_path=CHROMA_DB_PATH,
                client=self.inference_client,
            )

    def send_velocity_msg(self, velocity_field: np.ndarray) -> "SetVelocityField.Response":
        n_groups, h, w, c = velocity_field.shape
        msg = Float32MultiArray()
        msg.data = velocity_field.astype(np.float32).flatten(order="C").tolist()
        msg.layout.dim = [
            MultiArrayDimension(label="G", size=n_groups, stride=n_groups * h * w * c),
            MultiArrayDimension(label="H", size=h, stride=h * w * c),
            MultiArrayDimension(label="W", size=w, stride=w * c),
            MultiArrayDimension(label="C", size=c, stride=c),
        ]

        req = SetVelocityField.Request()
        req.velocity_field = msg

        response: SetVelocityField.Response = self.velocity_field_client.call(req)

        return response

    def send_arena_world_bounds_msg(self) -> tuple["SetArenaWorldBounds.Response", float, float, float, float]:
        # TODO: Optimize
        # Get Arena World size
        x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

        world = self._ctx.world_manager.world_compacted()
        for zones in world.zones:
            x_min, y_min, x_max, y_max = (
                min(x_min, *(corner.x for corner in zones.corners)),
                min(y_min, *(corner.y for corner in zones.corners)),
                max(x_max, *(corner.x for corner in zones.corners)),
                max(y_max, *(corner.y for corner in zones.corners)),
            )
        arena_world_bounds = [x_min, y_min, x_max, y_max]

        msg = Float32MultiArray()
        msg.data = arena_world_bounds
        msg.layout.dim = [
            MultiArrayDimension(label="bounds", size=4, stride=4),
        ]

        req = SetArenaWorldBounds.Request()
        req.arena_world_bounds = msg

        response: SetArenaWorldBounds.Response = self.arena_world_bounds_client.call(req)

        return response, x_min, y_min, x_max, y_max

    async def _prompt_to_config(self, prompt: str, top_p: float, generation_mode: str, local: bool = False) -> dict:
        world_info = self.preprocess_world_description(self._ctx.world_manager.world_compacted())

        messages = []
        crowd_pedestrians = None
        pipeline_start = time.time()

        if generation_mode == GenerationMode.BEHAVIOR_TREE.value:
            if generation_mode not in self.cached_context_name.keys():
                cache = self.inference_client.caches.create(
                    model=REMOTE_LM,
                    config=genai.types.CreateCachedContentConfig(
                        display_name=generation_mode + "_context",
                        system_instruction=SYSTEM_INSTRUCTION,
                        contents=BEHAVIOR_TREE_FORMAT,
                    ),
                )
                if cache.name is not None:
                    self.cached_context_name.update({generation_mode: cache.name})

            bt_nodes = get_relevant_bt_nodes(
                query=f'What are the nodes should be used for creating the behavior tree as described below: "{prompt}". Use GoTo node to guide agents to isolated places if needed.',
                collection=self.chroma_collection,
            )

            self._logger.warn(f"Choosen bt_nodes: {bt_nodes}")

            messages.append(
                f"Generate hunav agents data for a simulation where {prompt}. Generate data base on this world data as below <WORLD_DESCRIPTION>: {world_info}. Use these behavior tree nodes only: {bt_nodes}. Only return valid JSON using the format declared in the system context, with no explanation, thoughts, or extra text."
            )

        elif generation_mode == GenerationMode.ARENA.value:
            if generation_mode not in self.cached_context_name.keys():
                cache = self.inference_client.caches.create(
                    model=REMOTE_LM,
                    config=genai.types.CreateCachedContentConfig(
                        display_name=generation_mode + "_context",
                        system_instruction=SYSTEM_INSTRUCTION,
                        contents=ARENA_FORMAT,
                    ),
                )
                if cache.name is not None:
                    self.cached_context_name.update({generation_mode: cache.name})

            messages.append(
                f"Generate dynamic obstacles data for a simulation where: {prompt}. Generate data base on this world data as below <WORLD_DESCRIPTION>: {world_info}. Only return valid JSON under the 'dynamic' field, using the format declared in the system context, with no explanation, thoughts, or extra text."
            )

        elif generation_mode == GenerationMode.CROWDED_BT.value:
            if generation_mode not in self.cached_context_name.keys():
                cache = self.inference_client.caches.create(
                    model=REMOTE_LM,
                    config=genai.types.CreateCachedContentConfig(
                        display_name=generation_mode + "_context",
                        system_instruction=SYSTEM_INSTRUCTION,
                        contents=BEHAVIOR_TREE_FORMAT,
                    ),
                )
                if cache.name is not None:
                    self.cached_context_name.update({generation_mode: cache.name})

            # Split prompts for Ambient Agents and Spotlight agent
            split_prompt_res = self.inference_client.models.generate_content(
                model=REMOTE_LM,
                contents=f"Split prompts given this user prompt: {prompt}",
                config=genai.types.GenerateContentConfig(
                    system_instruction=SPLIT_PROMPT_INSTRUCTION,
                    top_p=top_p,
                    temperature=0.2,
                    top_k=40,
                    thinking_config=genai.types.ThinkingConfig(include_thoughts=False, thinking_level="low"),
                ),
            )
            split_prompt_answer = split_prompt_res.text
            assert split_prompt_answer is not None
            split_prompt_answer = split_prompt_answer.strip()
            if split_prompt_answer.startswith("```"):
                split_prompt_answer = split_prompt_answer.split("\n", 1)[1] if "\n" in split_prompt_answer else split_prompt_answer[3:]
                if split_prompt_answer.endswith("```"):
                    split_prompt_answer = split_prompt_answer[:-3]
                split_prompt_answer = split_prompt_answer.strip()
            split_prompt_answer = json.loads(split_prompt_answer)

            prompt = split_prompt_answer["spotlight_agents_prompt"]
            ambient_agent_prompt = split_prompt_answer["ambient_agents_prompt"]

            self._logger.debug(f"Spotlight Agents prompts: {prompt}\nAmbient_agents_prompt:{ambient_agent_prompt}")

            bt_nodes = get_relevant_bt_nodes(
                query=f'What are the nodes should be used for creating the behavior tree as described below: "{prompt}". Use GoTo node to guide agents to isolated places if needed.',
                collection=self.chroma_collection,
            )

            self._logger.warn(f"Choosen bt_nodes: {bt_nodes}")

            messages.append(
                f"Generate hunav agents data for a simulation where {prompt}. Generate data base on this world data as below <WORLD_DESCRIPTION>: {world_info}. Use these behavior tree nodes only: {bt_nodes}. Only return valid JSON using the format declared in the system context, with no explanation, thoughts, or extra text."
            )

            cgp_config = ATCPConfig(
                visual=False,
                save_path=os.path.join(
                    get_package_share_directory("arena_text_crowd"),
                    "generated_velocity_field",
                ),
                model=REMOTE_LM,
                top_p=top_p,
            )
            # text_crowd_unet_dir = os.path.join(
            #     get_package_share_directory("arena_text_crowd"),
            #     "models",
            #     "velocity_field_generation",
            #     "sd_unet_2d_conditioned",
            # )
            text_crowd_unet_dir = "/home/linh/ductai_nguyen_ws/Text-Crowd/text_crowd/Language_Crowd_Animation/Models_Server_ForTest/Field-Full-V2/checkpoint-270000/unet"
            vfgp_config = VFGPConfig(unet_dir=text_crowd_unet_dir)
            cgp = CrowdGenerationPipeline(cgp_config, vfgp_config)

            arena_world_bounds_res, x_min, y_min, x_max, y_max = self.send_arena_world_bounds_msg()
            self._logger.debug(f"Set Arena World bounds response: {arena_world_bounds_res.success}, {arena_world_bounds_res.message}")
            self.velocity_field_visualizer.update_world_bounds(x_min, y_min, x_max, y_max)

            scenario, arena_entity_to_semantic_entity_map = arena_world_to_text_crowd_scenario(self._ctx.world_manager.world_compacted(), scenario_size=(1024, 1024))

            self._logger.info("Generating velocity field...")
            velocity_field, crowd_pedestrians, text_crowd_scenario = cgp.generate(
                prompt=ambient_agent_prompt,
                scenario=scenario,
                arena_world_description=self._ctx.world_manager.world_compacted(),
                arena_entity_to_semantic_entity_map=arena_entity_to_semantic_entity_map,
            )  # (n_groups, 64, 64, 2) (g, y, x, 2)

            vel_res = self.send_velocity_msg(velocity_field)
            self._logger.debug(f"Set velocity field response: {vel_res.success}, {vel_res.message}")

            self.velocity_field_visualizer.publish_markers(velocity_field)

        if local:  # Currently not supported
            return {}
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(LOCAL_LM, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(LOCAL_LM)
            # Format using Qwen chat template
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            self._logger.info("Start inference...")
            start = time.time()

            # Tokenize input
            inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

            # Generate output
            outputs = model.generate(
                **inputs,
                max_new_tokens=32768,
            )

            # Extract generated tokens (excluding prompt)
            generated_ids = outputs[0][len(inputs.input_ids[0]) :]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            end = time.time()
            self._logger.info(f"Inference done, took: {end - start:.1f}s")

        else:
            self._logger.info("Start inference...")
            start = time.time()
            response = await self.inference_client.aio.models.generate_content(
                model=REMOTE_LM,
                contents=messages,
                config=genai.types.GenerateContentConfig(
                    cached_content=self.cached_context_name[generation_mode],
                    top_p=top_p,
                    temperature=0.2,  # Most of the example prompts use this set of parameters (except for top_p), see https://docs.cloud.google.com/vertex-ai/generative-ai/docs/prompt-gallery/samples/extract_tech_specs
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

        # Parse it into a Python dict
        config = self.llm_bt_output_to_config(
            json.loads(answer),
            generation_mode,
            crowd_pedestrians=crowd_pedestrians,
        )

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

            if generation_mode == GenerationMode.CROWDED_BT.value:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix="velocity_field_",
                    suffix=".npy",
                    dir=self.tmp_dir.name,
                    mode="wb",
                ) as file:
                    np.save(file, velocity_field)
                    self._logger.info(f"Saved velocity field to {file.name}")
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    prefix="text_crowd_scenario_",
                    suffix=".pkl",
                    dir=self.tmp_dir.name,
                    mode="wb",
                ) as file:
                    pickle.dump(text_crowd_scenario, file)

        return config

    async def _parse_prompt(self, prompt: str, top_p: float, generation_mode: str) -> _ParsedConfig:
        """
        Parses the prompt to generate obstacles config.

        Returns:
            _ParsedConfig: Parsed configuration containing static and dynamic obstacles.
        """
        assert GenerationMode.has_value(generation_mode)
        config = await self._prompt_to_config(prompt, top_p, generation_mode)

        static_obstacles: list[Obstacle]
        dynamic_obstacles: list[DynamicObstacle]

        static_obstacles = [
            # Obstacle.parse(obs)
            # for obs
            # in itertools.chain(
            #     config.get("obstacles", {}).get("static", []),
            #     config.get("obstacles", {}).get("interactive", []),
            # )
            # This causes bug so temporarily disabled
        ]

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
        # self.inference_client = InferenceClient(
        #     provider="together",
        #     api_key=os.environ["HF_TOKEN"],
        # )

        def _load_config(filename: str = "default.yaml") -> "HunavDynamicObstacle":
            """Load config from YAML file in arena_bringup configs."""

            # second priority: Install space
            config_path = os.path.join(
                get_package_share_directory("arena_bringup"),
                "configs",
                "hunav_agents",
                filename,
            )

            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                assert isinstance(config, dict), "Config file is not properly formatted."
                agent_config = config["hunav_loader"]["ros__parameters"]["agent1"]
                return agent_config

            except Exception as e:
                raise RuntimeError(f"Error loading config from {config_path}") from e

        # default_hunav_config = _load_config() # Is not used yet

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

        self.setup_chroma()

        self.tmp_dir = tempfile.TemporaryDirectory(
            dir=os.path.join(
                WorldIdentifier(self._ctx.world_manager.loaded_world).resolve_sync().path,
                "scenarios",
            )
        )  # Temporary directory to store behavior tree XML files

        node_fqn = self.node.get_fully_qualified_name()
        self.velocity_field_client = self.node.create_client(SetVelocityField, f"{node_fqn}/set_velocity_field")
        while not self.velocity_field_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f"Waiting for service {node_fqn}/set_velocity_field")
        self.velocity_field_visualizer = VelocityFieldVisualizer(
            self.node,
            topic_name=f"{node_fqn}/velocity_field_marker",
        )
        self.arena_world_bounds_client = self.node.create_client(SetArenaWorldBounds, f"{node_fqn}/set_arena_world_bounds")
        while not self.arena_world_bounds_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(f"Waiting for service {node_fqn}/set_arena_world_bounds")

    def __del__(self):
        try:
            # Delete caches
            for cache_name in self.cached_context_name.values():
                self.inference_client.caches.delete(name=cache_name)
            self.cached_context_name: dict[str, str] = {}
        except Exception as e:
            self._logger.error(e)
            self._logger.error("Can not delete cache! Maybe it was deleted earlier.")
