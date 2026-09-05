from arena_rclpy_mixins.declarations import declare_double, declare_enum, declare_string
from arena_rclpy_mixins.ROSParamServer import ROSParamServer
from arena_rclpy_mixins.shared import Namespace

from task_generator.tasks.registry import _REGISTRY_NAMESPACE

NS = _REGISTRY_NAMESPACE("prompt")


def declare_schema(node: ROSParamServer, ns: Namespace) -> None:
    declare_string(node, ns("user_prompt"), "An empty space with no pedestrian.", label="Prompt", description="Natural-language prompt describing the desired crowd.")
    declare_double(node, ns("top_p"), 0.3, label="Top-p", description="Nucleus sampling top-p for LLM generation.")
    declare_enum(
        node,
        ns("generation_mode"),
        "arena",
        choices=["arena", "behavior", "behavior_tree", "crowded_behavior_tree"],
        label="Generation mode",
        description="Generation mode. Under human:=arena: 'arena' generates waypoint-following agents, 'behavior' additionally generates arena_humansim agent-type behavior definitions. Under human:=hunav: 'behavior_tree' and 'crowded_behavior_tree' generate HuNav behaviour trees.",
    )
