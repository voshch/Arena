"""Behaviour-tree prompt context, used only by the HuNav prompt backend.

Kept out of `context.py` because that module was rewritten for the
arena_humansim `BEHAVIOR_FORMAT` schema; the two backends ask the LLM for
different output shapes and share only ROLE / REASONING_GUIDE /
WORLD_DESCRIPTION, which are imported from there.
"""

BEHAVIOR_TREE_OUTPUT_FORMAT = """
    Do NOT explain anything. Output JSON only. Use realistic (x, y, 0) coordinates. Output must strictly follow this structure:
    ```json
{
  "hunav_agents": [
    {
      "name": <agent name>,
      "pos": [
        <x>,
        <y>,
        <yaw>
      ],
      "type": <agent type>,
      "model": <agent model>
    },
  ... ,
  ],

  "single_agent_nodes": [
    {
      "name": <node name>,
      "attributes": {
        <node attribute>: <attribute value>,
      },
      "order": <node order>
    },
    ...
  ],

  "multi_agent_nodes": [
    {
      "name": <node name>,
      "attributes": {
        <node attribute>: <attribute value>,
      },
      "orders": {
        <agent 1 name>: <node order in agent 1>,
        ... ,
        <agent n name>: <node order in agent n>,
      }
    },
    ,
    ...
  ]
}
    ```
"""

BEHAVIOR_TREE_FIELD_DESCRIPTION = """
    Top-level structure
    - "hunav_agents" contains a list of hunav agents, each with:
      - `name`: the agent's unique identifier (e.g., "hunav_1").
      - `pos`: a list [x, y, yaw] representing the object's position and rotation. You should pay attention to where the agent should be spawned and faced, place the agent within the correct zone and adjust the yaw reasonably.
      - `type`: the type of dynamic obstacle (e.g., `adult`, `child`, etc.).
      - `model`: the type of model used for the dynamic obstacle. the type of model can be one of the following only:
        - "female_adult_business_02"
        - "female_adult_medical_01"
        - "female_adult_police_01"
        - "female_adult_police_02"
        - "female_adult_police_03"
        - "male_adult_construction_01"
        - "male_adult_construction_02"
        - "male_adult_construction_03"
        - "male_adult_construction_05"
        - "male_adult_medical_01"
        - "male_adult_police_04"

    - "single_agent_nodes": Contains a list of behavior tree nodes that one and only one agent involved in, each has:
      - `name`: name of the node, only use provided node name, do not modify!
      - `attributes`: a dictionary of key-value pairs (parameters passed to the node)
      - `order`: an integer represent the order of execution of this node in the agent behavior tree. The agent will handle nodes in a ascending order determined by this field. For each agent, every nodes must be unique no matter the type (single-agent or multi-agent nodes) is.

    - "multi_agent_nodes": Contains a list of behavior tree nodes that more than one agent involved in, each has:
      - `name`: name of the node
      - `attributes`: a dictionary of key-value pairs (parameters passed to the node)
      - `order`: a dictionary of key-value pairs (<agent name>-<order value>) represent the order of execution of this node in each agent behavior tree. The agents will handle nodes in a ascending order determined by this field. For each agent, every nodes must be unique no matter the type (single-agent or multi-agent nodes) is.
"""

BEHAVIOR_TREE_FORMAT = f"""
{BEHAVIOR_TREE_OUTPUT_FORMAT}
{BEHAVIOR_TREE_FIELD_DESCRIPTION}
"""

SPLIT_PROMPT_INSTRUCTION = """
Your task is split user prompt into 2 prompts for 2 human simulator pipeline.
One is Ambient Agents pipeline, which will control how group of pedestrians navigate, your prompt will affect the navigation direction of the pedestrian groups, where the pedestrians groups is spawned, and where the groups will start and finish. Your prompt should describes the sequence of places the pedestrian groups should follow, but don't make up places, only use places the user refers if mentioned.
The other pipeline is Spotlight Agent pipeline, which should be more detailed as this pipeline can control complex behavior of agents.

Do NOT explain anything. Output JSON only, and must strictly follow this structure:
```json
{
    "ambient_agents_prompt": <ambient_agents_prompt>,
    "spotlight_agents_prompt": <spotlight_agents_prompt>
}
```

Example:
Input: Depict an emergency evacuation where at first, there're 5 people waiting in line by the pharmacy room door, gradually advance to move forward, then a fire occurs and everyone in every rooms run out of their room, to the hallways, then toward the exit in the main hallways.
Output:
```json
{
    "ambient_agents_prompt": "People run out of their room, to the hallways, and through the main hallway entrance.",
    "spotlight_agents_prompt": "A group of 5 peopel stand into a queue in the main central hallway, by the pharmacy room door, every one should stand 1 meter away from the wall, and the first of the line should stand one meter away from the edge of the door. As soon as a spot opens at the front, each person immediately steps forward, advancing in sequence toward the waiting area door. Every person, when they reach the front of the line, must wait 20 seconds and then enter the pharmacy room, then he enters the pharmacy room."
}
```
"""
