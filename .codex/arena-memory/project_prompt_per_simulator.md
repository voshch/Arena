---
name: PROMPT registered per human simulator
description: TM_Obstacles.PROMPT is registered by each BaseHumanSimulator subclass via _register_task_modes, not centrally in tasks/registry.py
type: project
originSessionId: 41f13884-4923-44de-86d4-081896415ef3
---
`Constants.TaskMode.TM_Obstacles.PROMPT` is registered **per concrete human simulator**, not in `tasks/registry.py`'s `declare_obstacles()`. Each `BaseHumanSimulator` subclass overrides `_register_task_modes()` (called from `BaseHumanSimulator.__init__`) and registers its own `TM_Prompt` subclass:
- `HunavHumanSimulator` → `tasks/obstacles/prompt/hunav.py:TM_Prompt`
- `ArenaHumanSimulator` → `tasks/obstacles/prompt/arena.py:TM_Prompt`

`tasks/obstacles/prompt/__init__.py` only re-exports `TM_Prompt_Hunav` and `TM_Prompt_Arena` — there is no generic `TM_Prompt`.

**Why:** Different human simulators need different prompt task implementations, and only one human simulator is instantiated per run. Per-simulator registration delays the import until the relevant simulator is actually selected, avoiding hard dependencies on either backend.

**How to apply:** Don't add a `TM_Obstacles.PROMPT` entry to `tasks/registry.py:declare_obstacles()` — it will (a) collide with the per-simulator registration via the assertion in `register_obstacles`, and (b) have no valid loader since no generic `TM_Prompt` is exported. If a third human simulator is added, give it its own `_register_task_modes` override.
