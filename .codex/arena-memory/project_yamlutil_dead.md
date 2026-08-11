---
name: YAMLUtil is dead code
description: YAMLUtil class in task_generator/simulators/human/utils.py is dead code scheduled for deletion
type: project
originSessionId: 3cf40189-7b07-4ff9-8a87-96910ab58b31
---
`YAMLUtil` in [task_generator/task_generator/simulators/human/utils.py](task_generator/task_generator/simulators/human/utils.py) is dead code. Grep shows only self-references inside the file itself — no external importers.

**Why:** Confirmed by user on 2026-04-11 during the `_node` global removal refactor. User said: "yamlutil is honestly dead code. ignore it and schedule for deletion." It was the only blocker to a clean `rosparam_get` removal (it imported the deprecated helper), and rather than thread DI through it the decision was to delete it outright.

**How to apply:** When touching [simulators/human/utils.py](task_generator/task_generator/simulators/human/utils.py), do not invest effort in fixing `YAMLUtil` — delete it. If a future refactor needs to remove deprecated helpers that `YAMLUtil` imports, delete `YAMLUtil` in the same commit rather than threading compatibility shims through it.
