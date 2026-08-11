---
name: project-planner-peds-namespace
description: "arena_peds lives at the env/simulation namespace, not the robot one; DRL edge nests an extra task_generator_node level"
metadata: 
  node_type: memory
  type: project
  originSessionId: c83ced68-0743-4e9b-a405-9a300932ea71
---

Pedestrians for DRL planners are published on `arena_peds` at the **simulation/env namespace** `/arena/env_N/arena_peds` (27 Hz, `arena_people_msgs/Pedestrians`), by the human sim via its own `_namespace`.

Gotcha: the DRL edge node's namespace is `/arena/env_N/task_generator_node/<robot>` — it carries an **extra `task_generator_node` segment** vs the robot-manager namespace (`/arena/env_N/<robot>`) that `collision_tracker` uses. So `Namespace.simulation_ns` (a single `dirname`) is **off by one** from the edge node and lands at `/arena/env_N/task_generator_node/arena_peds` (no publisher). The correct env namespace `/arena/env_N` equals **`robot.node.get_namespace()`** (task_generator_node's own namespace).

Fix in place: collectors needing an env-level topic set `simulation_scoped = True` on the collector class (e.g. ArenaPedestrianCollector); the drl adapter passes `simulation_namespace=robot.node.get_namespace()` → edge_node → `Pipeline.from_config(simulation_ns=...)`, and the Pipeline resolves sim-scoped topics under that instead of `self._ns.simulation_ns`. Manifest topic must be `arena_peds` (was wrongly `pedestrians`).

Separately: CrowdNav-family planners read peds as `features.get("pedestrians") or []` which **crashes** (`ValueError: truth value of an array is ambiguous`) the moment a non-empty `(N,5)` ndarray arrives; masked while npeds was always 0. Use a `None` check, never `or []`, on any ndarray feature.
