---
name: arena-peds-name-is-sim-path
description: "arena_peds published Pedestrian.name is the env-prefixed sim_path, not the bare obstacle name; consumers must key on sim_path"
metadata: 
  node_type: memory
  type: project
  originSessionId: e37764d3-82f6-400b-8da5-731cfa627aa7
---

Both human backends publish `sim_path` (e.g. `env_0/agent_1`) as `Pedestrian.name` on `arena_peds`: arena_humansim via `_agent_names[id] = obstacle.sim_path` (arena_humansim.py:776), hunav via `HunavDynamicObstacle.from_dynamic_obstacle` `name=obj.sim_path`. Gazebo's PedSkeletonPlugin keys actors on that published name, so any other consumer (Isaac adapter `pedestrian_update`) must use the same key. Fixed 2026-06-11: all four `IsaacSimulator.pedestrian_*` methods now key on `sim_path` (prim `/World/env_N/<name>`), `_NS_PEDESTRIAN` removed, and `pedestrian_update` logs a throttled warning on NOT_FOUND results (they were silently discarded, which hid the frozen-peds bug).

Also: with `sim:=isaac` the default human backend is `arena` (arena_humansim), derived in task_generator.launch.py (~line 197); the launch/human README claiming `isaac → hunav` is stale.
