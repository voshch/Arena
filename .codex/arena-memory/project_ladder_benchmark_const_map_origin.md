---
name: project_ladder_benchmark_const_map_origin
description: "ladder benchmark worlds time out under DWB+jackal; constant (4.75,4.75) map origin is by-design env packing, NOT a bug"
metadata: 
  node_type: memory
  type: project
  originSessionId: 78d1d37e-bb50-4600-8935-4eeb6e8cc0c3
---

Investigated 2026-06-23 why ladder benchmark worlds (gazebo, jackal, dwb) failed. Initially
suspected a map-origin frame bug (all worlds load the nav map at the same global origin
(4.75,4.75) regardless of footprint). DISPROVEN by live test:

- Direct render check: render_map_files computes the CORRECT footprint-based origin per world
  (holodeck -0.25,-0.25 / arena -4.0,-2.75 / rag_floor 1.75,1.75); scenario start/goal inside.
- Live launch of rag_floor in isolation (1-world inline benchmark, --noexit): map_server map
  and global_costmap both frame=map, origin (4.75,4.75); the realizer shifts map, walls, robot
  teleport, and goal all by the same env reference, so they ALIGN. The (4.75,4.75) is the env
  packing anchor (confirm_world allocates a per-env reference so the footprint lands on a fixed
  global tile), NOT a bug.
- rag_floor produced a valid 59-pose /jackal/plan whose last pose (11.625,5.575) == realized
  goal (raw 8.625,2.575 + env-ref 3.0,3.0). It just TIMED OUT 60s ~1.4m short of goal.

Conclusion: these worlds are simply hard for DWB+jackal within the 60s stage timeout (tight
rooms, ~0.9m doors). Robot failing the task is valid benchmark data. holodeck succeeds,
arena/rag_floor time out. No frame/world/render bug. To raise success: longer timeout or
better planner tuning, not a code fix. The benchmark only looked catastrophic because of the
runner cascade ([[project_benchmark_runner_env_wedge_cascade]]), which is fixed.
