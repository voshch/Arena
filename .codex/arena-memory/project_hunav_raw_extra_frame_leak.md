---
name: hunav-raw-extra-frame-leak
description: "HunavDynamicObstacle prefers raw scenario extras (position, goals) over realized pose/waypoints, leaking unrealized coords when env ref or level origin is nonzero"
metadata: 
  node_type: memory
  type: project
  originSessionId: e37764d3-82f6-400b-8da5-731cfa627aa7
---

`HunavDynamicObstacle.from_dynamic_obstacle` (task_generator/simulators/human/hunav/__init__.py) builds `init_pose` from `extra['position']` and goals via `Goals.parse(extra)` when the scenario provides them, both RAW scenario yaml coordinates. The realizer only realizes `obj.pose` and `obj.waypoints`, extras pass through untouched. With a nonzero env reference (shelf packer) or level origin, hunav agents then live in the unrealized frame while the sim stage and spawned ped actors are realized: uniform position offset. The arena_humansim parser is clean (uses realized `obs.waypoints` in both branches). Found 2026-06-11 while chasing Isaac ped offsets, not yet fixed. Related: [[arena-peds-name-is-sim-path]].
