---
name: project-ros2-control-nan-commands
description: ros2_control topic bridge publishes NaN joint commands until a controller writes; sim backends must filter non-finite
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d265698-09bf-4737-84e8-7e1e932e5e46
---

ros2_control initializes command interfaces to NaN until the first controller write, and the vendored topic_based_hw fork publishes them verbatim on the joint-command topics. Any sim backend consuming `*/joint_commands_*` must treat non-finite values as "no command yet" and skip them, otherwise the physics solver explodes (Genesis: "Invalid constraint forces causing 'nan'" within ms of controller activation). Fixed in arena_genesis control.py stage_velocity/stage_position; check the same hazard when wiring new ros2_control-based sims ([[project-topic-based-hw-fork]]).
