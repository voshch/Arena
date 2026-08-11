---
name: project_external_adapter_no_launch_file
description: "mobile:=external without a launch_file is a supported no-op since 2026-08-01; cmd_vel comes from the sim spawn, and a failed adapter launch hangs the reset forever"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7bc0dc5d-e178-4e03-bfd2-e5a22cf5f9e3
  modified: 2026-08-01T13:34:28.551Z
---

`mobile:=external` with no `launch_file` starts no navstack and logs one line instead of erroring (verified live on gazebo + jackal 2026-08-01: task_server and map_server up, hand-published Twist on `<ns>/<robot>/cmd_vel` moved the robot). `mobile.launch_file:=<path>` now actually reaches `ExternalBringup` as a kwarg. No robot in the tree ships an `external:` cap block.

Two facts that are easy to get backwards:

- The control stack (`ros2_control_node`, controller spawners, `twist_stamper`, `urdf_publisher`, odom relay) is launched by the **simulator adapter during `spawn_robot`**, not by the per-robot navstack launch. A navstack launch failure therefore leaves `cmd_vel` driving fully intact and only costs you `task_server` (so goal dispatch dies, manual driving does not).
- The passthrough `GotoPose` handler's immediate `succeed()` is a dispatch receipt, not a completion signal. Phase completion is three-tier and falls through to the geometric `is_satisfied` check, so episodes do not race through goals.

OPEN: if an adapter's launch dies (bad `launch_file` path, for example), `Adapter.wait_until_ready` waits unbounded on the missing action server, warning every 10s, and the reset never completes. Generic to all adapters, not specific to external. Left unfixed by user's call.

Related: [[project_integrate_planner_skill]], [[feedback_arena_cmd_vel_convention]]
