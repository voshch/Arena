---
name: project_vla_taskkind_decision
description: why VLA reuses TaskKind.GOTO_POSE instead of a dedicated TaskKind.VLA / served action
metadata: 
  node_type: memory
  type: project
  originSessionId: b8f4ef6b-1b87-4513-8442-a001a25d0987
---

VLA navigation is a task_generator-side adapter behavior, not a served action. Decided 2026-06-14 on branch merge/vla.

`VLAPhase` has `kind = TaskKind.GOTO_POSE` (not its own kind). The `vla` mobile adapter (subclasses Nav2Adapter) is the central bridge: owns vla_server lifecycle, the inference loop, per-action-type handlers, intent viz. Task mode `TM_VLA` is thin (feeds the `instruction` ROS param as a VLAPhase). Adapter autoselected at launch via `_TM_REQUIRED_ADAPTER = {"vla": "vla"}` in task_generator.launch.py when `mobile:=` is empty; explicit `TM_VLA.reset` RuntimeError backstops the mismatch case (uses `RobotManager.mobile_adapter_kind`).

**Why no dedicated TaskKind.VLA / VLA.action:** a `.action` implies a *served* action (task_server in arena_robots runs the server, adapter is a thin client). The VLA loop is irreducibly task_generator-side: `_is_valid_pose` reads `world_manager.map.occupancy`, `_handle_waypoints` uses `environment_manager.ezilear/realize`. Dependency runs task_generator -> arena_robots; reverse is forbidden, so a task_server VLA handler cannot reach occupancy/env-transforms. goto_pose CAN be served (nav2 self-contained); VLA cannot. So the adapter (task_generator side) is the only correct home, and an adapter actuates via an existing client (goto_pose), no action needed.

Cost of GOTO_POSE reuse, accepted: kind no longer implies GoToPhase (footgun; submit_task keys on isinstance so OK today); VLA invisible to `robot.accepts`; autoselect is a name-map not capability-derived; mismatch detection is hand-rolled (no free UNSUPPORTED_CAP). If VLA grows first-class, the upgrade is "IDL-light" TaskKind.VLA (mint kind + module-level UNSUPPORTED status int + reuse GotoPoseClient, NO .action, no task_server endpoint) which buys capability visibility + native backstop. A served VLA.action (state B) only if the loop is relocated and task_generator-side filtering is dropped or replaced.

Typed intent/meta feedback, if wanted, is a status topic (loop is adapter-side), not action feedback. meta is optional on the wire (omnivla-edge emits none). See [[project_planner_peds_namespace]] for the related env-namespace nesting gotcha.
