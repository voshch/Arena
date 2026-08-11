---
name: project_mechanism_disc_gate
description: "Door/elevator hardening on feature/semantic: disc gate (slab never moves through an agent), activation default 3.0, cabin floor slab - COMMITTED in 76af189, live-validated"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3553827a-da09-4bfe-b7d6-44750c656f54
  modified: 2026-08-10T02:15:07.683Z
---

2026-08-10, metarepo feature/semantic, follow-up to [[semantic-layer-v1]] shape C. COMMITTED: amended into `76af189` "restruct semantic and prep for release" (together with shape C + the discs-API hoist + SIM_NAME folds, 14 files +358/-205), followed by `6f0eedc` "simplify sim interface" (reset hooks concrete default no-op in BaseSim, trivial overrides deleted, 5 files +7/-32). LIVE-VALIDATED 2026-08-10: user's manual gazebo elevator-boarding repro passes (door holds, cabin floor present) and isaac works fine. Compliance benchmark re-acceptance waived by user. Final zero-slop review pass 2026-08-10 amended 6 cosmetic fixes in (reset-dup fold, _iter_zones fold in node.py, indent artifact, comment trims, ASCII separators in eval tests, eval pin 4fa53a6); backup/pre-slop-amend holds the pre-rewrite tip. User-reported "door kicks robot at threshold" decomposed by real-_tick trace harness into THREE verified defects, all fixed:

1. CLOSE-SIDE KILL BAND: center-point clearance + INSIDE_DOOR_BLOCKER_RADIUS=0.05 let the elevator door close through any chassis whose center stopped 5-40cm past the plane. FIX: agents are discs (name, xy, radius); `_swept_slab_blocked` + gate in `_advance_state` - blocked close re-triggers (reverses, tested against FULL remaining path to closed), blocked opening step holds, teleport snap gated, vertical kinds never open-blocked. INSIDE_DOOR_BLOCKER_RADIUS / near_door / closing_abort DELETED, `_step_elevator` is pure intent (departing stays armed, commits at genuinely-safe CLOSED).
2. OPENING RACE: activation 1.5m < v_max*transition(1s)+latency, so >=1.5-2 m/s entries rammed the still-retracting slab (fire-and-forget move_box lags physics ~200ms). FIX: Elevator.activation_distance default 1.5->3.0 in shared/world.py (Door already 3.0), invariant comment.
3. NO CABIN FLOOR: `_elevator_wall_geometries` spawned back+2 sides only, cabins overhang authored floor polygons (three_storied 1_elevator spans y[-1,1], zones start y=0) -> fall to ground plane. FIX: 4th 'floor' slab entry, full footprint, top at pos.z + _BOX_FLOOR_CLEARANCE (1cm lip beats z-fight).

Plumbing (post-hoist): `MechanismITF` base owns `_agent_robots: dict[str, tuple[frame, radius]]`, lazy TF buffer/listener, concrete `robot_discs()`/`robot_pose()`; sims only call `_register_agent_robot(robot, model_params)` / `_forget_agent_robot` at spawn/remove (isaac keeps a separate `_robot_prims` for prim paths). `SIM_NAME: ClassVar[str]` per sim, consumed by base `__init__` for semantics `set_sim`. Radius via `robot_footprint_radius(model_params)` in _interface.py (mobile cap radius, DEFAULT_AGENT_RADIUS=0.3 fallback); PED_RADIUS=0.3 in task_generator human adapter (Pedestrian.msg carries no radius, constant is the honest floor). `_semantics._agents_xy` strips radii (M2 kinds unchanged).

Verified: real-_tick scenario harness (scratchpad fsm_verify.py, tmp) - stop-at-threshold and creep now WAIT, 2.0/2.5 m/s + 200ms lag clear; suites 160 runtime + 482 ass + 335 tg green in-container; ruff clean. Known conservatism: closing gate includes the slab's current position in the tested band, so an agent touching the parked-open slab also holds the door.

Trace-harness lesson: `_advance_state`/`_step_elevator`/`_tick` are drivable tick-by-tick with the test suite's stub-mech pattern - use that for any future mechanism timing claim instead of arguing from code.
