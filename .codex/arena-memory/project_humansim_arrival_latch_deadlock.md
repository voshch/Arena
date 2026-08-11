---
name: project-humansim-arrival-latch-deadlock
description: "humansim peds froze from a latch/waypoint-advance dead band; fixed 2026-08-01, uncommitted in arena_humansim"
metadata: 
  node_type: memory
  type: project
  originSessionId: cac64185-d706-4fea-b3e9-9f6b5259c733
  modified: 2026-08-01T13:35:10.525Z
---

`pool.latched` is a BRAKE (Schmitt trigger r_enter 0.15 / r_exit 0.30 + `tau_brake` damping, added 8cda0dc to kill SFM jitter), not an arrival test. It got reused as the arrival signal by `GoToNode` (navigation.py:122, resets on entry) and by `_robots_done` (sim shutdown). `_advance_waypoints` predates the latch and kept its own 0.10 m test, so `WaypointMovement` agents got the brake with no consumer: latch at 0.15 m zeroes velocity and clears `has_goal`, advance needs 0.10 m, release needs 0.30 m -> permanent freeze. 5910da2 re-pointed the latch from `goal_pos` to `snap_terminal(waypoint)`, giving brake and sequencer different targets.

Fix (agent_manager.py:1454, uncommitted in humansim repo): `if dist_sq > r * r and not bool(pool.latched[i]): continue`. Verified live in gazebo 2026-08-01: peds traverse and reverse at waypoints. Suite 664 pass / 4 pre-existing cadrl fails (no tensorflow).

OPEN: `waypoint_threshold >= arrival_r_enter` assert deliberately NOT added (would raise on current defaults 0.10/0.15, node refuses to start). Waypoints spaced closer than r_exit 0.30 advance one per tick without releasing.

Presents as "humansim gets laggy and freezes over time" because agents drop out one at a time. See [[project_human_manual_steering_plan]], [[feedback_no_arena_humansim_modifications]].
