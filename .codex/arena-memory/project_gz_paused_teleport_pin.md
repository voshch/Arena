---
name: gz-paused-teleport-pin
description: SetEntityPose on a paused gz world leaves the model unactuated after resume; re-set on a running world fixes it
metadata: 
  node_type: memory
  type: project
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-08T17:56:35.603Z
---

The reset-pin: episode resets teleport robots via SetEntityPose while the sim is paused.
After unpause, TF/odom show the new pose and joint commands execute (wheels spin, 30Hz
joint_states, controllers active) but contact forces never move the base. The identical
set_pose re-issued on a RUNNING world restores physics instantly (measured: 0.000m/15s
pinned, then 6.42m/25s after nudge). Deterministic on jackal (6/6 resets), stochastic
transient on mpo700 (1/3, ~132s self-recovery) - a timing race on whether the set lands
in a paused interval. Manual GUI teleport un-sticks for the same reason.

**Why:** ECM pose component updates while paused, but physics (dartsim) only adopts pose
commands inside an active step - inferred, not verified in gz source.

**How to apply:** UNSOLVED - two theories falsified (2026-08-09). (1) The unpause window
was never missing: GazeboSimulator.robot_move has wrapped moves in node.unpause_window()
since commit 03ee6a8, pin happens anyway. (2) Dwell does NOT fix it: holding the window
open until /clock provably advanced 0.05s past the set_pose still pinned ep2 identically
(cadrl/jackal, 0.0m over 351s, cmds flowing). So a pose set on a RUNNING world still
pins when the reset re-pauses afterward - "paused set is ignored" is dead as stated.
Open discriminators (scripted in .claude/scratch/repos/pin_discriminate.sh, never ran to
completion): does a bare pause/unpause cycle un-pin? same-pose set? tiny offset? or only
a far nudge? Also unsettled whether wheels actually spin during the pin (earlier
joint_states evidence was a latch artifact) - frozen wheels would point at a sleeping /
detached physics island that any set_pose wakes, not a lost pose. ep1 (first reset)
always works - possibly because spawn pose == start pose makes it a no-op teleport.
ep0 spawn (create) while paused is always fine.
