---
name: gz-rgbd-turn-tearing
description: "gz oakd_rgbd map-frame tearing during sharp yaw measured 2026-07-29, data/stamps largely exonerated, points to rviz-side or GUI-load"
metadata: 
  node_type: memory
  type: project
  originSessionId: c4774e32-1971-447e-aac9-2a457703fef1
  modified: 2026-07-29T11:34:17.565Z
---

User sees rgbd cloud tearing/wall-doubling in rviz (map frame) only during sharp yaw, forward/backward perfect (oakd_rgbd on turtlebot, gz). Measured 2026-07-29 with a wall-angle-vs-TF probe (yaw sweep 0/0.6/1.2/1.8 rad/s, lidar as control, headless):

- TF/stamp chain honest: lidar registration error under 1 deg at real ~1.1 rad/s spin.
- rgbd held full 30 Hz during max yaw, no render-thread frame skipping.
- rgbd registration error bounded to single-digit degrees at ~0.9 rad/s (mean -2.8 deg, ~50-60 ms equivalent upper bound, noise-limited). Far too small for the visible tearing.

Verdict: published cloud+stamps broadly sound, tearing likely rviz-side or specific to user's GUI-loaded setup. UNTESTED: probe ran headless, user's repro has gz GUI + rviz on the same GPU, render lag under contention is the remaining suspect.

Probe gotchas for reruns: blocking TF lookup in a single-threaded-executor callback starves the listener (defer lookups), null TM robot drives via plain Twist on <ns>/cmd_vel, velocity smoother caps achieved yaw (~1.1 of commanded 1.8), tm_obstacles environment spawns office furniture in map_empty that wedges the spinning robot, clutter+peds wreck wall-angle estimators (RANSAC worse than segment-average there). Related: [[urbanverse-feedback-triage]].
