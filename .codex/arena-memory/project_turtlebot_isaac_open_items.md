---
name: turtlebot-isaac-open-items
description: "end-of-cycle 2026-07-17 state — verified fixes, open debug items (lidar/points writer, go1 newton check), watch items (tilt ghosts, collision monitor)"
metadata: 
  node_type: memory
  type: project
  originSessionId: d1058176-4dfa-4e43-b9e3-c10599875612
---

Cycle 2026-07-17 (turtlebot/jackal isaac bringup) closed with turtlebot user-verified "pretty good in both sims". See [[urdf-loader-hardening]] for the loader mechanics.

Verified: turtlebot drives + turns in gz and isaac (wheel_drop lock, wheel geometry 0.036/0.233, caster mu 0.1 via fixed friction binding, camera optical frames, assembly wrapper parity).

OPEN:
- isaac planar lidar `lidar/points` writer attached but publishes nothing, zero warnings (scan works). Needs an instrumented live session. Blocks the "3D cloud marks, 2D scan clears" costmap hygiene option.
- go1 NEWTON walking check after foot friction binding (mu 0.8, no fdir1, now binds): the actual regression test, never run. go1 on PHYSX falls over backwards, likely the untuned path (generic 4e5/4e4 position gains, newton tuning is mjwarp-only); ruled out the ros2_control strip (canonical block with 12 joints intact in the rendered URDF). Probe recipe: compare isaac/joint_commands_position vs isaac/joint_states positions per joint, good tracking + bad posture = gait/friction, no commands = champ sidecar startup ordering.
- jackal skid-steer on physx is at the MODEL ceiling: isotropic rigid friction cone cannot express skid-steer (in-place saturates ~43% geometry-determined and mu-invariant, moving arcs fully pinned by lateral static friction). gz escapes via fdir1 friction pyramid, newton via pyramidal+impratio. Decision: skid-steer and legged run on newton; do not tune physx for them.
- jackal+nav2 ros2-bridge segfault (pre-existing) resurfaced, isaac container restarted mid-session.
- bumper contact sensor bump-test never reported (fix applied, init error should be gone).

WATCH:
- Tilt ghosts: pitching 2D lidar projects ground strikes into the scan plane, costmap marks them; local layer self-heals, GLOBAL costmap ghosts linger off-path and warp plans (benchmark path-inflation noise). Mitigations if benchmarks show it: marking via 3D cloud sources + scan clearing-only, IMU tilt gating, or scan obstacle_max_range cap ~3m.
- Robots with cameras but no optical frame links (a1, boxer, go1, rbkairos_plus, rbvogui_plus) hit the carb-warn fallback under isaac: tilted clouds until their chains grow optical frames or `<optical_frame_id>`.
- ridgeback front rocker (revolute effort=0) now auto-locked by the loader, untested.
- collision monitor pointcloud sources are live fleet-wide: false-stop risk on robots whose cloud sees their own body (anymal precedent).
- isaac depth images are all-finite (no REP-118 NaN/inf for no-return, unlike gz), rviz renders them white under the default 0-1 clamp (enable Normalize Range). Harmless for costmaps today (range caps + rolling window), but REP-118-assuming consumers (depthimage_to_laserscan etc.) would see phantom returns; parity fix = far-clip to inf conversion in the isaac depth writer graph.
