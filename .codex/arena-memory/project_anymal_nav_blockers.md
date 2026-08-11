---
name: project_anymal_nav_blockers
description: "anymal_c stands still under nav because the collision_monitor self-detects its body and zeroes cmd_vel, not locomotion and not the controller"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e3405d0-3812-4b3e-94dc-10216c6fa11c
---

anymal_c (legged, CHAMP) does NOT navigate even after the a1/go1 lidar recipe. Locomotion is fine (verified: direct cmd_vel injection walks AND turns, legs move, body translates smoothly). The "stands still" is one nav-pipeline root cause:

**collision_monitor self-detection (the blocker).** The collision_monitor reads anymal's 3D `lidar_points` through a footprint-sized StopPolygon and has NO footprint-clearing (unlike the costmap, which clears the robot's own footprint). anymal's high-mounted multi-beam lidar's downward beams hit the robot's OWN body inside the StopPolygon -> StopPolygon triggers -> zeroes the final `cmd_vel`. Confirmed: `cmd_vel_nav`=0.25 but final `cmd_vel`=silent, with live `polygon_stop`/`collision_points_marker`. Because the final cmd_vel is held at zero, NOTHING reaches CHAMP, neither linear nor angular. a1/go1 (small footprint 0.19x0.09) dodge this; anymal (0.45x0.30, radius 0.50) does not.

**Do NOT blame RPP / rotate_to_heading.** Early on I wrongly concluded RPP rotate_to_heading was oscillating and set `use_rotate_to_heading: false`. WRONG: RPP rotates in place fine via `use_rotate_to_heading: true`, which EVERY arena robot uses (the user corrected me). The angular "oscillation" I saw in `cmd_vel_nav` was RPP commanding open-loop into a pipeline the collision_monitor was holding at zero (robot never actually rotates, so the controller flails). Setting use_rotate_to_heading=false treated a symptom AND broke in-place turning. That edit was reverted.

**Fix direction (not yet done):** make the collision_monitor stop self-detecting. Cleanest is a scan-only collision source (the horizontal `lidar` LaserScan at the high mount clears the body, unlike the 3D cloud), which needs a per-robot collision-source override (mirror the global_observation substitution pattern in nav2.py) OR a lidar mount/FOV change. Runtime `lidar_points.enabled false` did NOT reconfigure it.

**Applied + kept (uncommitted, anymal_c is untracked):** lidar migrated to shared `gz_lidar` macro; `global_observation` height-gate (include lidar_points, min_obstacle_height 0.2); `footprint_padding: 0.20`. Correct hardening but NOT sufficient alone. Related turn-dynamics note: [[project_height_planner_oscillates]].
