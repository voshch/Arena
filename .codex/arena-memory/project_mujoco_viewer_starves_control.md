---
name: project_mujoco_viewer_starves_control
description: MuJoCo passive viewer on the physics tick starves joint_command subs (single-thread executor) so robot freezes; viewer must be on its own slower timer
metadata: 
  node_type: memory
  type: project
  originSessionId: 9a5a046c-21b4-4a27-b3be-1e4323777aca
---

In arena_mujoco run_mujoco.py the whole server runs on ONE rclpy timer under a single-threaded
executor: `_tick` = step physics -> `hooks.run_pumps()` (control pump applies staged ctrl) ->
viewer `sync()`. Commands are STAGED in a subscription callback (`stage_velocity` on
joint_commands_velocity) serviced by the same thread.

If `_update_viewer()`/`viewer.sync()` runs every physics tick, the timer callback is expensive
enough to be effectively always-ready, so the executor never services the joint_command
subscription -> `_staged` stays empty -> ctrl=0 -> wheels frozen (robot dead-still, cameras black).
Verified: viewer on = wheel vel ~0.0001, odom 0 m/7s; viewer off (headless) = 4.4 rad/s, 4.6 m.

Fix (2026-06-11): put `_update_viewer` on its OWN slower timer (_VIEWER_PERIOD_S=0.05, 20Hz),
NOT on `_tick`. Drops aggregate timer duty below saturation so commands flow. Robot then drives
with the viewer open.

Note: the `--headless` flag does not reliably suppress the viewer (it comes up anyway), so this
bit every run. Separate open issue: EGL offscreen camera renderer fails ("Failed to make the EGL
context current") while the GLFW viewer is up - camera renders fine headless (~24Hz) but is black
with the viewer. Needs the EGL render on its own thread/context. See [[project_mujoco_texture_facts]].
