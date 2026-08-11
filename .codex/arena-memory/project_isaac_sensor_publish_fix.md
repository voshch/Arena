---
name: project_isaac_sensor_publish_fix
description: "Isaac 6.0 lidar+camera publishing fix — multitick on / perSensorTickTlas off, RTX lidar needs config=, contact sensor isolation"
metadata: 
  node_type: memory
  type: project
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

Fixed (2026-06-22, uncommitted on `jazzy`) why `arena launch sim:=isaac robot:=turtlebot` published no lidar/camera. Four causes, all in `arena_isaac`:

1. **`parse_gazebo` had no per-sensor isolation.** Turtlebot's bumper `contact` sensor is the FIRST `<sensor>` in the URDF and throws on create ("needs a prim with collision API"), aborting the whole loop so lidar/camera (later in the URDF) never spawned. Jackal worked only because it has no contact sensor. Fix: `sensors.py` now wraps each sensor in try/except (`_spawn_sensor`), logs `carb.log_error`, continues.

2. **RTX sensors need multi-tick rendering, which the prior odom fix had disabled.** `/rtx/hydra/supportMultiTickRate` must be **True** or RTX lidar + camera render products never tick (writers attach, publisher count stays 0). BUT `/rtx/rendering/perSensorTickTlas` (per-sensor motion BVH) must stay **False** — with it True, Isaac **segfaults ~3 min in** (libomni.kit.loop-isaac). Verified by control test: multitick-off = stable 4.7min, both-on = crash, multitick-on+tlas-off = sensors publish AND stable 6+ min. Basic lidar/camera do not need motion-BVH. Re-enabling multitick did NOT bring back the odom TF_OLD_DATA flood the prior fix worried about (the kinematic `base_pose` odom is reset-safe). See [[project_isaac_gait_consume]].

3. **RTX `Lidar.create` needs `config=`, not bare `attributes=`.** A lidar authored from omni:sensor attributes has no firing pattern and casts no rays. Use built-in configs: `Example_Rotary` (3D points) + `Example_Rotary_2D` (2D LaserScan), then override `omni:sensor:Core:nearRangeM`/`farRangeM` on the prim post-create. Canonical example: `/isaac-sim/standalone_examples/api/isaacsim.ros2.bridge/rtx_lidar.py`. tick_rate must match the config scan rate (Example_Rotary = 10Hz; turtlebot urdf = 10Hz). Configs ignore the URDF's exact beam count/FOV; fine for nav2.

4. **camera depth topic** was `depth`, contract+gz want `depth_image`. Renamed in `camera.py`.

Verified working: lidar scan (360°, 0.164-12m, finite returns, ~4Hz render-limited), lidar/points (250k pts), camera image/info/depth_image/points (11-27Hz), odom upright. Debug: `carb.log_warn` is suppressed headless, only `carb.log_error` shows; `ros2 node list` does NOT show Isaac OG/replicator publisher nodes — use `ros2 topic info -v` (publisher count).

Still open (not turtlebot-blocking, documented for user): contact/bumper sensor still fails (needs a collision-bearing prim); camera frame uses the mount link `oakd_rgb_camera_frame` not the optical frame (verify orientation in RViz); `gazebo_ros2_control` controller_manager errors are a pre-existing control-layer mismatch, unrelated to sensors.

OPEN 2026-07-30: camera path parses only width/height/clip/rate/topics, horizontal_fov IGNORED (prim default optics), lidar parses full scan geometry. 159 arena_robots files carry the tag so a fix flips live rendering fleet-wide untested. Fix = parse tag + focal from aperture/2tan(fov/2), needs ONE isaac frame vs gazebo (unit quirks render plausible-but-wrong). camera_info self-consistent so divergence is silent downstream.
RESOLVED 2026-07-31 by user commit 93869ca "gazebo camera parity": horizontal_fov now parsed, focal set from aperture/2tan(fov/2), tag-absent path unchanged.
