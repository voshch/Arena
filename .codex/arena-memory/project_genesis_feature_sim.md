---
name: project-genesis-feature-sim
description: "Genesis optional feature sim implemented (2026-06-12), uncommitted on feature/genesis; design and known gaps"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d265698-09bf-4737-84e8-7e1e932e5e46
---

Genesis (genesis-world 1.1.1) is a working optional feature sim, committed as d3fe52c "init genesis" on `feature/genesis` (2026-06-12, unpushed). Mirrors the mujoco feature layout: `arena_genesis/` (server pkg + `arena_genesis_msgs` with contracts identical to arena_mujoco_msgs), `arena_runtime/.../sim/genesis_simulator.py` adapter (1:1 port of mujoco_simulator), `_meta/features/genesis` (+ docker symlink).

Design facts:
- Genesis cannot add/remove entities post-`scene.build()`: scene.py keeps an EntitySpec registry, spawn/delete mark dirty, `ensure_built()` rebuilds a fresh gs.Scene and restores snapshots. Poisoned (NaN) scenes rebuild from spec poses, not snapshots.
- Quaternions are w-x-y-z natively in Genesis; conversion from geometry_msgs happens exactly once at the service boundary.
- Pedestrians: pool of `fixed=True` kinematic cylinders (set_pos works post-build and moves collision), parked spread at x=3i, z=-1000. Free-parked overlapping bodies caused NaN.
- CPU backend is default (`ARENA_GENESIS_BACKEND=gpu` to opt in); dt=0.01, RTF ~0.93 with sensors.
- Lidar = Raycaster sensor; Genesis raycasters do NOT exclude the mounting entity, mount must be raised above the chassis collision top ([[project-genesis-lidar-self-hit]]).
- First-ever scene.build compiles kernels (>60s GPU cold); run_genesis warms the cache before advertising services. Cache: ~/.cache/quadrants/.

Verified live: 11 episode cycles, lidar ~9.3 Hz, cmd_vel drives robot, server survives bad episodes. Parked 2026-06-12 ("let it rest"), known gaps for resuming:
- sensor mount resolution is jackal-shaped: on rbvogui_plus all sensor frames collapse to base origin (both cameras render identical frames, lidar self-hits). Fix = compose base-to-frame transform from the URDF joint chain in sensors/core.py `sensor_mount`/`_mount_T`. This is the agreed first fix on resume.
- GPU forced off (`_gpu_usable` early-returns False): quadrants 1.0.2 asserts `graph_do_while_flag_dev_ptr` in the full stack, though single/multi/churn/raycaster probes all pass, suspect articulated-contact solver CUDA graphs. `ARENA_GENESIS_BACKEND=gpu` still overrides.
- native viewer experimental: binds one scene at build, rebuild churn leaves it black, single-live-scene guard added, rviz is the supported viz.
- CPU lag with RGBD robots (rbvogui_plus): cameras hardcoded 640x480@10Hz, fix = res/rate from SensorSpec + render-on-subscribe.
- swerve (rbvogui 4WIS) moves but needs physics tuning, jackal diff drive is fine. Doors/elevators and multi-env untested, `odom` double publisher pre-existing.

2026-07-04: user states neither mujoco nor genesis backends are ready to ship. Box3D (Erin Catto, released 2026-06-30, C17, deterministic, no bindings/URDF/sensors) evaluated as alternative: recommended waiting for post-v0.1 stabilization; if pursued, scope as kinematic collision-aware-dummy backend (raycast lidar + contacts, no full dynamics), not a full dynamics port.
