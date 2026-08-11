---
name: newton-skidsteer-test
description: Isaac 6 Newton (MJWarp) jackal skid-steer E2E working 2026-07-03; gains, contact cfg, stop-cycle resume, teleport-via-USD, monotonic clock, all engine-gated
metadata: 
  node_type: memory
  type: project
  originSessionId: 26189c30-8f48-467e-852a-a6dc9b883705
---

E2E VERIFIED 2026-07-03 (uncommitted, migration/isaac6.0.0 + arena_isaac submodule): `arena launch sim:=isaac isaac.physics:=newton` runs jackal with active controllers, exact GT odom TF, nav2 explore traversing 28 m with obstacle avoidance, working episode resets. Skid-steer verdict: commanded 0.5 rad/s spin -> 0.453 rad/s ground truth (91%), straight 0.5 -> ~0.47 m/s, no NaN, no veer; wheel_separation_multiplier 1.5 about right at low speed (bench: scrub loss grows with wheel speed, 1.226/2.61 kinematic at wheels ±5 rad/s).

Engine gating helper: `isaac_utils.graphs.physics_engine()` (deferred SimulationManager import), used by control graph, odom graph, geom.move.

The five Newton-specific mechanisms (each was a hard blocker):
1. Velocity drives (control/__init__.py): kd=0.35 USD + DriveAPI maxForce 16 Nm under newton (physx keeps 1e4/unclamped). physx-scale kd NaNs wheels on contact, any unclamped gain backflips the base.
2. Solver cfg per stop-cycle (run_isaacsim `_newton_apply_solver_cfg`): cone=pyramidal, impratio=0.05 (MuJoCo convex contact does not enforce max dissipation, elliptic@1.0 cancels skid yaw, spin locked at 0.13 rad/s, CPU MuJoCo identical), nconmax=400 njmax=2400 (mjwarp tile kernels specialize on these sizes, uncold values = minutes of nvrtc on the main thread; /root/.cache/warp survives container restart, wiped by `arena feature isaac update`).
3. Resume = ALWAYS world.stop()+play() (run_isaacsim): newton's per-frame pump dies on plain pause->play (only warmup's 2 direct steps run). Stop invalidates, play does full warmup reparse of USD. `ns.sim_time` restored after (init zeroes it). `_NewtonStaleGuard` (structural resyncs only, skips Shader/Material/OmniGraph/Render subtrees) bounces play-time resyncs through a 30-frame pause hold.
4. Teleports land via USD (geom.move): tensor-view pose writes are LOST under newton (views dead while paused + rebuild-from-USD discards state), so newton forces the XformPrim USD write, applied by the resume rebuild. Pre-fix the robot sat forever at the SpawnUrdf placement outside map_empty free space and every ComputePathToPose failed silently (log_level warn hides planner INFO, diagnose via behavior_tree_log topic: 632x GoalUpdated FAILURE, ComputePathToPose RUNNING->FAILURE).
5. Monotonic time everywhere, three layers, all needed: (a) `resetOnStop=False` on our 5 explicit IsaacReadSimulationTime nodes (time/odom/joint_states/topic_bridge/tf graphs), else a mid-run stop-cycle publishes /clock backward (1355->0 observed) and silently wedges every use_sim_time consumer (controllers "active" but no output, FollowPath instant FAILURE, zero error logs). (b) RTX sensor writers stamp from a PER-RENDER-PRODUCT IsaacReadSimulationTime in the syntheticdata graph (default resetOnStop=True); NVIDIA's ROS2RtxLidarHelper configures it in post_attach but our direct LidarSensor.attach_writer path bypasses that, so `monotonic_sensor_time()` (sensors/__init__.py) applies `SyntheticData.Get().set_node_attributes("IsaacReadSimulationTime", {"inputs:resetOnStop": False}, render_product)` after every writer attach (lidar points+scan, both camera classes). Skew symptom: scan stamps lag /clock by exactly the played time before the last stop-cycle, collision_monitor spams "extrapolation into the past" and its stale-source fallback throttles all nav cmd_vel. (c) run_isaacsim restores the timeline playhead across stop (set_current_time between stop and play) for anything reading raw timeline time. ns.sim_time restore covers only newton internals, none of the above.

Odom under newton: see [[newton-fabric-consumers]]. Static colliders ingest fine (robot climbed + descended a 17 cm obstacle, contact stops it; walls share the same add_usd path).

Open/known: SpawnUrdf binds wheel friction to absolute `/colliders/<link>` with silent continue, URDF mu never lands under AS3.0 (newton runs default mu 1.0, both engines affected); IMU + collision_events are physx-backed, unverified under newton; DWB+jackal goal completion is pre-existing (ladder benchmark memory); RTF ~0.5 on this box.

Debug channels for the newton loop: carb.log_warn never reaches the launch console at log_level:=warn and kit file logging is disabled, stdout pipe-buffered away; use a temp file logger in run_isaacsim. py-spy needs `docker exec --privileged -u root` on the CHILD pid of python.sh. `ros2 control` CLI errors in this setup, query the controller_manager list_controllers SERVICE instead. `ros2 topic echo` truncates arrays at 128 without --full-length.
