---
name: go1-newton-walking
description: go1 walks on Newton AND PhysX as of 2026-07-03 with staged fixes; Newton costmap marking corruption still open
metadata: 
  node_type: memory
  type: project
  originSessionId: 26189c30-8f48-467e-852a-a6dc9b883705
---

go1 E2E user-verified 2026-07-03 on BOTH isaac.physics:=physx and newton with the staged state, KEEP IT: thick invisible meshbox floor collider (SpawnFloors), impratio 0.05 + pyramidal + startup solver-cfg application (run_isaacsim), `open_loop_control: true` in go1 control.yaml, effort-scaled gains UNCHANGED at kp=4x effort (the quasi-kinematic kp~100x bench regime was NOT needed live, do not ship it). PhysX resume graph rebuilders also validated by the physx run.

Root cause of the original step-in-place: champ streams 200Hz wall-clock single-point trajectories, at RTF~0.25 they land 4x denser in sim time and closed-loop JTC segment replacement compounds tracking lag into ~4x amplitude attenuation (FL_thigh 0.092 vs 0.397 rad). Gazebo masks it (RTF~1). Any sim-time robot with a wall-clock trajectory streamer can hit this: check open_loop_control first.

Dead ends (don't revisit): impratio sweeps (high values "move" only by sinking), swing amplification, floor mesh subdivision, champ clock/config (gait was always nominal in sim time). Offline gait-replay bench scripts: isaac container /tmp/repro/gait_*.py on the live-imported USD.

OPEN: Newton costmap marking corrupted (map_empty + go1 + explore): white concentric rings + costmap marks smeared in arcs around the robot, physx clean. Symptom pattern suggests depth-camera ground returns entering the marking height band or TF/stamp skew on the Newton pose path (odom = ReadWorldPose + USD offset capture, [[newton-fabric-consumers]]). Not yet investigated. go1 camera image inversion is URDF-side, consistent across sims, ignore.
