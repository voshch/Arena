---
name: scene01-newton-crash-investigation
description: "Student's Isaac crash with scene_01 world on newton; not repro'd; scene's stray lights/rigidbodies diagnosed, spawn sanitize staged"
metadata: 
  node_type: memory
  type: project
  originSessionId: b5205d27-9cf1-4976-b302-670b756d398f
  modified: 2026-07-21T18:24:59.902Z
---

Student's machine (125GB RAM, RTX 5090, driver 580.105.08, Isaac 6.0.0) crashes ~39s in with `headless:=false` + newton + scene_01 + jackal; breakpad minidump written, lastCommands = IMU publisher graph wiring.

Findings (2026-07-21):
- scene_01 world zip (UrbanCousin collected scene, DUCANH pipeline) claims "visuals only, collision from occupancy map" in world.yaml but the composed USD carries 29 enabled kinematic rigid bodies + 201 colliders (111 raw trimesh, 29 Plane, 58 convexHull). Redundant vs runtime walls and double-collision-prone; should be stripped regardless.
- NOT reproduced locally on the 31GB box: headless newton + scene_01 + jackal survived the exact crash window (IMU graph wired, EPISODE STARTED #1) before being cgroup-OOM-killed at 16G ~10min in. Newton 0.7.14 handles the scene's trimesh/plane colliders headless.
- Local box cannot host this scene: Isaac base ~11.5G, +scene reaches 16G and keeps growing after episode start (possible leak, unverified). 14G cap = OOM during scene load.
- `physics:=newton` at arena-launch level is a SILENT NO-OP; correct arg is `isaac.physics:=newton` ([[semantic-layer-v1]] era launch args). Student's exec did receive newton.
- Remaining differentiators for student crash: headless:=false (RTX viewport + vMaterials MDL compile of collected scene), version skew of their build vs a16196c/753dcfc pins.
- Observed during local run (not the student's crash): one OgnIsaacArticulationController failure on jackal, fabric IStageReaderWriter v0.16-vs-v0.14 warning, /isaac/PauseSimulation 60s timeout ([[isaac-env-wipe-is-heartbeat-eviction]] service-backlog family).

Rendering diagnosis (2026-07-21, same zip): scene_01 dark under newton+Stage Lights = scene ships its own light rig (53 lights composed: 5 domes incl. an *invisible* DomeLight_01 at intensity 50000 x exposure 35.5 = ~2^35.5 nuke, 4 distants, giant 164m SphereLight sun at 14.7km, 8x radius-50 intensity-30k fills). Mechanism: RTX one-dome-per-scene conflict with arena's /World/Light_1 + suspected FSD light-visibility sync gap under newton (invisible nuke dome comes alive, auto-exposure crushes scene to black; Camera Light mode bypasses stage lights entirely, hence "works"). Warped roofs under newton = the 27 kinematic rigid bodies: newton ingests them, republishes poses to fabric worldMatrix at its own body frame, stomping authored xforms ([[newton-fabric-consumers]]); physx leaves kinematics alone. FIX STAGED (arena_isaac isaac6.0.0): `sanitize_asset` in isaac_utils/utils/prim.py, called from SpawnPrims prim_importer - deactivates UsdLux lights, RemoveAPI RigidBodyAPI, KEEPS CollisionAPI (standard arena models rely on authored colliders for robot collision; verified SM_Chair etc.). Validated offline vs real USD: 53->0 lights, 27->0 bodies, 98 colliders kept. /isaac/SpawnUsd service is dead code (client never called, body would AttributeError). Live sim verification pending. REVISED 2026-07-31 (uncommitted): light deactivation dropped by policy - assets keep their lights (trust authored lighting, incl. dark/night scenes); sanitize_asset now only strips RigidBodyAPI and returns has-active-lights; SpawnPrims deactivates fallback dome /World/Light_1 when the asset ships any light. scene_01 exposure-35.5 washout can return; fix belongs in the asset.

Next step: student must provide kit log (lines before the [crash] section) and the .dmp.zip from /isaac-sim/kit/data/... - minidump names the faulting module (renderer vs newton vs ros2 bridge). A/B on their box: headless true/false, physx vs newton, physics-stripped USD.

State left behind: scene_01 world files removed from source + install trees on user request (zip remains in ~/Downloads); `mem_limit: 16g`/`memswap_limit: 16g` still present (uncommitted) in _meta/docker/features/isaac/docker-compose.yml.
