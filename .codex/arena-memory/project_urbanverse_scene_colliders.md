---
name: urbanverse-scene-colliders
description: "Full collider/support census of urbanverse scene worlds; scene_03 stuck-zones diagnosis, red boundary wall, detected-walls are ped-only, mat:'' design pending"
metadata: 
  node_type: memory
  type: project
  originSessionId: d3de5f6d-eaa3-478b-8b4a-e3b95d5cec9a
  modified: 2026-07-31T11:11:05.267Z
---

Census 2026-07-31 (offline usd-core on install tree; scene_03 unless noted). Robot support in isaac = arena floor box top z=+0.01 (SpawnFloors thick-box, newton-compatible), catch-all add_ground_plane at z=-5 (run_isaacsim.py:279) is NOT support.

- scene_03 fly/stuck zones: scene road colliders (lane top -0.08, nearroad/nearbuffer tops exactly 0.00 in bands hugging carriageway edges y +5..8 / -7..-4.5 + west margin x<4) sit 1-9cm under the floor top; transient newton penetration snags/ejects. 12% of free cells directly hazarded, 62% within 0.5m. Clean ribbon y in [-4,+4], x>5; best teleports (81,0) (68,0) (99,0) (48,0).
- scene_01/02/04 have ZERO scene collider meshes under free space (robot rides arena floor box only). scene_03 is the outlier with 31% coverage.
- 142 collider specs authored, 79 active=False (USD-pruned everywhere, inert). Active=63. All 30 infinite CollisionPlanes horizontal below ground. Phantom collision copies (_02/_03 suffix, the 29 RigidBodyAPI prims) sit at garbage y-up transforms km off-map; scooter Object_02 = 1.54M-pt convexHull cook bomb (spawn-time blocking).
- scene_03/04 robot-vanishes-from-tree = heartbeat eviction: sync SpawnPrims compose+cook (main USD 1.3G/176M, millions of inlined collider points) blows heartbeat 5s / reset-hold 30s sweep budgets (sweep_verdict registry.py); scene_01/02 (5-6M USDs, payload-based) fit under. Confirm via 'evicted env_N (heartbeat_timeout)' in arena_node log.
- Red "fatass rectangle" around zone in isaac viewport = the scene USD's own /World/scene_03/Walls prim: 4 perimeter boxes 20m tall, displayColor (0.8,0.2,0.2), convexDecomposition collider, kept by sanitize_asset. NOT arena walls.
- Detected occupancy walls are ped-engine + rviz markers ONLY, never physics prims in isaac (human/__init__.py spawn_world forwards only authored walls to sim; collision_walls stay engine-side). walls:[] kills only authored walls. In isaac the ONLY physical robot containment in scene worlds = the USD's own colliders.
- SHIPPED 2026-07-31 (91f672d 'cleaner no-walls, no-floors', user-verified live): empty material = don't spawn. all_floors/all_ceilings/all_walls skip empty-name materials, new Zone.wall_material (default Marble) gates occupancy wall detection ('' on every zone suppresses), Zone structure hook aliases mat/ceiling_mat/wall_mat -> canonical fields. KEY: scenes' mat:'' was a FOREIGN KEY cattrs silently ignored pre-fix (canonical key is 'material'), so scenes actually had default-material floors; post-fix scene worlds are floorless -> scene_01/02/04 have nothing under free space (world.yamls need a real material or scene colliders). Documented in AUTHORING.md Key points.

See [[urbanverse-feedback-triage]] [[scene01-newton-crash-investigation]] [[isaac-env-wipe-is-heartbeat-eviction]].
