---
name: project-scene03-floor-usd-conflict
description: scene_03 robot falls through the arena floor because the 1.3GB USD static entity breaks it; floor alone works
metadata: 
  node_type: memory
  type: project
  originSessionId: 15633cbd-faa8-46fb-85cc-1c74705fd669
  modified: 2026-08-01T23:50:53.000Z
---

2026-08-01 drop added `material: [Concrete_Smooth, {}]` to scene_03's world.yaml. It is the ONLY
urbanverse scene with a floor material (01/02/04/05/09 all ship `mat: ''` and spawn no floor).
Robot falls through to z ~ -0.94..-1.00 and rests there.

Bisected by measurement, all on isaac headless, jackal, odom `pose.pose.position` (world-derived,
see [[project_newton_fabric_consumers]]):

- ladder_01_res_arena (control, has floors): stands at z=+0.0707 -> arena floors work on isaac+newton
- scene_03, floor kept, USD static entity REMOVED: stands at z=+0.0718 -> floor works in scene_03 too
- scene_03 as-dropped (floor + USD): falls to -0.9377 (newton) / -0.9365 (physx)

**The USD static entity breaks the arena floor's collision.** Floor alone is fine.
Suspect `sanitize_asset` in SpawnPrims (see [[project_scene01_newton_crash_investigation]]) running
concurrently via `asyncio.gather` in `_spawn_world_obstacles` and stripping physics off the floor.
NOT yet confirmed: probe the collider's `HasAPI(UsdPhysics.CollisionAPI)` AFTER episode start, not
at spawn time (at spawn it reads True).

Dead ends, do not revisit: newton stale-guard starvation, MakeInvisible on the collider, missing
MeshCollisionAPI, nconmax, collider size, pause-window registration (a full stop/play rebuild
provably runs 35s AFTER the spawn and does not help), CollisionAPI never applied (it is applied).
Engine-independent: physx and newton fail within 1.3mm of each other.

Workaround: set `material: ""` in the installed world.yaml -> scene_03 behaves like its five siblings.

Side findings: floor collider bbox is offset, min corner pinned at (5,5) instead of covering the
zone (real bug, NOT the cause, robot falls inside the footprint too). `scenario_file:=` does not
plumb through `arena launch`. Drop ships no `annotation.yaml` so human-sim has no scene footprint.
Two publishers share `/jackal/odom` (isaac graph + planar `odom_relay`), read min z not last.
