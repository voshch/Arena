---
name: project_gz_actor_root_freeze
description: "gz component-driven actors pin root to DAE bind, fix = freeze clip root, never rebase the DAE bind"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2b0ccdf4-5b9d-4112-a136-490780341021
---

gz-sim's component-driven actor path (AnimationName/AnimationTime, used by PedSkeletonPlugin) weight-zeroes the skeleton root and pins it to the skin DAE's full bind transform, discarding the clip's root channel. Two consequences for arena_humans bundles:

1. **DEAD END, do not retry:** rotating the DAE bind upright (redistributing Hips rotation into children, even world-bind-preserving) mangles animation, gz's BVH-to-skin alignment conjugates through the skin's *local* bind frames, so any local-frame redistribution breaks the composition regardless of how clips are rebased. Tried 2026-07-04, mangled twice.
2. **Correct scheme (implemented):** leave the DAE bind untouched, retime clips first (needs real root travel), then freeze the clip root channel (rotation identity, translation = bind_R.T @ bind_t) so script and component paths render identically, pelvis locked at rest orientation like arenian's walk.dae. Per-model ground lift = mean walk-clip hip z minus bind z, emitted as the actor SDF `<pose>` z, consumed by PedSkeletonPlugin as per-template lift (kZOffset 1.01 is arenian fallback only).

Also: gz names BVH skeleton animations by resolved file path, DAE clips by SDF `<animation name>`, plugin translates via ResolveAnimName. See [[project_arena_humans_pipeline]], [[project_worker_male_regen]].
