---
name: Obstacle reset semantics
description: How unuse_obstacles and remove_obstacles must handle WORLD vs INUSE vs UNUSED layers — avoid nuking world obstacles during reset
type: feedback
---

`unuse_obstacles` must only affect INUSE-layer obstacles. WORLD obstacles must survive resets untouched.

**Key rules:**
- `_remove_obstacles_impl` and `_remove_pedestrians_impl` are separate abstract methods (static vs dynamic)
- `unuse_obstacles` removes only UNUSED statics from human sim (stale from previous cycle), forgets them, then downgrades INUSE→UNUSED with `spawned=False`
- `spawned = False` must NOT be set on WORLD obstacles — only on the ones being downgraded
- `remove_obstacles(purge)` handles actual removal of stale UNUSED from both human sim and base sim
- hunav tracks obstacle-derived walls separately (`_obstacle_wall_*`) from explicit walls (`_explicit_wall_*`), merged into cached `_wall_segments`/`_wall_points` via `_rebuild_wall_cache()`

**Why:** WORLD obstacles were getting annihilated on every reset due to three bugs:
1. arena_humansim's old `_remove_obstacles_impl` nuked everything (statics + dynamics) in one call
2. `spawned=False` was set on all obstacles including WORLD ones
3. arena_humansim's `RemoveObstacles` service interprets `names=[]` as "remove all" — so `_remove_obstacles_impl` must early-return on empty names, never send an empty list to the service

**How to apply:** When touching obstacle lifecycle code, trace the full flow through `unuse_obstacles` → callback → `remove_obstacles(UNUSED)` and verify each layer is handled correctly. Never blanket-reset state across all obstacles. Always guard service calls that treat empty input as "all".
