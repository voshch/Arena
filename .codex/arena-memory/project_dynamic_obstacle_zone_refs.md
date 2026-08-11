---
name: Dynamic obstacle zone refs
description: Plan to allow dynamic obstacles to reference zones by name instead of explicit coordinates, resolved in TM_Scenario.reset() with controlled RNG
type: project
---

Allow dynamic obstacle poses and waypoints to reference world zones by name instead of requiring explicit coordinates.

**Why:** Scenario authors shouldn't need to look up exact coordinates. "doctor starts in exam_room_1, walks to patient_ward" is more natural and maintainable.

**How to apply:**

1. Add optional `pose_ref: str | None` and `waypoint_refs: list[str] | None` fields to `DynamicObstacle` (in arena_simulation_setup shared types). cattrs handles these as simple optional strings.

2. Create `resolve_dynamic_obstacles(obstacles, world, rng)` in **task_generator** (not arena_simulation_setup) — because the seeded RNG lives in task_generator. Samples concrete coordinates from zone polygons for any ref-based values. Pass-through for obstacles with explicit coordinates.

3. Call in `TM_Scenario.reset()` alongside `resolve_regions`:
```python
resolved_dynamic = resolve_dynamic_obstacles(scenario.dynamic, world, self._rng)
return scenario.static, resolved_dynamic
```

4. Zone polygon lookup tables can be built from `WorldDescription` directly, or factor the table-building out of `resolve.py` into a shared helper.

Key constraint: sampling must happen in task_generator (not during Scenario.load()) to keep it under the task_generator's seeded RNG for reproducibility.
