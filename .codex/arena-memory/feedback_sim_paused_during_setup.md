---
name: Reset pause is interrupted by one sim step
description: _reset_task pauses, sets up robots, then calls environment_manager.step(1) before tm_robots.reset, so TF/costmap get one tick to propagate during setup
type: feedback
originSessionId: 09c958ae-7ce5-40b3-b120-1db4dee6fff5
---
`Task._reset_task` pauses via `before_reset_task`, runs `robots_manager.set_up` (spawns/launches new robots), then calls `environment_manager.step(1)` to advance the sim by exactly one tick. After that tm_robots.reset teleports/dispatches and `after_reset_task` unpauses.

**Why:** nav2 lifecycle activation (costmap_2d in particular) needs TF from the robot's frames, which comes from bridge messages driven by sim stepping. A strict "paused during setup" invariant starved new-robot spawns of TF, hanging `wait_until_ready` on lifecycle ACTIVE forever. The first reset only worked by accident (gazebo's pause service wasn't bound yet, so the sim kept stepping).

**How to apply:**
- `wait_until_ready` can now wait on TF-backed lifecycle states — the one step gives enough propagation for activation to complete.
- Don't design readiness probes that need *many* ticks; one is the budget.
- Don't revive the "never wait on runtime data" rule — that's no longer the invariant.
- `BaseSim.step(n=1)` is the abstraction; default is a no-op, simulators that pause physics (gazebo, isaac) override it.

Nav2 + dummy sim is still a mismatch: dummy's `step` is a no-op so TF never flows. Use `none` adapter for tf-less environments.
