---
name: project-drl-planner-offline-probe
description: "Registry planners can be driven closed-loop with no simulator, which separates policy bugs from harness bugs"
metadata: 
  node_type: memory
  type: project
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-07T18:05:03.049Z
---

A registry planner's `step(features)` is a pure function, so it can be rolled out closed-loop
offline: synthesise `robot_pose`/`robot_state`/`goal_pose`/`laser_scan`/`pedestrians`, integrate
the returned action at the bridge's 10 Hz, and re-raycast. Run it under the planner's own venv:

    PYTHONPATH= /opt/arena_ws/install/arena_planners_<pkg>/venv/bin/python probe.py <name>

(package is `arena_planners_<name>` with hyphens as underscores; `ros2 run <pkg> python` is a
shim onto that venv). Harness lives in `_meta/repos/sbend_probe.py` and friends (untracked).

This is the cheapest way to tell "the policy is broken" from "the harness feeds it something
bad", and it does not contend with a running simulation. On 2026-08-07 it exonerated most of
the DRL field: 12 of 14 reached the goal offline through a bend of the corridor's real width,
including planners that time out in the benchmark.

**Why:** static reading of the wrappers could not settle whether the timeouts were policy or
plumbing; the offline rollout settles it in seconds per planner.

**How to apply:** ALWAYS validate the synthetic route against the world's walls before trusting
a result. A hand-drawn route that clips a wall makes every planner "collide" at the same point,
which reads as a damning policy result and is actually a harness bug. The tell is a planner
known to succeed in sim failing offline. See [[project-benchmark-discards-node-logs]].
