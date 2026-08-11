---
name: project-goal-to-plan-bridge-dedupe
description: DRL planners got no global plan because goal_to_plan_bridge issued exactly one ComputePathToPose ever
metadata: 
  node_type: memory
  type: project
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-07T08:33:07.010Z
---

Under `driver: drl`, nav2 runs in `planner_only` mode with no `bt_navigator`, so `arena_robots/goal_to_plan_bridge.py` bridges `<ns>/goal_pose` to the `compute_path_to_pose` action (planner_server only publishes `plan` as a side effect of that action).

The original `_on_goal` set `self._last_goal = msg` **before** checking the server, then early-returned when `wait_for_server(timeout_sec=0.0)` failed. Since `publish_goal_loop` republishes the *same* goal at 1 Hz, the `_same_goal` dedupe discarded every subsequent message — permanently. Net effect: one `ComputePathToPose` at startup at most, then silence. Its own comment claimed "will retry on next message", which the dedupe made impossible.

Consequence: `plan` had **zero** messages (topic absent from the bag entirely) for every DRL contestant, so `SubgoalGenerator` fell back to the raw distant goal and planners circled. All static checks passed — manifests had `depends.global_plan: true`, namespaces matched, planner_server and the action server were both up. Only the recorded bag revealed it.

Fixed (2026-08-07) by replacing the dedupe with a 1 Hz `replan_period` timer plus an `_in_flight` guard, matching how `bt_navigator` drives the planner. Verified: `plan` at 1.003 Hz, 179 msgs per 180 s episode.

**How to apply:** when a DRL planner wanders aimlessly, check `ros2 bag info` for a `plan` topic before suspecting the policy. Absent topic != bad planner. Related: [[project_benchmark_runner_owns_runtime]].
