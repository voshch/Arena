---
name: project-benchmark-discards-node-logs
description: Benchmark runs capture only the runtime launch; per-env node output (incl. the planner bridge) is lost
metadata: 
  node_type: memory
  type: project
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-07T18:04:51.378Z
---

`arena evaluation benchmark` writes `<run_dir>/runner.log` from the `arena_runtime.launch.py`
subprocess only (runner.py `_arena_log_file`). Per-env task_generator processes are spawned via
`/arena/spawn_env` by arena_node, and their output goes nowhere: `runner.log` stays a ~33-line
summary and `~/.ros/log/<stamp>/` holds only `launch.log`. The `arena_planners` bridge logger
lives inside the env process, so none of its diagnostics survive a benchmark run.

Consequence: grepping a benchmark log for bridge messages ("run_loop entered", "no action
received within") returns zero because the stream was never captured, NOT because the event
did not happen. Treat such a count as a null measurement, never as a refutation.

**Why:** cost a long detour on 2026-08-07 debugging why DRL planners stalled in the s-bend.

**How to apply:** to see bridge/env diagnostics, run `arena launch` manually with the console
attached instead of going through the benchmark runner. See [[project-drl-planner-offline-probe]].
