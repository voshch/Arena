---
name: project-benchmark-runner-owns-runtime
description: The benchmark runner launches its own arena_runtime; pre-starting one deadlocks env spawn
metadata: 
  node_type: memory
  type: project
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-07T08:32:53.897Z
---

`arena evaluation benchmark` launches its **own** `arena_runtime.launch.py` (runner.py `_run_steps`, ~line 1061) from `self._arena_passthrough`. Do NOT pre-start a runtime with `arena runtime` — two `/arena` nodes then compete for the same services, the env registers against the wrong one, and `task_generator_node` sits in `unconfigured` forever while the runner logs `still waiting for spawn_env` indefinitely.

Symptom of the duplicate: `ros2 node list` shows `/arena` twice, plus duplicated `gz_services_bridge` / `world_generator`.

The runtime gets only the passthrough args (suite `launch:` block + manifest + CLI `key:=value`). A stage's `map:` reaches the **env spawn** (`build_launch_args` emits `world:={s.map}`) but never the runtime, which therefore boots `map_empty` by default. That is by design — `all_maps_random.yaml` spans many maps — but for a single-map suite you can pin the sim by adding `world: <name>` to the suite's `launch:` block.

**Why:** cost ~40 min of dead runs before the duplicate was spotted in `ps`.

**How to apply:** just run the benchmark; let it own the runtime. Check `pgrep -fa arena_runtime.launch` returns exactly one line. See [[feedback_kill_container_processes_after_runs]] for sweeping between runs.
