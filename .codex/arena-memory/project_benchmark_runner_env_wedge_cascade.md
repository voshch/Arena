---
name: project_benchmark_runner_env_wedge_cascade
description: benchmark runner cascaded all steps to t=0.0 false-fails when one env wedged; fixed with respawn-on-systemic
metadata: 
  node_type: memory
  type: project
  originSessionId: 78d1d37e-bb50-4600-8935-4eeb6e8cc0c3
---

arena_evaluation benchmark/runner.py: all steps sharing (contestant,robot,sim) run in ONE
reused env (group_pending), switching worlds via QueueEpisode. A timed-out episode whose
cancel did not settle within `_CANCEL_SETTLE_S` (30s) left the task_generator run_episode
action server with `action_in_flight=True` (node.py `_goal_callback` rejects while in flight),
so EVERY later goal was rejected -> all remaining steps logged `episodes=0/3 t=0.0s` and the
sim sat on an empty world. Symptom user hit 2026-06-23: 1 partial + 47 false-failures.

Fix (uncommitted on jazzy): `_run_episodes` now returns a systemic ENV_SETUP StepResult on a
rejected goal OR a cancel that won't settle (instead of swallowing + continuing). `_run_group`
now, on any _SYSTEMIC step failure, despawns the wedged env and respawns a fresh one for the
remaining steps (new helpers `_spawn_and_setup_env` / `_despawn_env`) instead of skipping the
rest. This also changed FATAL handling from skip-rest to respawn-rest. Trade-off: respawns
Gazebo per wedge (slow if many wedge, but completes the suite correctly).

Trigger that caused the wedge in the first place: [[project_ladder_benchmark_const_map_origin]]
(navigation failures -> timeouts -> the hung cancel). That root cause is still open.
arena_evaluation is excluded from root `ruff check .` (submodule), so this file isn't linted by CI.
