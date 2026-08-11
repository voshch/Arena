---
name: project_launch_cancel_orphans
description: cancelling a LaunchService run_async task orphans its child processes; must call ls.shutdown()
metadata: 
  node_type: memory
  type: project
  originSessionId: dc9ecfee-81f1-47b0-966e-000798f2b15e
---

`launch.LaunchService.run_async()` handles `asyncio.CancelledError` by logging "run task was canceled" and `break`ing out WITHOUT calling `_shutdown()` (unlike the generic-exception branch). So **cancelling the launch task does not terminate the launched child processes** (controller_manager, task_server, nav2 stack, etc.) — they orphan.

This is harmless at full node shutdown (the Python process exits, so children die anyway), which is why `AsyncLaunchManager.kill_all` "worked". But for per-robot despawn teardown it left stale nodes under the robot's namespace; respawning the same name (next_name reuses the lowest-free `<model>_<n>`) collided → "more than one action server for .../goto_pose", controller spawner "Failed to acquire lock", controller_manager/list_controllers unreachable.

Fix (committed): `arena_rclpy_mixins.Async.LaunchHandle.shutdown()` calls `ls.shutdown()` (returns an awaitable when in-loop; emits the Shutdown event → SIGINT/TERM/KILL escalation on children) then awaits the task. `RobotManager.destroy()` uses it; `kill_all` now graceful too. See [[feedback_kill_container_processes_after_runs]].
