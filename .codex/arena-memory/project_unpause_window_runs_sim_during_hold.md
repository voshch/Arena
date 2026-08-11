---
name: project_unpause_window_runs_sim_during_hold
description: "to advance the sim during a reset hold use node.unpause_window(), not sim.step(); gazebo step() is broken (sends multi_step with pause unset)"
metadata: 
  node_type: memory
  type: project
  originSessionId: dc9ecfee-81f1-47b0-966e-000798f2b15e
  modified: 2026-08-01T12:32:40.397Z
---

During a reset hold the sim is paused at the arena_node lifecycle level (`hold("reset")` keeps gz `pause=True`). The way to actually run physics while a hold is active is `node.unpause_window()` (async context manager on the task_generator node, client to `/arena/sim_lifecycle/unpause_window`): ACQUIRE calls `_lifecycle.unpause()`, RELEASE re-pauses iff holds are still non-empty. `robot_move` (gazebo teleport) and isaac's `step()` both wrap their work in it.

Gotcha: **gazebo's `step()` does NOT open an unpause_window** (it just sends `multi_step` via ControlWorld), so `environment_manager.step()` during a hold does not advance the world. isaac's `step()` DOES wrap in unpause_window. This asymmetry made the first robot-bringup-during-reset attempt hang: controllers never activated because gz_ros2_control's `update()` only runs on a physics step, and the step never happened.

CORRECTED 2026-08-01 (source read, not live-verified): calling it a "no-op" was imprecise. gz-sim 8 `SimulationRunner::ProcessWorldControl` applies `SetPaused(control.pause)` FIRST, then queues `multiStep` only `if (this->Paused() && control.multiStep > 0)`. `ros_gz_interfaces/WorldControl.msg` has no default on `pause`, so python `WorldControl()` gives `pause=False`. `gazebo_simulator.py:645` sets only `multi_step`, so the guard never passes: the message degenerates into a plain **unpause that never re-pauses**, and still returns `success=True`. `pause=True` + `multi_step=N` in the SAME message is the only stepping combination (that is what "Paused after stepping multi_step" in the .msg means, and how the gz GUI step button works). Fix is one field. The observed "world never advanced" symptom is not fully explained by this reading (an unpause should have advanced it), so the live spike still has to run before trusting either account. Also: `max_step_size` is 0.0333 (`arena_bringup/configs/gazebo/empty.sdf`, always the loaded world since no per-world `.world` files exist), no python reads it, and there is no WorldStats subscription anywhere.

Fix used in `RobotsManager.launch_pending(drive_clock=True)`: open `async with self.node.unpause_window():` around the bringup gather so the sim free-runs while controllers come up, then re-pauses on exit. See [[project_launch_cancel_orphans]] and [[feedback_sim_paused_during_setup]]. Controller readiness gate is `RobotManager.wait_controllers_active()` (polls list_controllers until all active, grace fallback for control-less/dummy robots).
