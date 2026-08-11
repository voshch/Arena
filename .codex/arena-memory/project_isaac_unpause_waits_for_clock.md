---
name: isaac-unpause-waits-for-clock-not-a-timeout
description: "controller bringup gates on an observed /clock step, and the unpause path waits unbounded because isaac services it inline on its blocked main loop"
metadata: 
  node_type: memory
  type: project
  originSessionId: ec1410d5-4f4a-4dc7-a945-657de4ad4d8e
  modified: 2026-07-21T21:24:22.809Z
---

Isaac services pause/unpause/step inline on its single main loop (`rclpy.spin_once` interleaved with `simulation_app.update()`/`world.step()` in `run_isaacsim.py`). While a heavy spawn/warmup step blocks that loop, NO service is answered for an unbounded time, so a fixed unpause timeout is always a guess. The unpause handler itself is a trivial flag flip (`_running = True`); the main loop does the real work.

Fix shipped 2026-07-21 (main repo b6c30d9 on jazzy): unpause waits unbounded instead of timing out, and controller bringup gates on an observed clock advance.
- `ClientWrapper.call_forever` (Async.py): await a service response with no timeout.
- `IsaacHost.unpause` uses it; tg `unpause_window()` ACQUIRE uses it. `_cb_unpause_window` now honors the real unpause result.
- `TimeNode.await_sim_step(timeout_sec=None)` (Time.py): blocks until `/clock` advances past its current value; None default = wait forever.
- `RobotManager._launch_robot`: opens the unpause window, `await await_sim_step()`, THEN launches controllers. So spawners meet a live clock and never burn their 5-attempt lock-retry budget against a frozen one. `wait_controllers_active` is now poll-only (caller holds the window).

Why unbounded is safe: an isaac CRASH tears down the launch via the isaac ExecuteProcess `on_exit=[Shutdown()]`, which cancels the await. Only an alive-but-deadlocked isaac needs a manual cancel. The instant-ack side-thread route was rejected: the ack would say "flag flipped" not "sim stepping", a weaker gate. Dummy sim drives `/clock` via arena_node `_publish_clock_loop` (gated on paused flag), so await_sim_step resolves there too once the window unpauses it.

Do NOT reintroduce a finite timeout on the unpause path. See [[feedback_sim_paused_during_setup]], [[project_isaac_env_wipe_is_heartbeat_eviction]].
