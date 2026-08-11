---
name: project_humansim_engine_stamp_epoch
description: "humansim engine stamps AgentStates in its own from-zero tick timeline, Arena adapter re-stamps on arrival, engine unfixed"
metadata: 
  node_type: memory
  type: project
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

The arena_humansim engine (Arena/humansim, upstream rule applies per [[feedback_no_arena_humansim_modifications]]) stamps AgentStates with `_sim_time_ns = tick_count * dt`, a timeline starting at ZERO when the engine node starts, in subsystem mode too where the orchestrator owns /clock (agent_manager.py `_subsystem_timer_callback`). Since the engine starts long after the sim, stamps run tens of seconds behind /clock. Consequences before the fix (2026-07-02): Isaac's stamp-seeded `command_age` pinned at MAX_EXTRAPOLATION 0.5s giving every ped a constant ~0.7m lead over the SSOT markers (lidar visibly ahead), and the adapter's 50Hz interpolation alpha always clamped to 1 (interpolation was a no-op). Fixed Arena-side: `_agent_states_callback` in task_generator/simulators/human/arena_humansim/arena_humansim.py re-stamps incoming msgs with `self.node.sim_time.to_msg()`. Engine-side fix (seed `_sim_time_ns` from node clock in subsystem mode) would be cleaner but needs explicit approval. Residual marker lag after fix is one engine tick (dt default 0.05s, ~7cm at walk speed).
