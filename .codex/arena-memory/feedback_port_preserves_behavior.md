---
name: Porting preserves behavior — every line had a reason
description: When rewriting / replacing adapter or wrapper code, reproduce the original behaviors by default; don't silently drop them. The old code earned its warts.
type: feedback
originSessionId: 481f48ce-0d46-4cc0-b076-1c025647b313
---
When replacing existing adapter / handler / wrapper code, treat every line of the old version as load-bearing until proven otherwise. The person who wrote it hit the bug the hard way; dropping a line as "premature" or "task-mode concern" in a refactor means that bug will resurface and the user will have to re-teach you.

**Why:** this was the whole pattern during the `arena_robots` / task_generator adapter refactor. Every single regression was a behavior I dropped as "not strictly needed":

- `use_sim_time: true` on launched Nodes — obvious for a simulator, I didn't set it.
- Teleport-triggered `on_move` cancel + re-dispatch — present in old Nav2Adapter as "plan invalidated on teleport"; I dropped it and relied on reactive handler retry.
- Retry on nav2 rejection — old code had 30 attempts for nav2 warmup; I removed it as a "hack."
- Retry on nav2 ABORT — old code had resubmit-on-abort with the comment "nav2 kept getting the goal until it succeeded"; I dropped it.
- Arena IDL status vs rclpy transport status — old code had `GoalStatus.STATUS_SUCCEEDED` translated to arena status; I compared the wrong enums.
- `goal_handle.succeed()` / `abort()` / `canceled()` discipline — rclpy ActionServer needs this; old code did it implicitly via pattern, I skipped it.

**How to apply:** before a refactor lands, list every method on the old class and ask "what does this do, why is it there, does the new version still do it." If the new version's contract "doesn't need it," that's the claim to verify — not the default to assume. Every hack in the old code is a bug the old author already paid for.

Related: sim-paused-during-wait_until_ready, ROS glue in arena_rclpy_mixins — same pattern, different flavor. Default assumption: the codebase already knows; read it first.
