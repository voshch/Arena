---
name: project-height-planner-oscillates
description: HEIGHT planner oversteers/oscillates in-sim; accumulator under-damps vs Jackal dynamics; open issue
metadata: 
  node_type: memory
  type: project
  originSessionId: c83ced68-0743-4e9b-a405-9a300932ea71
---

The HEIGHT DRL planner (`arena_planners/planners/height`) still oscillates/oversteers in gazebo even after the circling fix: omega slams between +/-1.0 and theta swings, with a clean ~2 m goal, peds flowing, and no obstacles. Root cause is the `desiredVelocity` accumulator (discrete +/-0.1 omega increments) under-damping against the Jackal's real dynamics; upstream's turtlebot sim has perfect kinematic tracking so the policy never learned to damp against a laggy plant. Tick rate is ~native 10 Hz (default `planner_rate_hz`), so that's not the cause.

Already fixed (verified in-sim): the 20 m out-of-distribution goal that made it spin (goal-offset clamp to 4 m, plus recentre robot_node, see [[project-planner-peds-namespace]]); peds now reach it (npeds=10). The clamp matters only when there is no global planner (the no-global-plan config gives the far final goal).

Untried candidate mitigation (not a faithful port, needs sign-off): re-seed the accumulator from the robot's measured angular velocity each tick to cap windup.
