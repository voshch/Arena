---
name: isaac-gait-consume
description: "Isaac renders pedestrians via native omni.anim.people AnimGraph from pose+twist; does NOT consume Pedestrian.joint_state"
metadata: 
  node_type: memory
  type: project
  originSessionId: adb09fd1-2764-4e81-b411-5d8e3587be79
---

Isaac pedestrians animate with the native omni.anim.people AnimGraph, but translation is owned kinematically (changed 2026-06-11): the AnimGraph's root motion tops out at the Walk clip's pace, so any chase scheme (catch-up Walk values, slides, snap thresholds) lags agents commanded faster than ~1 m/s. `Person.update` dead-reckons the commanded pose+twist between `UpdatePedestrians` calls and pins the character every physics step; `PathPoints`/`Action`/`Walk` provide gait and heading visuals only. No speed limit, exact sync with arena_peds, foot-slide when commanded speed exceeds gait pace is the accepted cosmetic cost. `Person.update_command(position, velocity, stamp_sec)` replaced the waypoint deque API.

Jitter relief (2026-06-12): resetting `_command_age=0` at service receipt rewound the character by velocity times the (variable) transport delay on every 50 Hz delivery, a visible sawtooth. UpdatePedestrians.srv now carries the Pedestrians header stamp and `update_command` seeds the age with `world.current_time - stamp_sec` (both sides share /clock). Separately, `PathPoints` was re-seeded every physics step (restarting the locomotion blend, gait/heading stutter); now only refreshed on >5 degree heading change or within 0.5 m of the stale look point. Remaining known wobble source if it resurfaces: graph root motion added after the pin within a frame, cancelable by pre-compensating with the `update_state` readback residual (not done).

CRITICAL: once the AnimationGraph is attached, the character root lives in Fabric and is owned by the graph. USD `xformOp:translate`/`orient` writes do NOT reach the rendered character (they produced wildly wrong positions when tried as the correction channel). The only authoritative channel is `character_graph.set_world_transform(carb.Float3, carb.Float4)` (verified present in omni.anim.graph.core 107.3, host install at ~/isaac/extscache, binding tests show set/get round-trip). `Person.set_world_pose` must also write through the graph when `_character_graph` is already attached, USD-only teleports silently don't move the character.

Isaac does NOT consume `Pedestrian.joint_state`. The skeletal-consume path (SkeletalPoseBackend authoring UsdSkelAnimation from the Arena gait) was tried and reverted: omni.anim.people renders from Fabric, so plain-USD UsdSkel writes are ignored and characters T-pose. `GaitGenerator` stays the ROS4HRI/rviz articulation ground truth only; the 3D engines (Isaac native AnimGraph, Gazebo walk.dae clip-scrub) render plausible locomotion from pose+twist, not bone-for-bone identical to the canonical gait.

Inverting (deriving ROS joint_state from a sim) only pays for Isaac via Fabric readback, and only if a perception/SDG pipeline needs labels matching the rendered mesh; Gazebo has no actor-bone readback. Not done. Related: [[project_ros4hri_flat_frames]], [[project_prompt_per_simulator]], [[project_isaac_no_oneway_ceilings]].
