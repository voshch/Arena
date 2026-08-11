---
name: newton-fabric-consumers
description: Newton writes poses ONLY to fabric omni:fabric:worldMatrix; which pose readers track vs freeze, and the odom graph pattern that works
metadata: 
  node_type: memory
  type: project
  originSessionId: 26189c30-8f48-467e-852a-a6dc9b883705
---

Under Isaac 6 Newton, physics body poses go ONLY to Fabric `omni:fabric:worldMatrix` (FabricManager kernel, selects prims with worldMatrix + newton:index). Measured with a falling-cube probe (scratchpad read_pose_test.py pattern) and live:

- TRACKS: omni:fabric:worldMatrix, `isaacsim.core.nodes.IsaacReadWorldPose` (reads it), RTX sensors/rendering.
- FROZEN at spawn/stop value: USD xformOps, `omni.graph.nodes.GetPrimLocalToWorldTransform`, usdrt Rt.Xformable decomposed attrs (GetWorldPositionAttr, written once by SetWorldXformFromUsd at init).

Consequences in arena (fixed 2026-07-03):
- odom graph (isaac_utils/graphs/odom.py): under newton the live body pose comes from IsaacReadWorldPose; GetPrimLocalToWorldTransform stays wired for the one-time body->base offset capture (USD spawn poses are correct under both engines). The base_pose ScriptNode (transform.py) takes body_translation/body_orientation when `body_is_matrix` false, captures the offset from USD matrices ALWAYS, and falls back to the USD pose while the fabric read returns exact identity (ticks before first warmup, else the first-tick identity poisons the offset by |spawn| meters, seen as a 41 m TF error rotating with yaw).
- `PROBING fabric from python`: Rt.Xformable.GetWorldPositionAttr is the WRONG attr (stale); read prim.GetAttribute("omni:fabric:worldMatrix") on a usdrt stage.

Under physx none of this applies (physx writes USD xforms in our config, GetPrimLocalToWorldTransform is live), hence the engine branch. Related: [[newton-skidsteer-test]].
