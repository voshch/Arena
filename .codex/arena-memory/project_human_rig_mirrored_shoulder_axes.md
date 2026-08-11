---
name: human-rig-mirrored-shoulder-axes
description: "ros4hri human_description rig mirrors arm-chain axes L/R but not legs, and p_shoulder is abduction not sagittal swing, JOINTS.md role column is wrong"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ef89079-d14d-4ad8-8d0a-21291318923e
---

In ros4hri/human_description `human-tpl.xacro`, all joint origins are `rpy="0 0 0"` (frames body-aligned at rest, REP-155 X fwd / Y left / Z up). The arm macro reflects axes per side (`p_shoulder` axis `${reflect} 0 0` → left (1,0,0), right (-1,0,0)), while the leg macro hardcodes the same axis for both sides (`r_hip`/`knee` = (0,-1,0)). Consequences:

- Equal commands on l/r `p_shoulder` = mirror-symmetric motion; a π phase offset double-mirrors into in-phase world motion (verified 2026-07-02: GaitGenerator arms rendered in unison).
- `p_shoulder` is abduction (rotation about body-forward X), NOT sagittal swing; JOINTS.md's "arm sagittal swing" role label is wrong. The rig has no lateral-axis shoulder DOF; the only sagittal (0,-1,0) arm joints are the elbows.
- The shoulder triple (y about Z, p about X, r about Z) is a full ZXZ Euler set: constant y_shoulder = r_shoulder = pi/2 on BOTH sides conjugates p_shoulder into a true sagittal axis (0,-1,0), same as the legs, elbow axis preserved at p=0. r_shoulder clamps at 1.5 (4 deg error, invisible).
- IMPLEMENTED (2026-07-02, uncommitted): Pedestrian.joint_state = animation SSOT, wire values are SEMANTIC anatomical angles (JOINTS.md section 1 normative; p_shoulder = sagittal flexion positive-forward same-sign, antiphase in values). GaitGenerator math unchanged (was already semantic).
- RViz: rviz_utils/hri/rig.py semantic_to_rig emits constant pi/2 for {l,r}_{y,r}_shoulder (ZXZ conjugation makes p_shoulder sagittal), wired into BOTH hri_producer branches; exact pi/2 intended, robot_state_publisher ignores URDF limits. rviz_utils gained tests/ + pyproject (task_generator layout); pytest only importable in-container (hri/__init__ eagerly imports body_pool -> xacro).
- Isaac: ExternalPoseProvider seeds mapped bones from the NEUTRAL standing pose = idle clip frame 0 (clips["idle"].rotations[0]), replace-not-compose. NEVER seed from skeleton restTransforms: CMU rest is a T-POSE, rest-seeded peds render crucified with a wiggle (GUI-observed 2026-07-02). JointPose.rotations are absolute joint-local incl rest. CRITICAL catch: joint_order tokens are FULL PATHS (convert.py authors joint.path) while BONE_MAP targets leaf names, the provider matched full tokens so the whole external path was a silent production no-op; fixed via leaf-name bone_index (rsplit "/"), regression test with full-path tokens added. bone_map axes are now DERIVED, not guessed (2026-07-02): each desired world axis (ros4hri body frame, p_shoulder = (0,-1,0) both sides) mapped through the bone's neutral skeleton-global rotation transpose, from the fuel walk.dae/stand.dae skeleton; FK-verified 8/8 (heel back+up, hips/arms forward, abduction outward); regenerate if the arenian skeleton source changes (derivation script was scratchpad-only). test_external.py asserts via _expected_delta reading BONE_MAP, never hard-coded axes.
- Gazebo: clip fidelity only, gz-sim 8 actors have no per-bone control (PedSkeletonPlugin.cc header).
- Rejected alternatives: gait-off-the-wire (kills SSOT extensibility), elbow-only swing (degrades Isaac), forking upstream human_description.

- BONE_MAP REGENERATED 2026-07-19 (staged on feature/ped-gui): user caught knees bending backward in Isaac while steering. Parity harness arena_isaac/tests/peds/test_bone_map_parity.py (pytest + `report`/`regen` script modes, run in-container with PYTHONPATH appended not assigned) measures per-DOF world delta axes on BOTH rigs (contract = xacro+semantic_to_rig+FK, isaac = ExternalPoseProvider over fixtures/skeleton_neutral.json snapshot of a bundle meta.json) in anatomically-derived body frames (up from spine, forward from toes, so L/R swap detectable without trusting bone names). Old table: all 6 leg sagittal DOFs SIGN-INVERTED (dot -0.95..-0.99, the 2026-07-02 derivation was against the fuel walk.dae skeleton, not the roster bundle), arm/elbow axes tilted (0.65-0.92), reserved y/r shoulders actively mapped though the contract no-ops them. New table: measured from contract, all DOFs dot 1.00, reserved -> None. Regenerate via the harness whenever skeleton source or contract rig changes.

Related: [[usdskel-peds-pipeline]].
