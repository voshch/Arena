---
name: momask-ros4hri-student-audit
description: "momask->ros4hri converters: convert_v2 is current (36-DOF wire, reviewed 2026-08-05), from_pos kept for visualize_rviz, from_ang dead"
metadata: 
  node_type: memory
  type: project
  originSessionId: 51b0895b-63c5-451c-b405-3844d0003d85
  modified: 2026-08-05T05:38:35.440Z
---

Student (ductaingn, github.com/ductaingn/momask) converts MoMask/HumanML3D output to the ros4hri joint format. Repo at ~/dev/momask. Three pipelines, deliberate layering:

- `convert_v2/`: THE current converter. Reviewed + desloped 2026-08-05: synced to the 36-DOF contract v3 ([[JOINTS.md]]), emits WIRE convention by default (all-zeros rest, what Pedestrian.joint_state carries) with `shoulder_convention="rig"` for adapter-bypassing renderers. Its `human.urdf` copy and LIMITS/ROS_JOINT_ORDER hard-validated against the Arena contract table (0 mismatches). validate.py FK is renderer-faithful (wire -> arm_solve_to_rig -> raw chain). Solver: hierarchical spine stack (lumped yaw split evenly, chest ball-fit pins the shoulder line, conditioning-damped segment roll), closed-form collars and two-DOF ankles.
- `convert_from_pos/`: older positions pipeline, kept ONLY because visualize_rviz.py and recorded .npy artifacts depend on its old format (root_xz_yaw keys, rig-convention shoulders). User decided 2026-08-05 to keep the split for now. Consolidation path if ever asked: port visualize_rviz to convert_v2 with shoulder_convention="rig", then delete the from_pos converter.
- `convert_from_ang/`: fatally broken (reads 6D rotations at [130:256] but true HumanML3D layout is root[0:4], ric[4:67], rot[67:193], local_vel[193:259], feet[259:263]). Kept as cautionary reference, never use.

Output key naming is a deliberate loud-fail: v2 emits `root_xy_yaw` (ROS axes), old pipelines `root_xz_yaw` (HumanML3D axes), so a consumer wired for one format crashes on the other instead of silently swapping axes.

Repo style differs from Arena: nested closures and x0= defaults are unannotated by local idiom, do not retrofit ANN annotations.
