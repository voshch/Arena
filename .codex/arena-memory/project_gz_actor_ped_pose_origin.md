---
name: project_gz_actor_ped_pose_origin
description: "gz pedestrian actors' TF/pose resolves to world origin (env_0/ext_arenian_*); gz actor limitation, user chose to leave as-is"
metadata: 
  node_type: memory
  type: project
  originSessionId: b451dde0-63b6-46ff-96d0-7b7472e83d68
---

Gazebo pedestrian "arenian" actors are driven by PedSkeletonPlugin (gazebo/arena_gz_plugins/src/PedSkeletonPlugin.cc) via `components::TrajectoryPose` + `components::AnimationTime`. gz-sim 8 uses TrajectoryPose as the authoritative render pose for trajectory-following actors and never writes it back to `components::Pose`, so anything tracking the entity Pose / pose_info (TF) sees the unchanged spawn pose, i.e. world origin. That is why e.g. `env_0/ext_arenian_b7639d` resolves to origin instead of tracking.

This is a gz actor limitation, not an Arena bug. A workaround exists (also `SetComponent<components::Pose>` each PreUpdate with the *un-lifted ground* pose, not the feet-planting z-lifted TrajectoryPose value) but it risks a double-transform on the visual and a ~1 m-high TF if the lifted pose is copied verbatim.

User decided 2026-06-22 it is too risky and is happy as-is: do not re-flag or propose fixing it. Related: [[project_arena_peds_name_is_sim_path]], [[project_hunav_raw_extra_frame_leak]].
