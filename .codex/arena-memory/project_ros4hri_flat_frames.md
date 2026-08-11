---
name: project_ros4hri_flat_frames
description: "ROS4HRI pedestrian TF frames stay flat (head_env_0_agent_1), per-env isolation is at the topic layer only"
metadata: 
  node_type: memory
  type: project
  originSessionId: adb09fd1-2764-4e81-b411-5d8e3587be79
---

Decision (user, 2026-06-09): keep ROS4HRI pedestrian skeleton TF frames FLAT, e.g. `head_env_0_agent_1`, `body_env_0_agent_1`. Do NOT rewrite them to a hierarchical `env_0/agent_1/head` robot-style path.

**Why:** flat id-suffixed frames in one global TF tree are ROS4HRI's actual convention (`human_description` xacro suffixes every link with the body id; rsp publishes link names verbatim). Per-env isolation lives at the TOPIC layer (`/arena/env_0/humans/...`), not the frame layer. The global TF tree is inherent to ROS4HRI. Body ids are uniquified per env (`env_0_agent_1`) so frames never collide across envs.

**How to apply:** don't propose hierarchical frame namespacing for pedestrians; it would deviate from the standard and break stock ROS4HRI consumers expecting `body_<id>`/flat links. The single `Skeletons3D` display also shares one "TF Prefix" across all bodies, so per-body identity must stay in the URDF link names regardless. See [[project_prompt_per_simulator]].
