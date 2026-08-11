---
name: project_features_symlinked_into_docker_tree
description: "arena features authored in _meta/features/ must be symlinked into _meta/docker/features/ so the in-container `arena feature` CLI resolves them"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9a5a046c-21b4-4a27-b3be-1e4323777aca
---

There are two feature trees and the in-container `arena` CLI reads the docker one: `_meta/docker/features/docker/source.container` overrides `ARENA_FEATURES_DIR=$ARENA_DIR/_meta/docker/features` inside the container, while host `_meta/tools/source` uses `_meta/features`. So a feature added only under `_meta/features/<name>` is invisible to `arena feature <name>` inside the dev container (where Arena actually runs).

**Rule: symlink the feature into the docker tree.** The default way to bridge an authored `_meta/features/<name>` into `_meta/docker/features/` is a relative symlink, exactly like `_meta/docker/features/{evaluation,planners,robots} -> ../../features/<name>`. Only deviate when the feature needs container-specific content: gazebo is a divergent plain-file copy (its docker variant drops the host OpenUSD build), and isaac/vllm/training are real dirs holding their own `main` + `docker-compose.yml` (+ Dockerfile) because they run as separate containers (`_meta/docker/lib` `arena_docker_compose` merges those per-feature compose overrides from `.installed`).

**How to apply:** plain symlink for container-generic features (pip install, submodule, `ros2 launch`), NOT a gazebo-style copy or an isaac-style dir: `ln -s ../../features/<name> _meta/docker/features/<name>`. MuJoCo's symlink was created 2026-06-10 (uncommitted, shows as `?? _meta/docker/features/mujoco` in git status); the remaining MuJoCo handoff items are the voshch/arena_mujoco remote+.gitmodules wiring and the OBJ obstacle DB rebuild.
