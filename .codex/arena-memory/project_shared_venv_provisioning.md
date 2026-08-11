---
name: project_shared_venv_provisioning
description: "Arena/.venv is container-provisioned and fragile, canonical repair recipe, never uv sync from host"
metadata: 
  node_type: memory
  type: project
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

Arena/.venv is bind-mounted as /opt/venv into BOTH the arena and isaac containers and is the runtime for every python ROS node (console script shebangs) and kit's PYTHONPATH packages. Facts learned the hard way 2026-07-02 (an agent's host-side `uv sync` broke the whole stack overnight):

- Canonical provisioning (_meta/tools/source:169-170), must run IN-CONTAINER: `uv venv .venv --allow-existing --system-site-packages && UV_PROJECT_ENVIRONMENT=... uv sync --inexact --project .` from /opt/arena_ws/src/Arena. Host-side uv venv points the interpreter at host uv-managed python (dangling in-container) and omits system-site-packages (kills rclpy).
- NEVER plain `uv sync` (exact): removes lock-external packages. Feature layers are lock-external: training = `uv pip install -e arena_training -e rosnav_rl` (_meta/features/training/main:11) pulling torch etc. Also lock-external: pydantic, idna, transforms3d>=0.4.2 (system 0.4.1 is numpy<2-only and breaks tf_transformations under the venv's numpy 2.4.6), pytest-repeat, usd-core extras.
- system-site-packages trap: uv treats old SYSTEM dist-packages as satisfying lock deps and skips venv installs -> venv cattrs importing system attrs (no NothingType) crashed arena_node/world_generator. Cure: `uv sync --inexact --reinstall`.
- Corruption pattern: dist-info present but package files gone -> uv says "satisfied", import fails. Cure per package: `uv pip install --reinstall <pkg>`. Root-owned __pycache__ inside the venv can block reinstalls (clear with docker exec -u root).
- Verify imports with full overlay: `docker exec ... bash -lc 'source /opt/ros/jazzy/setup.bash; source /opt/arena_ws/install/setup.bash; /opt/venv/bin/python3 -c "import rosnav_rl, cattrs, arena_simulation_setup.utils.cattrs"'`.
