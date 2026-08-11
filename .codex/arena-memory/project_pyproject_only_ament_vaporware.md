---
name: project_pyproject_only_ament_vaporware
description: pyproject-only ament_python build is vaporware; colcon requires setup.py; use the reduced-shim for the combined-pyproject migration
metadata: 
  node_type: memory
  type: project
  originSessionId: 2c6c4c7b-c6d7-45f2-97dc-4c64f5a1c12f
---

For the combined-uv-workspace / per-package-deps migration ([[project_combined_pyproject_workspace_plan]]): building an ament_python package from pyproject.toml ONLY (no setup.py) is NOT possible with released colcon, and "adopting the prototype" is a dead end.

Evidence (investigated 2026-06-17, colcon_core 0.21.0 / colcon_ros 0.5.0, both pip-managed in ARENA_VENV):
- `colcon_ros/package_identification/ros.py` + `colcon_python_setup_py` route every `ament_python` package through `setup.py` extraction; `if not setup_py.is_file(): return` — no setup.py means the package isn't even identified.
- `colcon_core/python_project/spec.py` exists (reads pyproject, default backend `setuptools.build_meta:__legacy__`) but is DORMANT: `load_spec` is imported nowhere; `__init__.py` is the empty-file SHA.
- `colcon-python-project` (the opt-in extension) on `main` is an empty husk: only a zero-byte `__init__.py` + lint tests, and its `setup.cfg` registers ZERO entry points, so installing it wires nothing into colcon.
- Official: ROS 2 docs say ament_python = setup.py/setup.cfg; the pyproject path is a "PROTOTYPE FOR TESTING ONLY" (colcon-python-project repo), stalled since the 2023 Call-for-Testing (2024 Discourse: "not much has happened").

So pyproject-only would require us to implement + maintain the colcon build integration ourselves and fork colcon for all Arena builds. Not worth it.

Decision: use the REDUCED-SHIM. setup.py shrinks to the dynamic ament glue only (resource-index marker, package.xml install, globbed launch/worlds/configs/meshes data_files); pyproject `[project]` is authoritative for name/version/dependencies/scripts. No field declared in both -> no setuptools PEP 621 conflict. Members stay uv-virtual via `[tool.uv] package = false` so uv aggregates deps but colcon owns the build.

Cannot verify builds in Claude's sandbox: no ROS (`/opt/ros` empty, no ros2/ament_package), and the ARENA_VENV is broken (host python upgraded to 3.14, venv is 3.12). Build spikes must run inside the arena container.
