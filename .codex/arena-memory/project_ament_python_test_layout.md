---
name: ament_python test layout
description: standard test discovery pattern for ament_python packages in Arena
type: project
originSessionId: 3e4de120-3987-4ba6-861b-9e19d8bd06b3
---
Working pattern (matches task_generator and other Arena packages):

- Directory: `<pkg>/tests/` (plural, not `test/`).
- `<pkg>/pyproject.toml` with `[tool.pytest.ini_options]` → `testpaths = ["tests"]`.
- `<pkg>/tests/conftest.py` with `pytest_collection_modifyitems` that skips items under `tests/ros/` or `tests/integration/` if `rclpy` is not importable, plus a session-scope `rclpy_context` autouse fixture that init/shutdown rclpy when available.
- `setup.py`: use `extras_require={'test': ['pytest>=7', ...]}` (not the deprecated `tests_require=['pytest']`). Use `find_packages(where='.', include=[f'{package_name}*'])` over explicit `packages=[...]`.

Why: `arena_evaluation` originally had `test/` (singular) and no pyproject pytest config; `arena test arena_evaluation` reported "0 tests ran". Switching to `tests/` plural + pyproject `testpaths` matches the task_generator setup that already works in CI.

How to apply: when creating a new ament_python package or fixing test discovery, mirror task_generator's `pyproject.toml`, `tests/conftest.py`, and `setup.py` test-related fields.
