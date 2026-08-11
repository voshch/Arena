---
name: kinematics_substitutions lives in caps.py, not nav2.py
description: arena_robots.caps owns kinematics_substitutions(mobile) because arena_isaac transitively imports it through RobotView.control, and nav2.py imports ROS 2 launch which is broken in Isaac's vendored Python (missing lark).
type: project
originSessionId: baf043f8-7a6e-4226-959e-eb2fcac51e83
---
`kinematics_substitutions(mobile: MobileSpec)` is defined in `arena_robots/arena_robots/caps.py`, not `nav2.py`. `Robot.RobotView.control` calls it via `_resolve_kinematics_substitutions` to expand `${...}` placeholders in `control.yaml`. arena_isaac calls `RobotView.control` from `SpawnUrdf.spawn_urdf`, so anything that path transitively imports must avoid pulling in ROS 2 `launch` — Isaac ships a broken vendored copy under `/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy/launch/` that imports `lark` which isn't installed.

**Why:** Originally lived in `nav2.py`. Isaac side crashed with `ModuleNotFoundError: No module named 'lark'` because `nav2.py` does `import launch` at module top, for the `YAMLFileSubstitution` subclasses (KinematicsDerivedYAML, etc.). Moving the pure function out fixed it; `nav2.py` re-imports from `caps`.

**How to apply:** Anything called transitively from `RobotView.control` / `RobotView.model_params` must stay launch-free. If you add helpers that read caps/mobile.yaml or control.yaml, put them in `caps.py` or another launch-free module, not `nav2.py`.
