---
name: urdf-loader-hardening
description: ModelProvider_URDF.load strips unregistered ros2_control blocks and locks zero-effort passive joints; isaac loader args now render the assembly wrapper
metadata: 
  node_type: memory
  type: project
  originSessionId: d1058176-4dfa-4e43-b9e3-c10599875612
---

Added 2026-07-17 in `arena_simulation_setup/utils/models/urdf.py` post-processing:
- `_strip_unregistered_ros2_control`: drops `<ros2_control>` blocks whose hardware plugin is not in the ament-index whitelist (`hardware_interface__pluginlib__plugin` resources), plus the classic `libgazebo_ros2_control.so` host element. The canonical `gz_ros2_control/GazeboSimSystem` sentinel is always exempt (gazebo consumes it in-sim, the isaac adapter rewrites it to JointStateTopicSystem). Un-vendored upstream descriptions (irobot_create_description, champ legs, clearpath arms) leak gazebo-classic blocks that crash the external CM otherwise.
- `_lock_passive_joints`: any prismatic/revolute joint with `<limit effort="0">` becomes fixed. Create3's unsprung wheel_drop suspension (3 cm travel) collapsed under gravity in both sims: belly-drag + transient tips in gz (tilted camera marked floor into global costmap), full beach + 2 mm/3 s stall in isaac.
- isaac `_robot_loader_args` now renders `xacro_wrapper` + `control_joint_patch` like gazebo does (before this, isaac got the bare chassis xacro: no assembly-mounted sensors at all, so no rplidar/oakd prims). See [[turtlebot-isaac-open-items]].

**Why:** the loader is the single choke point every spawned robot passes through, so sim-agnostic URDF repairs belong there, not in per-sim adapters.
**How to apply:** extend the loader post-chain for future URDF pathologies; unit tests in `arena_simulation_setup/tests/unit/test_urdf_ros2_control.py` take explicit sets so no mocking is needed.
