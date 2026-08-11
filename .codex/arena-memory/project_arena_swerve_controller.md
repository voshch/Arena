---
name: arena_swerve_controller is first-party 4WIS controller
description: arena_swerve_controller lives at arena_robots/deps/arena_swerve_controller/ and is the working ros2_control plugin for rbvogui/rbvogui_plus's 4-wheel independent steering chassis
type: project
originSessionId: 562b37a8-626c-4593-805a-6651517eb139
---
`arena_swerve_controller/SwerveController` is a first-party ros2_control plugin we wrote (Apache-2.0) replacing the dead `robotnik_controllers/RBVoguiController` reference that never had a ROS 2 implementation.

Lives at `arena_robots/deps/arena_swerve_controller/`. Will later be lifted into `github.com/arena-robots/arena_swerve_controller` and submoduled back.

Used by rbvogui and rbvogui_plus (both via `control.yaml` -> `type: arena_swerve_controller/SwerveController` on the `robotnik_base_controller` instance).

Param schema lives in `src/arena_swerve_controller_parameters.yaml` (gen_param_lib): `wheel_joint_names`, `steering_joint_names`, `wheel_radius`, `wheel_positions_x/y`, `max_steering_angle`, `allow_reverse_drive`, `use_stamped_vel`, etc.

**Why**: No public Robotnik ROS 2 controllers exist for 4WIS (we searched the org); rbvogui was the last fleet member with no motion. Writing the controller in-tree as a standalone ament_cmake package was cheaper than waiting for upstream or building a workaround with ackermann_steering_controller (which loses crab/spin steering).

**How to apply**: When debugging or extending rbvogui/_plus motion, this is the controller in play. Known TODOs before "release-ready": cache the 8x3 LSQ A matrix at on_configure (currently rebuilt every tick, heap-allocates inside ColPivHouseholderQR), populate odom covariance from params, add controller_manager-based gtest, CHANGELOG/lint.
