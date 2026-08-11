---
name: project_turtlebot_wheel_tf_orphan
description: "turtlebot RViz \"No transform from left_wheel/wheel_drop_* to map\" is known and intentionally ignored, not a bug to fix"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7bc70661-03b3-4c88-9584-ebf92afee2f6
---

turtlebot (Create 3 base) shows RViz warnings "No transform from env_0/turtlebot/{left_wheel,right_wheel,wheel_drop_left,wheel_drop_right} to map". Root cause: Create 3 puts a passive prismatic joint in series between base_link and each wheel (base_link -> wheel_drop_left -> left_wheel). The gz_ros2_control joint_state_broadcaster only publishes the two actuated wheel joints, so RSP can't place wheel_drop_* (no state source), and the wheels dangle off those orphaned frames. joint_state_publisher that would fill passive joints at 0 is commented out in gazebo_simulator.py (~L694-705).

As of 2026-06-18 the user chose to IGNORE this: it is cosmetic (RViz RobotModel display only). Navigation chain map->odom->base_link->sensors is fully intact, robot drives fine. Do NOT re-flag as a bug.

If ever revisited, two clean fixes: (1) add wheel_drop_{left,right}_joint as state-only `<state_interface name="position"/>` in turtlebot.ros2_control.urdf (we own that file; gz reads sim position; no extra node) or (2) re-enable joint_state_publisher with source_list as a robot-agnostic catch-all. NOT in create3.urdf.xacro (upstream irobot_create_description, not vendored).
