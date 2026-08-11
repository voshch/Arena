---
name: isaac-no-native-ros2-control
description: "Isaac Sim 5.x ships no in-process controller_manager; the topic_bridge OmniGraph + external CM is the only viable architecture for ros2_control plugins (including arena_swerve_controller, JTC/MoveIt) on Isaac."
metadata: 
  node_type: memory
  type: project
  originSessionId: 562b37a8-626c-4593-805a-6651517eb139
---

Isaac Sim 5.x has **no** native ros2_control hardware plugin or in-process controller_manager. `isaacsim.ros2.sim_control` is for sim orchestration (play/pause/spawn/set_entity_state), not joint control. NVIDIA's own MoveIt example uses the same external-CM-via-topic-bridge pattern we have.

Architecture is fixed: `topic_bridge` OmniGraph (publishes `joint_states`, subscribes per-kind `joint_commands_{velocity,position}`) ↔ external `controller_manager` process loaded with URDF whose hardware plugin is `joint_state_topic_hardware_interface/JointStateTopicSystem`. Sibling community projects (e.g. `hijimasa/isaac-ros2-control-sample`) use the same upstream plugin.

**Why:** ruled out a "pivot to Isaac native ros2_control" : the thing simply doesn't exist. Decision is forced by requirements: we need `arena_swerve_controller` (4WIS) on rbvogui/_plus and `JointTrajectoryController` + MoveIt on arms; dropping ros2_control would mean writing bespoke OmniGraph kinematics for every drive type and losing MoveIt arm execution. Bridge is the only path that preserves those.

**How to apply:** when something breaks in the Isaac control plane, debug the bridge components (Isaac OmniGraph, the topic-based hardware plugin, CM/controller config) : do not look for a native shortcut. The Isaac docs page on "ros2 simulation control" is orchestration, not motor control. See also [[project-topic-based-hw-fork]] for the upstream bug that bit us inside this architecture.
