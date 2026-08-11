---
name: project-topic-based-hw-fork
description: We vendor voshch/topic_based_hardware_interfaces (main) because the apt-released 1.0.0 of joint_state_topic_hardware_interface skips publish for non-position command interfaces.
metadata: 
  node_type: memory
  type: project
  originSessionId: 562b37a8-626c-4593-805a-6651517eb139
---

`ros-jazzy-joint-state-topic-hardware-interface` 1.0.0 has a bug in `write()`: the diff loop guards `if (interface.name != HW_IF_POSITION) continue;` so any hardware block with only velocity (or effort) command interfaces never sees diff > 0, never trips the publish gate, and silently drops every command. Manifests as wheels not spinning on rbvogui/_plus despite the swerve controller computing correct `wheel_omega` (introspection topic confirms the value lands on the hardware-side command storage).

Fixed in upstream main by PR #118 (commit `babecd80`, "Publish JointState commands for velocity-only and effort-only interfaces"), but no release > 1.0.0 has been cut. We forked to `voshch/topic_based_hardware_interfaces` (only `main` branch) and added it to [_meta/repos/arena.repos](_meta/repos/arena.repos) so the workspace overlay shadows the apt build.

**Why:** required to make any velocity-command-only or effort-command-only `<ros2_control>` block actually drive joints through the Isaac topic_bridge : i.e. all wheels on every robot in the fleet. See also [[isaac-no-native-ros2-control]] for the architectural reason we're committed to this plugin at all.

**How to apply:** when upstream cuts a release > 1.0.0, drop the fork by removing the `arena.repos` entry and deleting `src/deps/topic_based_hardware_interfaces`. Until then, do not let the apt package shadow our overlay (workspace install order should already handle this). The Isaac side also sets `trigger_joint_command_threshold=0.0` in the URDF transform ([arena_runtime/.../isaac_simulator.py](arena_runtime/arena_runtime/arena_runtime/sim/isaac_simulator.py)) since a sim bridge has no reason to gate on diff at all.
