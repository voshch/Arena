---
name: project_isaac_jackal_nav2_bridge_crash
description: "Open issue, Isaac 6.0 segfaults in ros2.nodes plugin ~2-3min into jackal+nav2 runs, peds not involved"
metadata: 
  node_type: memory
  type: project
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

Open as of 2026-07-02: `arena launch sim:=isaac robot:=jackal tm_obstacles:=random headless:=true` segfaulted twice reproducibly ~2-3 min in, identical signature: `libisaacsim.ros2.nodes.plugin.so` `std::vector<unsigned char>::_M_default_append` via `libomni.graph.action_core` (ROS bridge OG node during graph execution, likely a message buffer resize). Zero pedestrians were spawned in either crash run, and the same session's turtlebot config (`robot:=turtlebot tm_robots:=random tm_obstacles:=random` with 3 peds) ran 5.3+ min clean, so the new peds pipeline is not implicated. Suspects unexplored: jackal's sensor set (16-beam lidar 640 samples vs turtlebot), nav2 explore cmd_vel subscriber path, message size growth in a publisher. Not the perSensorTickTlas crash (different stack, that one is libomni.kit.loop-isaac, see [[project_isaac_sensor_publish_fix]]). Needs a dedicated repro/bisect session.
