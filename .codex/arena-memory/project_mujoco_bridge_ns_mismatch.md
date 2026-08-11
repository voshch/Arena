# MuJoCo ros2_control bridge ns mismatch (blocker)

The mujoco ros2_control bridge control loop never closes because adapter and server
derive the topic namespace from two structurally different sources.

- Adapter (arena_runtime/.../sim/mujoco_simulator.py robot_spawn) bakes the bridge URDF
  command/state topics from `ns = str(service_namespace(robot.name))`, which is the node's
  fully-qualified name: with the default register_env ns `arena/env_<id>/task_generator_node`
  (registry.py:80), the launch sets node namespace=dirname, name=basename, so FQ name is
  `/arena/env_<id>/task_generator_node`. Thus ns = `/arena/env_<id>/task_generator_node/<robot>`.
  Bridge state topic = `<ns>/mujoco/joint_states`.

- Server (arena_mujoco/.../control.py `_namespace`) derives ns from `tf_prefix.rstrip('/')`.
  tf_prefix = `robot.frame.tf()` = `<realizer_prefix>/<robot>/` where realizer prefix is the
  `prefix` ROS param = `env_<id>` (launch line 318). So server ns = `env_<id>/<robot>`, and it
  publishes/subscribes at relative `env_<id>/<robot>/mujoco/joint_states` under node `/mujoco`
  (global ns) -> `/env_<id>/<robot>/mujoco/joint_states`.

Mismatch: `/arena/env_0/task_generator_node/jackal/...` (adapter) vs `/env_0/jackal/...` (server).
The mujoco server IGNORES request.joint_states_topic/cmd_vel_topic/odom_topic entirely.

Contrast Isaac (works): Isaac server CONSUMES request.joint_states_topic verbatim into the
OmniGraph topicName, and Control() derives ns = `cmd_vel_topic.rsplit('/',1)[0]` from the
passed cmd_vel_topic, so both sides use the adapter's FQ-name-derived strings and agree.

Fix: mujoco server should bind topics from the adapter-supplied request fields
(joint_states_topic, cmd_vel_topic minus leaf for commands), not re-derive from tf_prefix.
Equivalent fix: adapter passes tf-prefix-derived topics, but that diverges from Isaac.
The odom subsystem (odom.py) has the same tf_prefix-derived ns and the same latent bug if
any consumer expects the FQ-name odom topic.
