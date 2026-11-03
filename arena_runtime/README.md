# arena_runtime

Workspace folder containing two co-located ROS packages:

- [arena_runtime/](arena_runtime/README.md): Python package; owns `arena_runtime_node` (the runtime binary), `EnvRegistry`, `HoldRegistry`, `CleanupManager`, the per-simulator adapters (gazebo, isaac, mujoco, flatsim, dummy), and the per-env layer every env node runs on: the [managers](arena_runtime/arena_runtime/env/README.md) with `EnvNode`, `Realizer` and the robot adapters, the [constants](arena_runtime/arena_runtime/env/constants/README.md), and the [human](arena_runtime/arena_runtime/humans/README.md), acoustics and hearing simulators.
- [arena_runtime_msgs/](arena_runtime_msgs/README.md): rosidl interfaces consumed by `arena_runtime_node` and per-env consumers.

`task_generator` (the task modes) depends on `arena_runtime` and arena_runtime loads it through the `arena_runtime.task_layer` entry point; the dep graph is one-way.
