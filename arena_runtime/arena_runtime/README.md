# arena_runtime

Arena-Rosnav runtime: the `arena_runtime_node` lifecycle node (env registry, holds, cleanup) plus the per-simulator adapters (gazebo, isaac, MuJoCo, flatsim, dummy). It also holds the per-env layer (`EnvNode`, `EpisodeNode`, the robot, world and environment managers, the `entities` and `episode` MCP domains), the human, acoustics and hearing simulators, and the launch files for an env, its human simulator, acoustics and hearing.

`task_generator` (the task modes) consumes this package; the dep graph is one-way. It registers the one `arena_runtime.task_layer` entry point that the env node loads.

See [arena_runtime/README.md](arena_runtime/README.md) for the runtime primitives and service / topic surface, and [arena_runtime/sim/README.md](arena_runtime/sim/README.md) for the `BaseSim` interface.
