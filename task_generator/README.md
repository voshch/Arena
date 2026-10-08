# task_generator

Per-env episode loop for Arena. Assigns goals to robots, populates the
simulator with obstacles, and coordinates across episodes via a three-axis
task-mode registry.

The `arena_node` runtime, env/hold registries, and simulator adapters live
in [`arena_runtime/`](../arena_runtime/README.md).

## Guides

- [Task system](task_generator/tasks/README.md): `Task`, `TaskMode` ABC,
  `TaskContext`, the three mode registries, reset semantics.
- [Robot task modes](task_generator/tasks/robots/README.md): `TM_Robots`
  subclasses, fleet manager, adapters.
- [Robot adapters](task_generator/tasks/robots/adapters/README.md): `Adapter`
  ABC, shipped kinds, adding a new one.
- [Obstacle task modes](task_generator/tasks/obstacles/README.md): `TM_Obstacles`
  subclasses, shipped modes, zone-ref resolution, PROMPT registration.
- [Modules](task_generator/tasks/modules/README.md): `TM_Module` lifecycle
  hooks, shipped modules.
- [Managers](task_generator/manager/README.md): `RobotsManager`,
  `RobotManager`, `WorldManager`, `EnvironmentManager`, `Realizer`.
- [Sim interface](../arena_runtime/arena_runtime/arena_runtime/sim/README.md):
  `BaseSim` and its four sub-interfaces; registered implementations.
- [Human simulator](task_generator/simulators/human/README.md):
  `BaseHumanSimulator`, PROMPT registration, hunav default agent.
- [Acoustics simulator](launch/acoustics/README.md): `acoustics:=` axis,
  `BaseAcousticsSimulator` declares the map-server requirement, the RViz
  displays, panels and tools, the recorded topics and pedestrian hearing, and
  receives the sounds module's `SoundEmission`s. The `arena` backend
  (`simulators/acoustics/arena/`) is the only code importing
  [arena_auditory](../arena_auditory/README.md) (`arena feature auditory
  install`).
- [Robot hearing](launch/hearing/README.md): `robot.hearing:=` axis,
  `BaseHearing` declares the per-robot RViz displays and recorded topics,
  `nav2_overlay()` the Nav2 params overlay. The `arena` backend
  (`simulators/hearing/arena/`) is the only code importing
  [arena_hearing](../arena_hearing/README.md) (`arena feature hearing
  install`).
- [Utils](task_generator/utils/README.md): generic `Registry`, arena helpers,
  GPT shim, map generator.
- [Constants](task_generator/constants/README.md): `Configuration(server)`
  factory; all published ROS parameters.

## Internals

### The `Constants.TaskMode` registry

[`constants/__init__.py`](task_generator/constants/__init__.py) defines three
`enum.Enum` axes inside `Constants.TaskMode`:

| Axis | Enum | Default |
| --- | --- | --- |
| Robot goal dispatch | `TM_Robots` | `random` |
| Obstacle population | `TM_Obstacles` | `random` |
| Cross-cutting modules | `TM_Module` | *(empty set)* |

All three are wired at import time in
[`tasks/registry.py`](task_generator/tasks/registry.py) via three
`TaskModeRegistry` instances: `ROBOTS_MODES`, `OBSTACLES_MODES`, and
`MODULE_MODES`. Each registration stores a lazy loader (a zero-argument
callable that imports and returns the class) plus a `TaskModeMeta` (namespace
+ optional schema) keyed by the mode enum. Metadata is available without
invoking the loader, so the impl module is not imported until the mode is
first selected.

`Task.__init__` reads `task.robots`, `task.obstacles`, and `task.modules` from the
ROS parameter server (via `node.conf.TaskMode.*`) and calls the matching
loaders. On each reset `Task._reset_episode` re-reads the parameters, swapping
the active mode if it changed.

### Episode loop

See [task_generator/tasks/README.md](task_generator/tasks/README.md#reset-semantics)
for the full `Task._reset_episode` ordering and the WORLD-layer invariant.
