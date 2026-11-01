# task_generator

The task layer of Arena: the three-axis task-mode registry and its modes, which
assign goals to robots and populate the simulator.

The package registers [`tasks/layer.py`](task_generator/tasks/layer.py) under the
`arena_runtime.task_layer` entry point group (see [setup.py](setup.py)).
[`arena_runtime`](../arena_runtime/README.md) loads it through
[`env/task_layer.py`](../arena_runtime/arena_runtime/arena_runtime/env/task_layer.py)
and runs it in the env node, which owns the simulator, managers, episode loop,
launch files and the `episode` MCP domain, see the
[env layer](../arena_runtime/arena_runtime/arena_runtime/env/README.md).

## Guides

- [Task system](task_generator/tasks/README.md): `Task`, `TaskMode` ABC,
  `TaskContext`, the three mode registries, reset semantics.
- [Robot task modes](task_generator/tasks/robots/README.md): `TM_Robots`
  subclasses, fleet manager, adapters.
- [Obstacle task modes](task_generator/tasks/obstacles/README.md): `TM_Obstacles`
  subclasses, shipped modes, zone-ref resolution, PROMPT registration.
- [Modules](task_generator/tasks/modules/README.md): `TM_Module` lifecycle
  hooks, shipped modules.

## Internals

### The `Modes` registry

[`modes.py`](task_generator/modes.py) defines three `enum.Enum` axes inside
`Modes`, importable without loading a mode:

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

The `task.robots`, `task.obstacles`, `task.config` and `task.modules` params are
`ModeParams` in [`tasks/layer.py`](task_generator/tasks/layer.py), reached as
`node.task_params`. `create_task` there reads `task.modules` (plus the
always-on modules) and calls `Task.create`, which binds the `task.robots` and
`task.obstacles` modes and then `robots_manager.set_up()`. `Task.__init__` only
instantiates the modules. On each reset `EpisodeNode._reset_episode` calls
`Task.prepare_reset`, which re-reads the params and swaps the active mode if it
changed.

See [Reset semantics](../arena_runtime/arena_runtime/arena_runtime/env/README.md#reset-semantics)
for the full reset ordering and the WORLD-layer invariant.
