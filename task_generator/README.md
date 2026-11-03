# task_generator

The default task runtime of Arena: the three-axis task-mode registry and its
modes, which assign goals to robots and populate the simulator. Other runtimes
register beside it, see
[task runtimes](../arena_runtime/arena_runtime/arena_runtime/env/README.md#task-runtimes).

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

| Axis | Enum |
| --- | --- |
| Robot goal dispatch | `TM_Robots` |
| Obstacle population | `TM_Obstacles` |
| Cross-cutting modules | `TM_Module` |

Each enum's `default()` names its default.

All three are wired at import time in
[`tasks/registry.py`](task_generator/tasks/registry.py) via three
`TaskModeRegistry` instances: `ROBOTS_MODES`, `OBSTACLES_MODES`, and
`MODULE_MODES`. Each registration stores a lazy loader (a zero-argument
callable that imports and returns the class) plus a `TaskModeMeta` (namespace
+ optional schema) keyed by the mode enum. Metadata is available without
invoking the loader, so the impl module is not imported until the mode is
first selected.

The `task.robots`, `task.obstacles`, `task.config` and `task.modules` params are
`TaskModeParams` in [`tasks/layer.py`](task_generator/tasks/layer.py), which the node
holds as `task_params` and hands to `create_task`. Its `modes` property is the
live selection per axis (`robots`, `obstacles`, `modules`). `validate_modes(modes)` is
the one validator of the task layer contract: for each listed axis it returns the
names in their canonical lowercase form, or raises a `ValueError` naming the axis and
the bad value and listing the allowed names, for an unknown axis, an unknown mode
name, or a `robots` or `obstacles` entry without exactly one name. `modules` takes any
number of names. The `task.robots`, `task.obstacles` and `task.modules` params parse
through it, so a bad launch value fails with the same message as a bad queued
episode. `resolve(modes)` is the full selection `select(modes)` would make live from
validated input: `robots` and `obstacles` take the given name, `modules` takes the
given names plus the always-on ones, plus `sounds` when the node's `acoustics` param
is not `none` or `auditory.static_sounds` is set, and a missing axis keeps the live
selection. `modes` is `resolve({})`, and `select(modes)` writes the resolved axes whose
names differ from the live ones. `create_task` calls `Task.create`,
which binds the `task.robots` and `task.obstacles` modes and then
`robots_manager.set_up()`. `Task.__init__` only instantiates the modules. On each
reset `EpisodeNode._reset_episode` calls `Task.reset_robots(record)`, which
re-reads `modes` and swaps the active mode if it changed. Every sim tick the
env calls `Task.tick(ended)`, which forwards to the robot mode. The layer's
`next_episode` returns the queued overrides unchanged.

See [Reset semantics](../arena_runtime/arena_runtime/arena_runtime/env/README.md#reset-semantics)
for the full reset ordering and the WORLD-layer invariant.
