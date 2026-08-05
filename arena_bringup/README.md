# arena_bringup

Top-level launch entry point for Arena, plus shared Python launch helpers,
task configs. Everything that ties together the simulator,
task-generator, and navigation stack lives here.

## Guides

- [Usage (BRINGUP.md)](BRINGUP.md) - minimum-viable invocations, common arg
  permutations, expected behavior.
- [Launch surface](launch/README.md) - full `arena_runtime.launch.py` argument table,
  top-level composition, and the `utils/` helpers.
- [Simulator dispatch](launch/simulator/sim/README.md) - `SelectAction`
  convention, per-sim subdirs (`gazebo/`, `isaac/`), adding a new simulator.
- Human simulator dispatch: see `task_generator/launch/human/README.md`
  (moved alongside its `BaseHumanSimulator` adapters).
- [Task configs](configs/tasks/README.md) - `TaskModeSpec` schema,
  fleet-manager allocation, examples.
- [Benchmark configs](../arena_evaluation/arena_evaluation/configs/benchmark/README.md) - suites,
  contests, and how the runner consumes them.
- [Python helpers](README.md) - `LaunchArgument`, `SelectAction`,
  `IsolatedGroupAction`, YAML substitutions, and log-level extension.

## Internals

The `arena_bringup` Python package ([arena_bringup/README.md](README.md))
provides launch-time helpers used across all launch files in this package:

- **`LaunchArgument`** / **`SelectAction`** / YAML substitutions
  ([arena_bringup/substitutions.py](arena_bringup/substitutions.py))
- **`IsolatedGroupAction`**
  ([arena_bringup/actions.py](arena_bringup/actions.py))
- **`PythonExpression`** / **`IfElseSubstitution`**
  ([arena_bringup/future.py](arena_bringup/future.py))
- **`NodeLogLevelExtension`** / **`SetGlobalLogLevelAction`**
  ([arena_bringup/extensions/NodeLogLevelExtension.py](arena_bringup/extensions/NodeLogLevelExtension.py))
