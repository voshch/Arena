# task_generator tasks

Core abstractions for the episode loop: the `Task` driver, `TaskMode`
base, the `TaskContext` dependency bundle, and the three `TaskModeRegistry` instances
that wire everything together.

## Key types

### `Task`

[`task.py`](task.py)

The task modes bound to one env: one `TM_Robots`, one `TM_Obstacles`, and
zero-or-more `TM_Module` instances. `EpisodeNode` drives it. `Task.create(...)`
binds the modes the `task.*` params select (the `params` `create_task` passes in) and calls
`robots_manager.set_up()`. `__init__` only instantiates the modules. Entry
points:

| Method | Purpose |
| --- | --- |
| `Task.create(...)` | async factory; binds the initial modes and calls `robots_manager.set_up()` |
| `reset_robots(record)` | rebind the modes the params select, run `before_reset()` on the modules, reset `tm_robots`, run `on_reset_plan(plan)` on the modules, return the `ResetPlan` |
| `reset_obstacles(plan, record)` | reset `tm_obstacles` into the plan, seeded with `record.seed` |
| `tick(ended, assigned)` | forward to `tm_robots.tick(ended, assigned)`, `ended` maps a robot to the phase that ended on this tick and `assigned` holds the robots given a task from outside since the last tick |
| `after_reset()` | run `after_reset()` on the modules |
| `set_tm_robots(enum)` | swap the active `TM_Robots` mode |
| `set_tm_robots_from_config(path)` | bind the composite a task config file describes |
| `set_tm_robots_composite(specs)` | bind a multi-TM composite via `FleetManager` |
| `set_tm_obstacles(enum)` | swap the active `TM_Obstacles` mode |
| `done` | async property; `tm_robots.done` |
| `teardown()` | release both modes |
| `submit_task(request, robot_name)` | bypass `TM_Robots` and submit directly to one robot |

The episode end state (`abort_episode`, `force_reset`, timeout) and the
`reset_start` and `reset_end` topics are on `EpisodeNode`, see
[Reset semantics](#reset-semantics).

### `TaskMode`

[`mode.py`](mode.py)

Abstract base for all three axes. Provides `_ctx: TaskContext`,
`_namespace: Namespace`, and a `namespace(*path)` helper for constructing
parameter names scoped to the mode. Extends `NodeInterface` so every mode has
access to `self.node`, typed as the generic `ArenaMixinNode`: ROS params,
publishers, subscriptions, services, clocks. Everything env-specific comes from
`_ctx`.

### `TaskContext`

Defined in
[`env/task_layer.py`](../../../arena_runtime/arena_runtime/arena_runtime/env/task_layer.py)
and built by `EpisodeNode`.

`attrs.define` dataclass bundling the parts of the env a mode works on:

```python
@attrs.define
class TaskContext:
    environment_manager: EnvironmentManager
    robots_manager: RobotsManager
    world_manager: WorldManagerROS
    realizer: Realizer
    simulator: BaseSim
    acoustics_simulator: BaseAcousticsSimulator
    reset_lock: asyncio.Lock
    conf: TaskConfiguration
    episode: EpisodeContext
    interactive: InteractiveContext | None = None

    @property
    def robots(self) -> dict[str, RobotManager]: ...
```

`TaskConfiguration` is the part of the env `Configuration` modes read, under the
same names: `RNG`, `HUMAN`, `SAFE_DIST`, `GOAL_TOLERANCE_RADIUS`, `TIMEOUT` and
`KG_DERIVE`.

`EpisodeContext` is what a mode may tell the running episode: `compromise`,
`abort_episode`, `force_reset`, `register_timeline`, `register_conditions` and
`register_sound_levels`. `InteractiveContext` is the env's rviz side: `markers`, the
interactive marker hub, `robot_colors` and `tf_buffer`. It is `None` in a
host that has no rviz side, and the modes that use it assert it where they first need it.

`TM_Composite` replaces `robots_manager` with a scoped view so each sub-TM
only sees its allocated fleet slice.

### Mode registries

[`registry.py`](registry.py)

Three `TaskModeRegistry` instances map each enum value to a lazy loader plus
a per-key `TaskModeMeta`:

| Instance | Key type |
| --- | --- |
| `ROBOTS_MODES` | `Modes.TM_Robots` |
| `OBSTACLES_MODES` | `Modes.TM_Obstacles` |
| `MODULE_MODES` | `Modes.TM_Module` |

Loaders are zero-argument callables that import and return the concrete class
(lazy import pattern). All registrations fire at import time from each mode's
`__init__.py`.

`TaskModeMeta` is an `@attrs.frozen` dataclass with fields
`namespace: Namespace` and `schema: Callable[[ROSParamServer, Namespace], None] | None`.
Metadata is stored on the registry, keyed by enum value, and is reachable via
`<axis>_MODES.meta(key)` without invoking the loader.

`declare_mode_schemas(layer, node)` (in `arena_runtime.env.task_layer`) is called
once at node init. It walks `layer.mode_namespaces()` and calls the layer's
`declare_mode_schema(node, axis, mode)` for each registered mode, which fires the
schema stored in `reg.meta(key).schema`, forward-declaring all TM parameters
regardless of which modes will be activated. It never triggers a loader. The
robots and obstacles `scenario` modes share the `task.scenario` namespace.

## TM package structure

Each TM is a package with two files:

- `__init__.py` (eager): registers the mode on the appropriate
  `TaskModeRegistry` instance, declares `_NS`, and (if the mode has tunable
  parameters) defines the `_declare_schema(node, ns)` function. Imported at
  node startup.
- `impl.py` (lazy): contains the class body. Imported only when the mode is
  first activated.

The schema function calls one typed helper from
[`arena_rclpy_mixins.declarations`](../../../utils/arena_rclpy_mixins/arena_rclpy_mixins/declarations.py) per parameter:

```python
# __init__.py
from task_generator.modes import Modes
from arena_rclpy_mixins.declarations import declare_catalog, declare_int_pair
from task_generator.tasks.registry import ROBOTS_MODES, _REGISTRY_NAMESPACE

_NS = _REGISTRY_NAMESPACE("mymode")


def _declare_schema(node, ns):
    declare_int_pair(node, ns("static", "n"), [5, 15],
                     label="Static count", description="[min, max] count.")
    declare_catalog(node, ns("file"), "default", catalog="scenarios",
                    label="Scenario file", description="Scenario file name.")


@ROBOTS_MODES.register(Modes.TM_Robots.MYMODE, namespace=_NS, schema=_declare_schema)
def _load_mymode() -> "type[TM_Robots]":
    from .impl import TM_MyMode
    return TM_MyMode
```

Each helper builds the `ParameterDescriptor` (type, `additional_constraints`
mini-DSL, description) internally, schema authors don't touch
`ParameterDescriptor` directly.

## Parameter namespace

Parameters live under `task.<mode>.<leaf>`. Example: `task.random.static.n`.
The mode name is shared across families (e.g. `task.scenario.file` is read by
both `TM_Robots.scenario` and `TM_Obstacles.scenario`, which intentionally
load the same scenario file and project different sections out of it).

## Available declare helpers

| Helper | Underlying type | DSL token in `additional_constraints` | GUI widget |
| --- | --- | --- | --- |
| `declare_int_pair` | `INTEGER_ARRAY` | `range:int_pair` | min/max paired spinboxes |
| `declare_float_pair` | `DOUBLE_ARRAY` | `range:float_pair` | min/max paired double spinboxes |
| `declare_catalog` | `STRING` | `catalog:<name>` | combobox from `query/<name>` |
| `declare_catalog_array` | `STRING_ARRAY` | `catalog:<name>` | multiselect from `query/<name>` |
| `declare_enum` | `STRING` | `enum:a,b,c` | combobox of literal choices |
| `declare_string` | `STRING` | - | line edit (or text edit if "prompt"-flavoured) |
| `declare_int` | `INTEGER` | - (uses `integer_range` if `lo`/`hi` given) | spinbox |
| `declare_double` | `DOUBLE` | - (uses `floating_point_range`) | double spinbox |
| `declare_bool` | `BOOL` | - | checkbox |

Catalog names map 1:1 to query services on the env node:
`objects`, `pedestrians`, `scenarios`, `parametrizeds`, `environments`.

All helpers accept `label="Friendly Name"` and `description="..."`. The label
becomes the row title in the rviz panel; the description is the hover
tooltip.

## The three axes

| Axis | ABC | Enum | README |
| --- | --- | --- | --- |
| Robot goal dispatch | `TM_Robots` | `Modes.TM_Robots` | [robots/](robots/README.md) |
| Obstacle population | `TM_Obstacles` | `Modes.TM_Obstacles` | [obstacles/](obstacles/README.md) |
| Cross-cutting modules | `TM_Module` | `Modes.TM_Module` | [modules/](modules/README.md) |

## Reset semantics

`EpisodeNode._reset_episode` in arena_runtime drives the reset. It calls
`Task.reset_robots(record)`, which rebinds the modes the params select, runs
`module.before_reset()` for every active module, resets the robot mode, hands its
`ResetPlan` to `module.on_reset_plan(plan)` and returns it. The node applies it
(submits each entry's `task` from its `start_pose`, forbids its `forbidden_zones`), then calls
`Task.reset_obstacles(plan, record)`, which adds the obstacles. After the respawn and the
robot resets, `Task.after_reset()` runs `module.after_reset()` for every active
module. The simulator is paused throughout. The full ordering and the WORLD
layer invariant are in the
[env layer](../../../arena_runtime/arena_runtime/arena_runtime/env/README.md#reset-semantics).
