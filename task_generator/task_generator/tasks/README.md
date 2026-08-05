# task_generator tasks

Core abstractions for the episode loop: the `Task` driver, `TaskMode`
base, `TaskContext` dependency bundle, and the three `ClassRegistry` instances
that wire everything together.

## Key types

### `Task`

[`task.py:33`](task.py#L33)

The top-level driver. Combines one `TM_Robots`, one `TM_Obstacles`, and
zero-or-more `TM_Module` instances into a single episode loop. Entry points:

| Method | Purpose |
| --- | --- |
| `Task.create(...)` | async factory; calls `robots_manager.set_up()` before returning |
| `reset(**kwargs)` | run one full episode reset (see sequence below) |
| `set_tm_robots(enum)` | swap the active `TM_Robots` mode |
| `set_tm_robots_composite(specs)` | bind a multi-TM composite via `FleetManager` |
| `set_tm_obstacles(enum)` | swap the active `TM_Obstacles` mode |
| `is_done` | async property; `True` when `tm_robots.done` or `force_reset()` called |
| `force_reset()` | set the force-reset flag so the next `is_done` poll returns `True` |
| `submit_task(request, robot_name)` | bypass `TM_Robots` and submit directly to one robot |

Published ROS topics: `reset_start` and `reset_end` (`std_msgs/Empty`).
ROS parameter `resetting` (`bool`) is declared via `declare_parameters`.

### `TaskMode`

[`mode.py:14`](mode.py#L14)

Abstract base for all three axes. Provides `_ctx: TaskContext`,
`_namespace: Namespace`, and a `namespace(*path)` helper for constructing
parameter names scoped to the mode. Extends `NodeInterface` so every mode has
access to `self.node`.

### `TaskContext`

[`context.py:10`](context.py#L10)

`attrs.define` dataclass bundling the three managers:

```python
@attrs.define
class TaskContext:
    environment_manager: EnvironmentManager
    robots_manager: RobotsManager
    world_manager: WorldManager

    @property
    def robots(self) -> dict[str, RobotManager]: ...
```

`TM_Composite` replaces `robots_manager` with a scoped view so each sub-TM
only sees its allocated fleet slice.

### Mode registries

[`registry.py`](registry.py)

Three `TaskModeRegistry` instances map each enum value to a lazy loader plus
a per-key `TaskModeMeta`:

| Instance | Key type |
| --- | --- |
| `ROBOTS_MODES` | `Constants.TaskMode.TM_Robots` |
| `OBSTACLES_MODES` | `Constants.TaskMode.TM_Obstacles` |
| `MODULE_MODES` | `Constants.TaskMode.TM_Module` |

Loaders are zero-argument callables that import and return the concrete class
(lazy import pattern). All registrations fire at import time from each mode's
`__init__.py`.

`TaskModeMeta` is an `@attrs.frozen` dataclass with fields
`namespace: Namespace` and `schema: Callable[[ROSParamServer, Namespace], None] | None`.
Metadata is stored on the registry, keyed by enum value, and is reachable via
`<axis>_MODES.meta(key)` without invoking the loader.

`walk_schemas(node)` (module-level in `registry.py`) is called once at node
init to fire every registered schema, forward-declaring all TM parameters
regardless of which modes will be activated. It reads schemas via
`reg.meta(key).schema` and never triggers a loader.

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
from task_generator.constants import Constants
from arena_rclpy_mixins.declarations import declare_catalog, declare_int_pair
from task_generator.tasks.registry import ROBOTS_MODES, _REGISTRY_NAMESPACE

_NS = _REGISTRY_NAMESPACE("mymode")


def _declare_schema(node, ns):
    declare_int_pair(node, ns("static", "n"), [5, 15],
                     label="Static count", description="[min, max] count.")
    declare_catalog(node, ns("file"), "default", catalog="scenarios",
                    label="Scenario file", description="Scenario file name.")


@ROBOTS_MODES.register(Constants.TaskMode.TM_Robots.MYMODE, namespace=_NS, schema=_declare_schema)
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

Catalog names map 1:1 to query services on the task-generator node:
`objects`, `pedestrians`, `scenarios`, `parametrizeds`, `environments`.

All helpers accept `label="Friendly Name"` and `description="..."`. The label
becomes the row title in the rviz panel; the description is the hover
tooltip.

## The three axes

| Axis | ABC | Enum | README |
| --- | --- | --- | --- |
| Robot goal dispatch | `TM_Robots` | `Constants.TaskMode.TM_Robots` | [robots/](robots/README.md) |
| Obstacle population | `TM_Obstacles` | `Constants.TaskMode.TM_Obstacles` | [obstacles/](obstacles/README.md) |
| Cross-cutting modules | `TM_Module` | `Constants.TaskMode.TM_Module` | [modules/](modules/README.md) |

## Reset semantics

`Task._reset_episode` runs in this order:

1. `robots_manager.set_up()`: reconcile fleet (spawn/remove robots).
2. `environment_manager.before_reset_episode()`: pauses the simulator. The sim
   is paused for the entire body below; only node-discovery and lifecycle
   signals are observable here.
3. `module.before_reset()` for every active module.
4. `tm_robots.reset()`: compute new start/goal positions.
5. `tm_obstacles.reset()`: produce `(obstacles, dynamic_obstacles)` lists.
6. `environment_manager.respawn(callback)`: marks all current `INUSE`
   obstacles as `UNUSED`, runs the callback (which spawns the new lists), then
   removes everything still `UNUSED`.
7. `module.after_reset()` for every active module.
8. `environment_manager.after_reset_episode()`: unpauses the simulator.

**WORLD layer invariant:** entities spawned with `ObstacleLayer.WORLD` (walls,
doors, floors, world static entities) are never touched during `respawn`. They
survive all episode resets for the lifetime of the world.

## `extend()`: runtime obstacle/robot injection

Both `TM_Obstacles` and `TM_Robots` base classes expose an async `extend()`
method for spawning additional entities into a running episode:

- `TM_Obstacles.extend(kind, model, pose=None) -> str`: spawn one static or
  dynamic obstacle. When `pose` is `None`, `_placement.random_placement()` picks
  a free position. Returns the server-assigned entity name.
- `TM_Robots.extend(model, name=None, pose=None) -> str`: spawn an additional
  robot. Same placement semantics. Returns the assigned robot name.

Calling `extend()` via the `runtime/spawn_*` services flips
`EpisodeRecord.integrity = False` for the current episode.

## `EpisodeRecord`

Episode state is tracked in `EpisodeRecord` (an `attrs.define` dataclass on the
node). Key fields: `episode_id`, `world`, `seed` (derived via blake2b from
`run_seed|world|episode_id`), snapshot task-mode strings, `outcome_state`,
`outcome_info`, `goal_uuid`, and `integrity`. A new record is created at every
NEXT reset; integrity starts `True` and flips `False` on any manual mutation
(`extend()`, `set_robot_position`, `set_robot_goal`).
