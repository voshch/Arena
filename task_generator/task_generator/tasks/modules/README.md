# task_generator modules

`TM_Module` and its shipped subclasses. Modules run cross-cutting logic before
and after every episode reset without owning the robot or obstacle axes.

## `TM_Module` ABC

[`__init__.py:8`](__init__.py#L8)

```python
class TM_Module(TaskMode):
    _task: "Task"

    def before_reset(self): ...
    def after_reset(self): ...
```

Both hooks default to no-ops. Subclasses override only the ones they need.
`_task` is the live `Task` instance, giving modules access to
`world_manager`, `robots_manager`, and `force_reset()`.

Modules are instantiated in `Task.__init__` from the `tm_modules` ROS param
(a comma-separated list of `Constants.TaskMode.TM_Module` values). Each is
registered on `MODULE_MODES` in
[`tasks/registry.py`](../registry.py).

## Package structure

Each `TM_Module` subclass is a package:

- `__init__.py` (eager): declares `_NS` and (optionally) `_declare_schema`, then registers the mode on `MODULE_MODES` (a `TaskModeRegistry` from `tasks/registry.py`) with `namespace=_NS` and `schema=_declare_schema`. Imported at node startup.
- `impl.py` (lazy): contains the class body. Imported only on first activation.

Parameters live under `task.<mode>.<leaf>`.

## Shipped modules

| Enum value | Class | File | `before_reset` | `after_reset` |
| --- | --- | --- | --- | --- |
| `clear_forbidden_zones` | `Mod_ClearForbiddenZones` | [`clear_forbidden_zones/`](clear_forbidden_zones/) | calls `world_manager.forbid_clear()` | - |
| `rviz_ui` | `Mod_OverrideRobot` | [`rviz_ui/`](rviz_ui/) | - | - |
| `staged` | `Mod_Staged` | [`staged/`](staged/) | loads new stage config when stage index changes; publishes `goal_radius` and obstacle counts | - |

### `Mod_ClearForbiddenZones`

[`clear_forbidden_zones/impl.py:4`](clear_forbidden_zones/impl.py#L4)

Clears all dynamically forbidden map cells before each reset so obstacles
from the previous episode do not pollute free-cell sampling.

**Limitation (multi-level worlds):** `forbid_clear()` with no argument only clears the
global (single-level) map. Per-level forbidden zones written via `forbid(..., level_id=X)`
accumulate across resets and are not cleared by this module. This is a known gap for
multi-level worlds; a follow-up should extend `forbid_clear()` to accept `level_id="*"`
semantics that clears all per-level forbidden zones.

### `Mod_OverrideRobot`

[`rviz_ui/impl.py:8`](rviz_ui/impl.py#L8)

Subscribes to `<task_generator_node>/initialpose` (`PoseWithCovarianceStamped`),
`<task_generator_node>/goal_pose` (`PoseStamped`), and
`<task_generator_node>/clicked_point` (`PointStamped`), all namespaced under
the task_generator node so multiple instances do not cross-talk. Forwards
set-position and set-goal calls to `Task.set_robot_position` /
`set_robot_goal`; a clicked point calls `task.force_reset()`. Provides
interactive RViz-based control without modifying the active task mode.

### `Mod_Staged`

[`staged/impl.py:47`](staged/impl.py#L47)

Reads a curriculum YAML (list of stages, each with `static`, `dynamic`, `goal_radius`, and optional `dynamic_map` fields). Stage index is
advanced via `next_stage` / `previous_stage` ROS topics. `before_reset`
publishes the stage's `goal_radius` and obstacle counts as ROS params when
the stage index changes.

## Adding a module

1. Create `tasks/modules/<name>/` as a package.
2. In `__init__.py`: declare `_NS = _REGISTRY_NAMESPACE("<name>")`, define `_declare_schema(node, ns)` if the module has tunable parameters (using helpers from [`arena_rclpy_mixins.declarations`](../../../../utils/arena_rclpy_mixins/arena_rclpy_mixins/declarations.py)), then register the loader with `@MODULE_MODES.register(Constants.TaskMode.TM_Module.<NAME>, namespace=_NS, schema=_declare_schema)`.
3. In `impl.py`: define the class extending `TM_Module`; override `before_reset` and/or `after_reset`.
4. Add `<NAME> = "<name>"` to `Constants.TaskMode.TM_Module` in [`constants/__init__.py`](../../constants/__init__.py).
5. `walk_schemas` (module-level in `tasks/registry.py`) picks up your schema automatically at node init.
