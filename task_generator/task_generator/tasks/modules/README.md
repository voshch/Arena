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
| `sounds` | `Mod_Sounds` | [`sounds/`](sounds/) | - | renders world-, scenario- and launch-declared `sound` semantic entities |
| `rviz_ui` | `Mod_OverrideRobot` | [`rviz_ui/`](rviz_ui/) | - | puts the robot and goal handles |
| `zone_edit` | `Mod_ZoneEdit` | [`zone_edit/`](zone_edit/) | - | puts one drag handle per zone corner |
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

### `Mod_Sounds`

[`sounds/impl.py:152`](sounds/impl.py#L152)

A pure renderer, not an owner of state. After each reset it resolves every
`sound` entity from the loaded world, the active scenario's episode-scoped
`sounds:` and the world-independent `static_sounds` launch configuration
through the task generator realizer against the sound catalog
(`arena_simulation_setup.tree.assets.sound_catalog`), and every 0.1 s hands
one `SoundEmission` per resolved entity, built from the live
`sounding`/`volume_db` semantics the engine already tracks, to the acoustics
simulator's `emit_sounds`. The wire format and retired-source repeats belong
to the acoustics backend, a removed sound is simply no longer listed. With
`acoustics:=none` the noop backend drops them. It serves
`runtime/spawn_sound` and `runtime/remove_sound` (`task_generator_msgs/srv`)
for RViz-driven runtime sources. Toggling a declared sound is a `SetSemantic`
write, not a module service. A runtime sound not attached to a TF frame gets a
drag handle with a Remove menu. Dragging moves the source in place while the
handle moves, so it keeps its entity name and keeps playing.

### `Mod_OverrideRobot`

[`rviz_ui/impl.py:8`](rviz_ui/impl.py#L8)

Subscribes to `<task_generator_node>/initialpose` (`PoseWithCovarianceStamped`),
`<task_generator_node>/goal_pose` (`PoseStamped`), and
`<task_generator_node>/clicked_point` (`PointStamped`), all namespaced under
the task_generator node so multiple instances do not cross-talk. Forwards
set-position and set-goal calls to `Task.set_robot_position` /
`set_robot_goal`; a clicked point calls `task.force_reset()`. Provides
interactive RViz-based control without modifying the active task mode.

It also puts drag handles on the node's marker server (the Handles display).
Every robot gets a `robot/<robot>` handle in its base TF frame, so rviz
carries it along with the robot. Dropping it teleports the robot there and the
handle snaps back onto it. Robots whose task mode has no goal editor of its own
(see `TM_Robots.goal_editing_robots`) also get a `goal/<robot>` handle at the
first go-to of their current task. On release it sends that robot alone a
single go-to through `Task.submit_task`. The goal handle moves through
`RobotManager.watch`, which fires after every teleport and every dispatched
task. Handles are rebuilt after every reset and on every fleet change. Each
robot keeps one palette color ([`interactive/colors.py`](../../interactive/colors.py))
for its robot and goal handles, its guided waypoints and its Plan and Trail
displays.

### `Mod_ZoneEdit`

[`zone_edit/impl.py`](zone_edit/impl.py)

Off unless listed in `task.modules`. After every reset it puts one handle per
zone corner of every loaded level. Dragging a corner rewrites it in the world
description the world manager holds, so placement by zone name follows at
once, and republishes the world overlay (the World display, off by default).
Floors are not respawned, they catch up when the world reloads. Each corner's
menu has "Save world to <dir>", which rewrites only the changed corner
coordinates in each loaded level's `world.yaml` and leaves every other byte
untouched ([`utils/zone_corners.py`](../../utils/zone_corners.py)). The next
reset that applies the world reloads it, because the world manager compares
`world.yaml` mtimes. A world resolved from a download cache is saved into
that cache.

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
