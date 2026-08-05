# task_generator robots

Task-dispatch side of Arena's robot stack. This is the runtime counterpart
to [`arena_robots`](../../../../arena_robots/README.md): that package ships
per-robot config (URDFs, model_params, mappings) and the launch halves of
navigation adapters; this dir owns the Python that turns scenario-level
intent into goals sent at a specific robot through its bound adapter.

## Guides

- [Navigation adapters](adapters/README.md): `Adapter` ABC, `AdapterCtx`,
  registration, how `RobotManager` binds one, dispatch and teardown flow,
  adding a new adapter. Paired with the launch-side
  [`arena_robots/launch/adapters/README.md`](../../../../arena_robots/arena_robots/launch/adapters/README.md).
- [Fleet manager](#fleet-manager) (below): `TaskModeSpec` schema,
  allocation rules, the `null` sink and `composite` fan-out.

## Package structure

Each `TM_Robots` subclass is a package:

- `__init__.py` (eager): declares `_NS` and (optionally) `_declare_schema`, then registers the mode on `ROBOTS_MODES` (a `TaskModeRegistry` from `tasks/registry.py`) with `namespace=_NS` and `schema=_declare_schema`. Imported at node startup.
- `impl.py` (lazy): contains the class body. Imported only on first activation.

Parameters live under `task.<mode>.<leaf>`. Mode names are shared across the
three axes; e.g. `task.scenario.file` is read by both `TM_Robots.scenario` and
`TM_Obstacles.scenario`, which intentionally load the same scenario file.

## Level selection

`TM_Robots` is level-agnostic: it places robots on the whole compacted map (all loaded
levels), with start and goal sampled anywhere. Which levels exist is set at load time
(`world:=name` loads all, `world:=name[0,3]` loads only 0 and 3), not per reset.

Crossing floors is handled below the task mode, in `RobotManager.submit_task`: when a
`GoToPhase` targets a different level than the robot's current one, it injects
elevator-boarding subgoals (`WorldManager.elevator_route` BFS over the elevator graph)
ahead of it. The robot drives into each cabin, the mechanism shim teleports it across,
and the next leg becomes reachable. No explicit wait phase: the robot cannot path
to the disconnected goal until the teleport, then proceeds.

## Task modes (`TM_Robots` subclasses)

`TM_Robots` ([`__init__.py`](__init__.py)) is the base: one instance drives
all robots in its scope, exposing `reset`, `set_position`, `set_goal`, and
an async `done` flag. Shipped modes:

| Kind | File | Behavior |
| --- | --- | --- |
| `random` | [`random/`](random/) | one random reachable goal per robot per episode |
| `explore` | [`explore/`](explore/) | extends `random`; when a robot finishes or times out, a fresh random goal is assigned |
| `guided` | [`guided/`](guided/) | external controller drives the goal sequence |
| `scenario` | [`scenario/`](scenario/) | reads `start`/`goal` pairs from the world's scenario YAML |
| `null` | [`composite.py`](composite.py) | idle sink for robots unallocated by the fleet manager |
| `composite` | [`composite.py`](composite.py) | fan-out: each sub-TM sees a scoped `TaskContext` covering only its allocated robots |

Modes are registered by `Constants.TaskMode.TM_Robots` enum value in
[`tasks/registry.py`](../registry.py). `null` and `composite` are not in
the enum, they are only reachable via `set_tm_robots_composite`.

## Request types

`TM_Robots` subclasses build [`TaskRequest`](request.py) values and hand
them to `RobotManager.submit_task`:

- `TaskKind`: canonical vocabulary of phase kinds (currently just
  `GOTO_POSE`).
- `TaskPhase`: abstract; `is_satisfied(robot)` is the Tier-3 completion
  fallback when neither the request predicate nor the adapter gives a
  verdict.
- `GoToPhase(pose, tolerance_radius?, tolerance_angle?)`: navigate to a
  pose with optional per-phase tolerance overrides.
- `TaskRequest(phases, done_predicate?)`: ordered phase list plus an
  optional Tier-1 completion predicate.

## Fleet manager

[`fleet_manager.py`](fleet_manager.py) resolves episode-level
`task_modes:` entries (from scenario config) to the live
`RobotManager` instances.

### `TaskModeSpec`

```yaml
task_modes:
  - kind: random              # TM_Robots kind (or "null")
    produces: goto_pose       # default; matched against robot.accepts
    assignments: [jackal_0]   # pin specific robots by name; [] = pool
    config: { ... }           # pass-through to the TM loader
```

### Allocation (`FleetManager.match`)

1. **Pinned first.** Every `name` in each spec's `assignments` must exist in
   the robot set and must not appear on multiple specs. The spec's
   `produces` kind must be in that robot's `accepts`; otherwise error.
2. **Pool next.** Each unpinned robot joins the first unpinned spec whose
   `produces` is in the robot's `accepts`. Greedy / first-fit, specs
   earlier in the list get priority.
3. **Null sink.** If a spec with `kind == "null"` exists, every still-
   unallocated robot joins it. Without a null spec, unallocated robots
   receive no TM (they sit idle).

Result: `dict[TaskModeSpec, list[RobotManager]]`, one entry per spec, in
the input order.

### Composite wiring

[`Task.set_tm_robots_composite`](../task.py) takes the `FleetManager`
allocation and:

1. For each `(spec, robots)` pair, resolves the loader:
   `Constants.TaskMode.TM_Robots(spec.kind)` via the standard registry,
   falling back to `get_extra_tm_loader(spec.kind)` for sentinel kinds
   (`null`).
2. Builds a scoped `TaskContext` via `_scoped_ctx` that exposes only the
   allocated robots through `ctx.robots`.
3. Instantiates one sub-TM per spec and wraps them all in
   [`TM_Composite`](composite.py), whose `done` is `all(sub.done)` and
   whose `set_position` / `set_goal` fan out.

The parent `TaskMode` enum slot is set to `None` after composite bind, so
the "new_tm != current" check in `_reset_episode` does not retrigger a rebind.

## Integration points

- **`RobotManager`** ([`manager/robot_manager/robot_manager.py`](../../manager/robot_manager/robot_manager.py)):
  one per spawned robot. Owns the bound adapter, the current `TaskRequest`,
  the phase index, the goal-republish loop, and the tf-backed `pose`
  property. Entry points used from here: `submit_task`, `move`, `is_done`,
  `accepts`.
- **`RobotsManager`** ([`manager/robot_manager/robots_manager.py`](../../manager/robot_manager/robots_manager.py)):
  diff-driven fleet lifecycle. Parses the `robot` ROS param (comma-
  separated list of robot names, `model[count]`, or `.yaml` setup refs),
  reconciles against existing `RobotManager` instances, and creates /
  destroys / updates to match. Reset-time entry is `set_up`.
- **`TaskContext.robots`**: the mapping `TM_Robots` subclasses iterate;
  `TM_Composite` replaces it with a scoped view so each sub-TM only sees
  its fleet slice.
