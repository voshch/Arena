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

Crossing floors is handled below the task mode, in `RobotManager.submit_task`:
[`route_phases`](../../manager/robot_manager/elevator_routing.py) expands the explicit
`via` entries of a `GoToPhase` (elevator or stair half names), then, when the goal is on a
different level than the one the robot ends up on, the fewest-hop route over the elevators
and the stairs the robot climbs (`WorldManager.level_hops`). A tie in hop count goes to the
route with fewer elevator rides. Each ride becomes a door-aligned approach leg
(`ElevatorLegPhase`), an `ElevatorRidePhase` and a door-aligned exit leg. The ride is a
goto phase whose `target` is the boarding cabin, whose dispatch pose sits at the cabin's
back wall and whose `until` atom is `robot in <destination cabin>`, so it is met once the
robot has boarded and the elevator has delivered it, and replay judges it like any other
phase. The starting level comes from the `start` pose a task mode passes along with the
request (a reset submits before the robot is moved there), or from the robot's live pose
when none is given. The mechanism shim teleports the robot across while it waits inside
the cabin.

A stair is routable only through a boarding half (one with `to`: every stair half, the
boarding half of an escalator) and only for a robot whose mobile cap lists its kind in
`climbs` (see the [robots README](../../../../arena_robots/arena_robots/robots/README.md)).
A robot without it never routes through stairs, and a `via` naming a stair it does not
climb, or an exit-only escalator half, is an error. A stopped escalator stays routable, the
robot walks it. Each crossing becomes three phases, all in the abstract frame:

1. a `StairsLegPhase` on the axis outside the boarding mouth, facing the seam, 1 m or the
   robot radius plus 0.1 m out, met within 0.25 m once facing the seam within 0.3 rad
2. a `StairsRidePhase` with its dispatch pose past the trigger line (`STAIRS_SEAM_INSET`
   before the seam) as deep as the robot's disc fits 0.05 m short of the seam wall, at least
   0.05 m past the line, and a tolerance of half that overshoot, so a planner stopping within
   it has crossed. Its `target` is the partner half, so it arrives once the runtime has teleported
   the robot there
3. a `StairsLegPhase` on the axis outside the partner's mouth, facing out, met only with the
   robot centre 0.1 m past the mouth

Steps and treads are solid geometry every sensor sees, the robot climbs them under its own
physics and its planner gets no hint that they are passable.

## Task modes (`TM_Robots` subclasses)

`TM_Robots` ([`__init__.py`](__init__.py)) is the base: one instance drives
all robots in its scope, exposing `reset`, `set_position`, `set_goal`, and
an async `done` flag. Shipped modes:

| Kind | File | Behavior |
| --- | --- | --- |
| `random` | [`random/`](random/) | one random reachable goal per robot per episode |
| `explore` | [`explore/`](explore/) | extends `random`; when a robot finishes or times out, a fresh random goal is assigned |
| `guided` | [`guided/`](guided/) | external controller drives the goal sequence. 2D Nav Goal clicks append waypoints, which show as numbered drag handles with Delete, Insert after and Clear chain menus |
| `stationary` | [`stationary/`](stationary/) | robot stays parked without goal dispatch, at `pos_x`/`pos_y`/`pos_theta` or a free pose sampled per episode |
| `scenario` | [`scenario/`](scenario/) | reads `start`/`goal` pairs from the world's scenario YAML. A robot past the `robots:` entries, or on an entry without `phases`, stays parked, and an episode with only parked robots runs until `task.episode.timeout` |
| `characterization` | [`characterization/`](characterization/) | open-loop maneuver sweep: publishes exact `cmd_vel` profiles through the robot's rated envelope (no nav goals), tags each maneuver with `characterization_phase` markers, odom stall watchdog |
| `null` | [`composite.py`](composite.py) | idle sink for robots unallocated by the fleet manager |
| `composite` | [`composite.py`](composite.py) | fan-out: each sub-TM sees a scoped `TaskContext` covering only its allocated robots |

Modes are registered by `Constants.TaskMode.TM_Robots` enum value in
[`tasks/registry.py`](../registry.py). `null` and `composite` are not in
the enum, they are only reachable via `set_tm_robots_composite`.

## Request types

`TM_Robots` subclasses build `TaskRequest` values and hand them to
`RobotManager.submit_task`. The types live in
[`arena_simulation_setup.shared.task`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/task.py)
so scenario YAML, runtime generators, the episode record and arena_evaluation
share one definition. [`request.py`](request.py) re-exports them with
`kind_of(phase) -> TaskKind`, the action kind an adapter dispatches.

- `TaskPhase`: one step. Every phase takes `on_failure`
  (`continue` | `stop_task` | `abort_episode`), `until` (clauses joined by
  ` or `, each atoms of the `conditions:` grammar joined by ` and ` with an
  optional ` for <seconds>` dwell; the phase completes once any clause has
  held, its atoms together, for its dwell, counted from arrival for a goto,
  or a structured condition node with `for:` holds, numeric comparisons and
  `some`/`each` over a field family, see
  [worlds/README.md](../../../../arena_simulation_setup/worlds/README.md)),
  `conditions` (clauses judged over the phase only)
  and `text` (an authored instruction overriding the rendered one).
- `LoopPhase(phases, until?)`: the phases repeated in order; the `until` is
  judged on every tick and, once it holds, drops the running phase and
  continues with the phase after the loop.
- `GoToPhase(pose | target, tolerance_radius?, tolerance_angle?, hold_time?, signal?)`:
  navigate to a pose, or to a zone, door, elevator or pedestrian by name. A
  zone target dispatches to a free map cell inside the zone with the robot's
  spawn clearance, a door or elevator to a point inside its polygon, and a
  `pose` authored next to the name (`{goto: pharmacy, pose: [4.0, 27.5, 0.0]}`)
  is kept as the dispatch pose.
  A pedestrian target dispatches to a standoff pose 0.8 m short of the
  pedestrian on the robot's side, facing them, taken from the judge's sample
  when the phase starts (`standoff_pose` in the task runner).
  Arrival on a named zone is `robot in <zone>`, on a pedestrian
  `robot within tolerance_radius of <ped>`. `hold_time` is the park time
  (stationary within 5 cm and 5 degrees) before the phase counts as met. With
  a `signal` such as `arrived` the robot itself ends the phase by sending that
  signal (read from `Adapter.signal`) and is judged against the tolerance at
  that moment: within it the phase is met, elsewhere the phase fails and the
  episode is aborted as `signaled <signal> <d> m from goal`. `submit_task`
  rejects a signal the robot's adapter cannot send (`Adapter.signals`). With
  neither pose nor target the phase holds the robot where it starts.
- `ReachPhase`, `PlayGesturePhase`: arm phases, completed by their action
  result. A reach pose's `frame` (the base frame by default) is a robot link,
  put under the robot's TF prefix unless it already carries it.
- `TaskRequest(phases, conditions?)`: ordered phase list plus clauses judged
  over the whole request.

Unset tolerances, hold time and signal take the `task.episode.goto_pose.*` launch
parameters at submit. Phase indices are episode-wide: a second `submit_task`
appends its phases and marks the unfinished ones of the previous request
dropped. A `LoopPhase` is expanded at submit: its first iteration takes the
indices where the loop stands, every further iteration is appended at the end
of the list when the previous one ends, and the loop's `until` is judged on
every tick a loop phase is active. Once it holds, the running phase and the
rest of its iteration are marked dropped and the phase after the loop becomes
active. A loop without `until` repeats until the episode ends. Everything is judged by the task runner
([`manager/robot_manager/task_runner.py`](../../manager/robot_manager/task_runner.py))
on the node's 30 Hz sim tick from a `Sample` of robot poses, ped positions and
semantic fields, with the same predicate code arena_evaluation replays offline
([`shared/judge.py`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/judge.py)).
The robot's progress is published as a `robot` semantic entity (`phase`, `met`,
`failed`, `dropped`, `violated`), its judged pose on `<robot_ns>/task_pose` (the
flattened world frame the judge and the compacted world share, not the env's map
frame, published once the episode's reset has landed the robot and a phase is
active), and the resolved phases in `EpisodeRecord.phases` together with
`map_poses`, the realized map-frame goto pose per phase.
[`shared/render.py`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/render.py)
renders any phase list to one instruction sentence per phase, and
[`shared/route.py`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/route.py)
words a goto as walking directions through the world's doors and openings.

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

Entry point: the `task.config` node param, read by
`Task.create` and at every reset. A set config takes precedence over
`task.robots`. The composite is rebound when the config path changes or the
fleet's robot names differ from the last allocation.

## Integration points

- **`RobotManager`** ([`manager/robot_manager/robot_manager.py`](../../manager/robot_manager/robot_manager.py)):
  one per spawned robot. Owns the bound adapters, the task runner with this
  episode's phases, the goal-republish loop, and the tf-backed `pose`
  property. Entry points used from here: `submit_task`, `stop_task`, `move`,
  `is_done`, `accepts`.
- **`RobotsManager`** ([`manager/robot_manager/robots_manager.py`](../../manager/robot_manager/robots_manager.py)):
  diff-driven fleet lifecycle. Parses the `robot` ROS param (comma-
  separated list of robot names, `model[count]`, or `.yaml` setup refs),
  reconciles against existing `RobotManager` instances, and creates /
  destroys / updates to match. Reset-time entry is `set_up`.
- **`TaskContext.robots`**: the mapping `TM_Robots` subclasses iterate;
  `TM_Composite` replaces it with a scoped view so each sub-TM only sees
  its fleet slice.
