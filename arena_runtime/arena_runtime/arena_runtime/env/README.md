# env

The per-env layer: managers that own the simulator state (the robot fleet, the
map, and the environment's obstacle lifecycle plus simulator adapter), and
[`EnvNode`](node.py), the lifecycle node they run in, and its one subclass
[`EpisodeNode`](episode_node.py), which adds the episode loop.

## `EnvNode`

`EnvNode` is one env: its setup and teardown, the simulator, acoustics, hearing
and human simulator, the managers, sim holds, unpause windows and single steps,
the 30 Hz sim tick that builds the judge `Sample` and judges every robot, the
semantics registry, write path and `state/semantics` snapshot, the viz manifest,
and the services that need no episode: `lifecycle/wait_for_world`, the asset
`query/*` services, `semantics/set` and `runtime/require_map`.

[`testing/env_host.py`](../testing/env_host.py) hosts an `EnvNode` without a task layer
for tests that need a live node to run managers on.

## `EpisodeNode`

[`EpisodeNode`](episode_node.py) is the one subclass of `EnvNode`. It holds the
episode record and queue, the staged params, the reset cycle, the termination
watcher, the scenario timeline, the episode conditions, `run_episode` and the
entity spawn services (`runtime/spawn_*`, `runtime/move`). The ROS node is named
`Constants.ENV_NODE_NAMESPACE` (`arena_env_node`), the one source of the name
the registry, the env peers and the launch derive from, started by the `arena_env_node` console script through
[`env.launch.py`](../../launch/env.launch.py), which `arena_runtime_node` launches for
every `spawn_env`.

The runtime reaches the task layer only through [`task_layer.py`](task_layer.py):
`task_layer(name)` loads the module registered as `name` under the
`arena_runtime.task_layer` entry point group. The `task_layer` launch arg selects
it for an env, `$ARENA_TASK_LAYER` for a process that has no launch args (the
MCP domains, suite parsing), and the default is `task_generator`'s
[`tasks/layer.py`](../../../../task_generator/task_generator/tasks/layer.py).
`spawn_env` refuses a name nothing registers. `task_layer_name(name)` is the one
source of the runtime name (the argument, else `$ARENA_TASK_LAYER`, else
`task_generator`): the node resolves it once and sets its `task_layer` param to
it, so a live env reports the runtime it runs and the MCP tools on that env load
the same one. The `TaskLayer` and `EpisodeTask` protocols, `TaskContext`,
`TaskConfiguration`, `ResetPlan`, `RobotReset`, `EpisodeRecord`,
`TaskModeOverrides` and `TaskParams` are all `EpisodeNode` sees of it. The
selection params of the runtime's axes are its `TaskParams`, reached as
`node.task_params`. `TaskContext.interactive` is optional, and a mode that uses the rviz side asserts it.
A task layer reaches the runtime only through `task_layer.py` (the protocols, the
records and `TaskContext`), [`task_api.py`](task_api.py) (the runtime types and
helpers task modes call, such as `NodeInterface`, `TaskRequest`, `PhaseState`,
`Realizer` and `RobotManager`), `arena_runtime.shared` (the value types) and
`arena_runtime_msgs`. `task_generator/tests/unit/test_runtime_seam.py` holds
`task_generator` to that list. The node is what task modes get as `self.node`, and it
builds the `TaskContext` they work on in `_set_up_task` (between the managers
and the world sync, the one abstract hook) and binds its collision callback on
the robots manager there. The optional hooks of `EnvNode` it overrides: simulator incidents (`_compromise`, logged by
default), its tick (`async _on_tick(ended)`, after the robots are judged, `ended`
mapping each robot to the `PhaseState` that ended on this tick), entity snapshots
(`_extra_semantic_states`), env displays (`_interaction_displays`), a step once the
env is initialized (`_after_setup`), the arena watchdog's 1 s wall tick (`_on_watchdog`) and its teardown (`_teardown_episode`, after the tick
loop stops, before the robots and the simulator), plus `on_task_submitted` and
`reset_timeline`.

### Episode loop

`_build_next_record` builds the next record, `_run_reset_cycle` resets the task, and
`_termination_watcher` ends the episode. Episode end state lives on the node:
`abort_episode(reason)` and `force_reset()` set the flags, `_episode_done`
returns true on a flag, on the `Robot.TIMEOUT` sim-time limit (which aborts with
`timeout`) or when the task is done. Every sim tick the node judges the robots
and then, while the episode is RUNNING and the reset lock is free, awaits
`EpisodeTask.tick(ended, assigned)` with the phases that ended on it and the
robots given a task from outside (`SubmitTask`, `RunTask`) since the last task
tick, so a task can reassign a robot the moment its task finishes and keep its
hands off one assigned from outside. The node keeps that set, hands it over at
each task tick and clears it at every reset. A task error in the tick is logged
and does not stop the loop.

The next episode is one function of `EpisodeRuntime.pending` (a
`TaskModeOverrides`, merged into by `queue_episode` and `reset_episode`) and an
explicit world or seed: its id is the current one plus one, its world the
explicit one, else `pending.world`, else the loaded world, else the current one,
its seed the explicit one (when not negative), else `pending.seed` (when not
negative), else derived from `run_seed|world|episode_id`, and its modes the live
`task_params.resolve(pending.modes)`, the full selection the build then selects
(an empty value on a single-mode axis keeps the live mode, an empty modules axis
leaves the always-on modules). `state/queue` is published
from that function, so the preview equals the record the build then produces
unless `run_episode` or `reset_episode` pass a world or seed. A negative seed
from the runtime means derive it.

The runtime decides once per episode: `next_episode(previous=, queued=pending)`
runs when an episode's outcome is final (after it is published, so `previous`
carries the outcome the node assigned: timeout, no progress, collision,
SKIPPED) and the queue state is republished, and lazily at the start of the
build when it has not run. The answer replaces `pending` only when it came back:
a raising `next_episode` at episode end is logged and the build asks again, at
build time the error ends the episode FATAL, and `pending` is never lost. The
build clears `pending` and the decision only after it succeeded. One lock
serializes the decision and every merge into `pending` (`queue_episode`,
`reset_episode`), so a merge during the decision is kept and the decision never
overwrites one. A `reset_episode` that arrives while an episode is in flight sets
`EpisodeRuntime.respawn_requested` (and ends the running episode SKIPPED), which
`_run_episode` consumes to start the next episode, so one arriving after the
outcome is set is not lost.

### Reset semantics

The `resetting` parameter and `state/resetting` belong to the node. Async ROS I/O in
this package goes through [`arena_rclpy_mixins`](../../../../utils/arena_rclpy_mixins/arena_rclpy_mixins).

`EpisodeNode._reset_episode(record)` runs in this order:

1. Publish `reset_start` (`std_msgs/Empty`), reseed the RNG and reset the
   scenario asset pointer.
2. `robots_manager.set_up()` reconciles the fleet (spawn and remove robots),
   then `launch_pending()`.
3. `environment_manager.before_reset_episode()` pauses the simulator. The sim
   is paused for the entire body below; only node-discovery and lifecycle
   signals are observable here.
4. Apply the staged task and human params: each staged full name that the node
   declares is set in one atomic batch, an undeclared one is warned about and
   dropped.
5. `Task.reset_robots(record)` rebinds the modes the `task.*` params select, calls
   `module.before_reset()` for every active module, resets the robot mode and
   hands its `ResetPlan` to `module.on_reset_plan(plan)`. The plan holds a
   `RobotReset` per robot (`start_pose`, `task`, `humans`) and the
   `forbidden_zones`. The node submits each task from its start pose and forbids
   the zones. Then `Task.reset_obstacles(plan, record)` adds
   the obstacles, dynamic obstacles and lights to spawn.
6. `environment_manager.respawn(callback)` marks all current `INUSE` obstacles
   as `UNUSED`, runs the callback (which spawns the plan), then removes
   everything still `UNUSED`.
7. Every `RobotManager.reset(...)` runs, placed at the `start_pose` of its entry. Failures are classified: the episode
   fails with the reasons, and a run of consecutive reset failures raises.
8. `Task.after_reset()` calls `module.after_reset()` for every active module.
9. `environment_manager.after_reset_episode()` unpauses the simulator, and
   `reset_end` is published.

Each reset logs one info line, `reset took <s> (<step> <s>, ...)`, with the wall-clock
seconds of the hold, the world swap and each step above. `setup()` logs `setup took`
the same way for the env's one-time setup.

**WORLD layer invariant:** entities spawned with `ObstacleLayer.WORLD` (walls,
doors, floors, world static entities) are never touched during `respawn`. They
survive all episode resets for the lifetime of the world.

### Task runtimes

`task_generator` is one task runtime among any that register under the
`arena_runtime.task_layer` entry point group. A runtime is a module with the
functions of the `TaskLayer` protocol, selected with `task_layer:=<entry point
name>`:

```python
setup(entry_points={"arena_runtime.task_layer": ["my_runtime = my_runtime.layer"]})
```

| `TaskLayer` | What it does |
|---|---|
| `declare_task_params(node)` | Returns the `TaskParams`: `resolve(modes)` is the selection `select(modes)` would make live, `select(modes)` sets the listed axes. |
| `declare_mode_schema(node, axis, mode)` | Declares the params of one mode. `declare_mode_schemas(layer, node)` declares every registered mode. |
| `create_task(node=, ctx=, params=)` | Binds an `EpisodeTask` to the env's `TaskContext`. |
| `mode_namespaces()` | The registered modes per axis, each with its param namespace: the one source of param names. `registered_modes(layer)` is its names. |
| `multi_axes()` | The axes that take several modes at once, the ones it passes to `validate_modes`. |
| `fleet_only_robots()` | Robot modes reachable only through a fleet config. |
| `next_episode(previous=, queued=)` | Chooses what the next episode changes, given the finished record and what was queued. |

`mode_param(axis)` is the parameter that selects an axis, `selection_args(modes)`
the `k:=v` launch arguments that select a whole selection, and
`validate_modes(modes, vocabulary, multi_axes)` the one validator: a layer
passes its own vocabulary and its multi-mode axes, the runtime answers the
canonical selection.

A runtime owns its axes. `TaskModes` is a dict of mode names per axis,
`TaskModeOverrides(modes, world, seed)` is what an episode changes (an axis left
out, an empty world and a negative seed keep what is live) and `EpisodeRecord.modes`
is the selection an episode ran with. `task_generator` has the axes `robots`,
`obstacles` and `modules` and returns `queued` unchanged. The wire form of
modes is `encode_modes` and `decode_modes`, one `KeyValue` per axis with the
names comma-separated. `QueueEpisode.modes`, `EpisodeRecord.modes`,
`GetTaskModes.modes` and `QueryTaskModes.modes` carry any axis and are the only
mode fields on the wire. `QueueEpisode` validates the request's modes with
`TaskLayer.validate_modes` and answers its `ValueError` message unchanged as
`error_msg`: an unknown axis, an unknown mode and a single-mode axis without
exactly one name are refused, `[]` clears a multi-mode axis such as modules.
`QueryTaskModes.namespaces` is `mode_namespaces()` on the wire (`encode_namespaces`,
`decode_namespaces`): one `KeyValue` per registered mode, keyed `<axis>/<mode>`, valued
with the mode's param namespace. Clients take param names from it and never build
`task.<mode>`. `EpisodeRecord.task_layer` is the name of the runtime that ran the
episode, the env's resolved `task_layer` param.

Staged params are one dict by full name. `QueueEpisode.params` carries full names
and upserts them. Changing an axis's mode drops
the staged names under the namespaces the old selection used and the resolved new
selection does not, so always-on modules never look switched off. `EpisodeRecord.params` carries the effective params of the selected
modes, with the staged values over them on a QUEUED record.

`arena_evaluation` reads the vocabulary and the mode params through this seam, and
`$ARENA_TASK_LAYER` selects the runtime for suite parsing.

### Reassigning a robot mid-episode

`RobotManager.submit_task` drops the robot's unfinished phases and appends the
new ones, `RobotManager.stop_task` drops them and cancels the adapter goals:
that pair is the reassign operation, from a task's `tick` as much as from the
services. The runner state stays reachable as `ctx.robots[name].runner`
(`TaskRunner.active_state`, `outcomes(outcome)`).

The node serves it two ways through one private method that parses the YAML,
shifts map-frame poses, submits under the reset lock (or stops the robot on an
empty phase list) and returns the manager and the new phase indices:
`runtime/submit_task` (`SubmitTask`, answers with the indices) and
`runtime/run_task` (`RunTask` action). A `RunTask` goal is tracked off the
runner on each tick: feedback names each of its phases as it leaves `pending`,
and the goal ends `met` when all its phases are met, `failed` once its phases
have ended and one failed (under `on_failure: continue` the robot runs the rest
first, a failure policy ends it at once), `dropped` when one was dropped (a later submit or goal for the robot, a stop,
an episode reset, detected by the episode id the goal was assigned in). An empty
task ends `met` at once. A goal's outcome comes only from
`TaskRunner.request_outcome` for the request its submit made. A new goal for a
robot with a running goal replaces it, the old one ending `dropped`. Cancel stops
the robot (`stop_task`, which drops with `DropReason.STOPPED`) only while the
goal's request is open and ends the goal canceled, a cancel that arrives while
the assign still waits is honored once it returns. `met` and `failed` succeed the
action, any other `dropped` aborts it. Refused input (unknown robot, bad YAML, env not initialized) aborts the goal
with `error_msg`. A loop's exit marks its remaining phases `dropped`, so a goal
over a loop ends `dropped`.

### Hand spawns

`runtime/spawn_static` and `runtime/spawn_dynamic` spawn one obstacle into a running
episode, `runtime/spawn_robot` queues or spawns one robot. Without a pose,
[`placement.py`](placement.py) picks a free one. Each adds a reason to
`EpisodeRecord.integrity`.

### `EpisodeRecord`

Episode state is tracked in `EpisodeRecord` (an `attrs.define` dataclass in
[`task_layer.py`](task_layer.py)). Key fields: `episode_id`, `world`, `seed`
(derived via blake2b from `run_seed|world|episode_id`, or chosen by the runtime),
`modes` (the selection per axis), `outcome_state`,
`outcome_info`, `goal_uuid`, and `integrity`. A new record is created at every
NEXT reset. `integrity` lists why the episode is compromised and starts empty:
every manual mutation (`runtime/spawn_*`, `runtime/move`, `runtime/despawn_robot`,
`set_robot_position`, `set_robot_goal`, task parameters changed mid-episode) and
every failure the simulator adapter cannot recover from (`set_incident_callback`)
adds a reason.
Consumers treat `not record.integrity` as intact.

### MCP domain `episode`

[`episode_mcp.py`](episode_mcp.py) contributes the `episode` domain to the
[arena_mcp](../../../../arena_mcp/README.md) server, a client of the node of one
env. List its tools with `python3 -m arena_mcp ls episode`. The `entities`
domain of the same node is in
[arena_runtime](../README.md#mcp-domain-entities).

Tools: reset (blocking until the new episode runs), run through the
`RunEpisode` action as a job on a managed env (`task.episode.auto_reset` false,
an auto-advancing env answers busy, `jobs_wait` for the outcome, `jobs_cancel`
cancels the goal), wait for an episode state, current and queued records, the
history of records with their final outcomes, queueing the next episode, task
modes with their parameter schemas, the `task.episode.*` parameters, the startup
configuration, the asset catalogs and task submission to a robot. Resources
`arena://episode/{env_id}/current` and `arena://episode/{env_id}/history`.

Every tool takes `env_id`, optional when exactly one env runs. `episode_queue`
keeps every field it is not given and takes `modes` (axis to a name or a list of
names, `[]` clears an axis with many modes like modules, an axis with one mode
needs exactly one name), full-name `params`, validates `modes` with the env
layer's `validate_modes` and types params by the
node's declaration. Task modes and parameter names come from the env's task
runtime, which a tool on a live env loads by the env's `task_layer` param, and
[`constants/runtime.py`](constants/runtime.py): `episode_task_modes` reads the
live env, or the registry of the runtime `task_layer` names (default
`task_layer_name()`) with defaults when no env runs, `episode_config` takes the
same optional `task_layer`. `episode_task_modes_active` answers `modes` per axis. Spawn, move and
pedestrian poses are map-frame. Pausing the sim is a runtime hold, not an
episode tool.

`runtime/submit_task` (`arena_runtime_msgs/SubmitTask`) is the node service
behind `episode_submit_task` and the goals of the `robots` domain: a `phases:`
list or `{phases, conditions}` mapping in the scenario format, parsed by
`TaskRequest.parse` and handed to `RobotManager.submit_task`, answering with the
episode-wide phase indices. Goto poses are map-frame and shifted into scenario
coordinates the way an rviz goal handle's are. An empty phase list stops the
robot: `RobotManager.stop_task` drops its unfinished phases and cancels the
in-flight adapter goals.

A phase that ends while its action still runs (a reach `stop` holding, a loop `until`
exit, a failure policy or an episode abort) has that action canceled and the cancel answered
before the next phase dispatches. A reach the judge met by `stop` reads no status from its
client while the action runs, so the canceled status that follows does not fail it.

### Goal progress

[`goal_progress.py`](goal_progress.py) is `EpisodeNode`'s ROS-independent per-robot goal
progress tracking. `GoalProgress` holds a robot's start distance,
running-minimum distance and accumulated path length toward its goal.
`closed_fraction` is `(start - min) / start`, `1.0` when the start distance was
zero. `GoalProgressTracker` keys `GoalProgress` by robot name and exposes
`least_progress()`, the tracked robot with the lowest `closed_fraction`. It is
sampled by `EpisodeNode._sample_goal_progress` and reset once per episode in
`_build_next_record`.

### Hung nodes

While an episode runs, the node pings every node under its env namespace each
`task.episode.node_ping.period` wall seconds, driven by `EnvNode`'s 1 s arena
watchdog tick through `_on_watchdog`. [`node_watch.py`](node_watch.py) pings a node on its
lifecycle `get_state` service where it has one (it must report active) and on its
parameter service otherwise. A node that has answered before and then misses
`task.episode.node_ping.misses` pings in a row, or leaves the graph mid-episode,
ends the episode FATAL naming that node, which the benchmark runner counts as a
systemic step failure and retries. A round most nodes miss counts for nobody, so a
stalled box or event loop does not fault one node. With `task.episode.auto_reset`
on (standalone) a hung node is only logged.

## Map generator

[`map_generator.py`](map_generator.py) is the standalone `generate_map` console
script of arena_runtime. It procedurally generates a hallway-and-rooms world:
random room widths on each side of a central hallway, door gaps, wall YAML,
zones YAML, `map.yaml` and a PNG image, written to
`$ARENA_WS_DIR/src/arena/simulation-setup/worlds/<map_name>`. The env does not
use it at runtime.

## `RobotsManager`

[`robot_manager/robots_manager.py`](robot_manager/robots_manager.py)

Diff-driven fleet lifecycle. Parses the `robot` ROS param (comma-separated
list of `model`, `model[count, item, ...]`, or `.yaml` setup file references)
into a `_RobotDiff` that records robots to add, update, or remove. Entry
point is `set_up()`, called at the start of every episode reset.

Bracket items are `<int>` (count), `<cap>.adapter=<kind>` (per-robot adapter
kind, e.g. `mobile.adapter=drl`), or a bare `<type>=<value>` morphology key
(sensor/actuator parts; currently hard-errors as not yet implemented).
`[...]` is a shell glob class: quote the whole entry, e.g.
`robot:='jackal[2,mobile.adapter=drl]'`. `arena robot caps <model>` lists a
model's caps, mounts, catalog variants, and default sensors before you write
the bracket.

| Method / property | Purpose |
| --- | --- |
| `managers` | `dict[str, RobotManager]`: live instances keyed by robot name |
| `set_up()` | apply `_diff`: destroy removed managers, update changed ones, create new ones; awaits `RobotManager.set_up_robot()` for each addition |
| `provide_node_paths(paths)` | context manager that polls `get_node_names_and_namespaces()` into a set while fleet setup runs |

Parsing rules:
- Explicit names (`name=model`) are matched first against existing managers.
- Anonymous entries (`model`) match any existing manager whose robot is
  compatible (same model family), then fill new names with auto-generated
  suffixes.
- The `robot` param is a `ROSParamT[_RobotDiff]` with a parse callback, so
  it re-computes the diff on every parameter change.

After `set_up()`, the active fleet is published latched on `state/robots` as a `arena_runtime_msgs/RobotFleet` carrying one `RobotState` (`descriptor` `{name, model, ns, frame}`, resolved `caps[]`, resolved `params[]`) per robot. Consumers should subscribe with `TRANSIENT_LOCAL` durability and read `robots[i].descriptor` for identity. (A `robot_names: list[str]` rosparam is also kept up to date on the node for back-compat with external tooling.)

## `RobotManager`

[`robot_manager/robot_manager.py`](robot_manager/robot_manager.py)

One instance per spawned robot. Owns:

- The bound `Adapter`, resolved from `robot.adapters` through `ADAPTERS`.
- The current `TaskRequest` and phase index.
- A goal-republish timer loop.
- A tf-backed `pose` property (reads the robot's base frame from `tf_buffer`).

Key public surface:

| Method / property | Purpose |
| --- | --- |
| `robot` | the `Robot` config (model, name, initial pose) |
| `pose` | current `Pose` in the map frame; `None` during reset/respawn windows |
| `start_pos` / `goal_pos` | last set start and goal positions |
| `goal` | pose of the first `GoToPhase` in the current `TaskRequest`, or `None` |
| `submit_task(request, start=None)` | hand a typed `TaskRequest` to the adapter, routed across levels from `start` (the live pose when omitted), loop bodies routed and defaulted like top-level phases, dropping the unfinished phases of the previous request |
| `stop_task()` | drop the unfinished phases and cancel the in-flight adapter goals |
| `move(pose)` | teleport the robot via `EnvironmentManager.move_robot`, blocking until TF shows it at the target |
| `is_done` | whether the current task request is satisfied |
| `accepts` | `frozenset[TaskKind]`: the set of task kinds this robot's adapter handles |
| `set_up_robot(node_paths)` | async; spawn robot, bind adapter, start navigation stack |
| `bring_up_controllers()` | async; drive the robot's ros2_control controllers to active, raises on an explicit controller_manager refusal |
| `destroy()` | tear down adapter and remove robot from sim |

A teleport under a hold (every reset) advances the held sim one physics step at
a time through arena_runtime_node's `sim_lifecycle/step` until a TF sample newer than
the step puts the robot within 0.3 m of the target, at most 5 steps per
attempt and 3 attempts, re-sending the move on each. When arena_runtime_node refuses
the step (no hold, as for a mid-episode teleport, or another caller's unpause
window open) or does not answer it within 10 s, the teleport runs inside an
unpause window instead and polls TF for up to 2 s per attempt.

`_launch_robot` order: unpause window, one sim step, tracked launch, adapter
readiness, `bring_up_controllers()`, `adapter.on_controllers_active` (arm
adapters park every arm at its `stow` pose and wait until it settled), window
close. The first reset then places the robot with its arms already parked.
Controllers are driven here, not by `controller_manager/spawner`: each round
reads `list_controllers` and issues the one call the snapshot needs (load an
absent one, configure an unconfigured one, one grouped STRICT switch for all
inactive ones, nothing during a transient state), so a lost or slow reply costs
a re-read rather than a failure. The wait is unbounded, switches go out in
10 s bursts retried on controller_manager-side timeout, and only an explicit
refusal or a finalized controller raises.

## `WorldManager`

[`world_manager/world_manager.py`](world_manager/world_manager.py)

Owns the static map and the loaded `World`. Provides free-cell sampling for
obstacle and robot placement.

| Method / property | Purpose |
| --- | --- |
| `world` | `World`: levels with their zones, walls, doors and objects, world-scope schedules and signals, the KG |
| `map` | `WorldMap`: occupancy grid + resolution + origin |
| `update_world(world_map, world_description)` | replace the active world; marks world-entity cells as occupied |
| `get_positions_on_map(n, safe_dist, ...)` | sample `n` free positions with mutual safe distance |
| `get_position_on_map(safe_dist, ...)` | single-position convenience wrapper |
| `forbid(zones)` | mark `PositionRadius` zones as forbidden for subsequent sampling |
| `forbid_clear()` | clear all dynamically forbidden cells |

`WorldManagerROS` ([`world_manager/world_manager_ros.py`](world_manager/world_manager_ros.py))
extends `WorldManager` with ROS wiring: subscribes to the `map` topic,
uses `ClientWrapper` to call `map_server/load_map`, and triggers registered
callbacks when the world changes. The `start()` coroutine must be awaited
before the first reset.

`require_map_server()` launches `map_server` and its lifecycle manager under
the task generator namespace, logging by the env's `log_level` rule set. When the server does not report ACTIVE,
`ensure_map_server()` shuts that bringup down and launches a fresh one in the
same namespace.

It also publishes the latched `world_markers` MarkerArray
([`world_manager/markers.py`](world_manager/markers.py)): zone outlines and
labels, walls, doors and elevators, one marker namespace per layer so rviz
toggles them individually. Doors and elevators are recolored from the
semantic snapshot on every state change.

`sync(timeout)` blocks until the internal `_map_name` matches `_world_name`,
indicating the map server has processed the latest world change.

## `EnvironmentManager`

[`environment_manager.py`](environment_manager.py)

Adapter between the task system and the two simulator layers
([`BaseSim`](../sim/README.md) for physics objects,
[`BaseHumanSimulator`](../humans/README.md) for pedestrian logic).
All obstacle/robot operations go through here.

| Method | Purpose |
| --- | --- |
| `spawn_world_obstacles(world, detected_walls=None, world_map=None)` | spawn floors, walls, doors, static WORLD entities; under debug.map_source:=disk, per-level occupancy-derived walls in detected_walls are fed to the human-sim as collision-only geometry; `world_map` seeds `collision_grid` |
| `collision_grid` | per-world labelled occupancy ([collision_grid.py](collision_grid.py)): the map's walls layer as MAP, doors/elevators cleared, authored walls rasterized as WALL, every spawned static footprint stamped as STATIC; the robot collision tracker reads it |
| `spawn_obstacles(setups)` | spawn episode-scoped static obstacles (`INUSE`) |
| `spawn_dynamic_obstacles(setups)` | spawn episode-scoped dynamic obstacles (`INUSE`) |
| `spawn_robot(robots)` | spawn robots in both sim and human-sim layers |
| `move_robot(robots)` | teleport robots |
| `remove_robot(robots)` | remove robots from both layers |
| `respawn(callback)` | `unuse_obstacles()` → `callback()` → `remove_obstacles(UNUSED)` |
| `notify_stimulus(agent, stimulus, intensity)` | forwards to the human simulator, `agent` is an engine id or a spawned agent's name or `sim_path` |
| `reset(purge)` | remove all obstacles at or below `purge` layer |
| `before_reset_episode()` | delegates to `BaseSim.before_reset_episode()` (pauses sim) |
| `after_reset_episode()` | delegates to `BaseSim.after_reset_episode()` (unpauses sim) |

It also publishes the latched `affordance_markers` MarkerArray
([`affordance_markers.py`](affordance_markers.py)) with the placed affordance
slots of every spawned static obstacle, world and episode alike: per slot its
box with the top diagonals and an arrow from the box edge for each allowed
facing, one marker namespace per kind (`sit`). A slot an agent holds, walking up or seated, is pink
and the slots overlapping it grey, from `AffordanceClaims` over the seats the
human simulator reports held on the object (`held_seats()`). It is republished
whenever obstacles spawn or are removed and whenever the held seats change, and
rviz shows it as the
Affordances display in the Interaction group, off by default.

**Obstacle/pedestrian removal is split:** `BaseHumanSimulator.unuse_obstacles`
calls `_remove_obstacles_impl` (the human-sim-side pedestrian removal) and
flips `INUSE` layers to `UNUSED`. The subsequent `remove_obstacles(UNUSED)`
then calls `obstacle_delete` / `pedestrian_delete` on the physics simulator.
This two-phase pattern keeps the human-sim state consistent with the physics
sim. See [Reset semantics](#reset-semantics) for the
full episode-reset ordering and WORLD-layer invariant.

## `Realizer`

[`realizer.py`](realizer.py)

Coordinate-frame transform applied to all entities before they reach the
simulator. Translates by `(x, y)` and prepends a `prefix` to names and sim
paths, supporting multi-robot namespacing in a shared world.

Handles: `str`, `Position`, `Pose`, `Wall`, `Door`, `Floor`, `Elevator`,
`Entity`, `DynamicObstacle`.
