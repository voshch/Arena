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
`task_generator_node`, started by the `env_node` console script through
[`env.launch.py`](../../launch/env.launch.py), which `arena_node` launches for
every `spawn_env`.

The task layer is reached only through [`task_layer.py`](task_layer.py):
`task_layer()` loads the module the `arena_runtime.task_layer` entry point group
names (`task_generator`'s
[`tasks/layer.py`](../../../../task_generator/task_generator/tasks/layer.py)),
and the `TaskLayer`, `EpisodeTask` and `RobotModes` protocols,
`TaskContext`, `ResetPlan` and `TaskParams` are all `EpisodeNode`
sees of it. The `task.robots`, `task.obstacles`, `task.config` and
`task.modules` params are the layer's `TaskParams`, reached as
`node.task_params`. The node is what task modes get as `self.node`, and it
builds the `TaskContext` they work on in `_set_up_task` (between the managers
and the world sync, the one abstract hook) and binds its collision callback on
the robots manager there. The optional hooks of `EnvNode` it overrides: simulator incidents (`_compromise`, logged by
default), its tick (`_on_tick`, after the robots are judged), entity snapshots
(`_extra_semantic_states`), env displays (`_interaction_displays`), a step once the
env is initialized (`_after_setup`) and its teardown (`_teardown_episode`, after the tick
loop stops, before the robots and the simulator), plus `on_task_submitted` and
`reset_timeline`.

### Episode loop

`_run_reset_cycle` builds the next record and resets the task, and
`_termination_watcher` ends the episode. Episode end state lives on the node:
`abort_episode(reason)` and `force_reset()` set the flags, `_episode_done`
returns true on a flag, on the `Robot.TIMEOUT` sim-time limit (which aborts with
`timeout`) or when the task is done.

### Reset semantics

The `resetting` parameter and `state/resetting` belong to the node. Async ROS I/O in
this package goes through [`arena_rclpy_mixins`](../../../../utils/arena_rclpy_mixins).

`EpisodeNode._reset_episode(seed)` runs in this order:

1. Publish `reset_start` (`std_msgs/Empty`), reseed the RNG and reset the
   scenario asset pointer.
2. `robots_manager.set_up()` reconciles the fleet (spawn and remove robots),
   then `launch_pending()`.
3. `environment_manager.before_reset_episode()` pauses the simulator. The sim
   is paused for the entire body below; only node-discovery and lifecycle
   signals are observable here.
4. Apply the staged task and human params.
5. `Task.prepare_reset(seed)` rebinds the modes the `task.*` params select,
   calls `module.before_reset()` for every active module, then
   `tm_robots.reset()` and `tm_obstacles.reset()`. It returns a `ResetPlan`
   with the obstacles, dynamic obstacles and lights to spawn.
6. `environment_manager.respawn(callback)` marks all current `INUSE` obstacles
   as `UNUSED`, runs the callback (which spawns the plan), then removes
   everything still `UNUSED`.
7. Every `RobotManager.reset(...)` runs. Failures are classified: the episode
   fails with the reasons, and a run of consecutive reset failures raises.
8. `Task.after_reset()` calls `module.after_reset()` for every active module.
9. `environment_manager.after_reset_episode()` unpauses the simulator, and
   `reset_end` is published.

**WORLD layer invariant:** entities spawned with `ObstacleLayer.WORLD` (walls,
doors, floors, world static entities) are never touched during `respawn`. They
survive all episode resets for the lifetime of the world.

### Hand spawns

`runtime/spawn_static` and `runtime/spawn_dynamic` spawn one obstacle into a running
episode, `runtime/spawn_robot` queues or spawns one robot. Without a pose,
[`placement.py`](placement.py) picks a free one. Each adds a reason to
`EpisodeRecord.integrity`.

### `EpisodeRecord`

Episode state is tracked in `EpisodeRecord` (an `attrs.define` dataclass on the
node). Key fields: `episode_id`, `world`, `seed` (derived via blake2b from
`run_seed|world|episode_id`), snapshot task-mode strings, `outcome_state`,
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
keeps every field it is not given, `tm_modules` included, and types per-mode
params by the node's declaration of `task.<mode>.<leaf>`. Task modes and
parameter names come from the mode registry and
[`constants/runtime.py`](constants/runtime.py): `episode_task_modes` reads the
live env, or the registry with defaults when no env runs. Spawn, move and
pedestrian poses are map-frame. Pausing the sim is a runtime hold, not an
episode tool.

`runtime/submit_task` (`task_generator_msgs/SubmitTask`) is the node service
behind `episode_submit_task` and the goals of the `robots` domain: a `phases:`
list or `{phases, conditions}` mapping in the scenario format, parsed by
`TaskRequest.parse` and handed to `RobotManager.submit_task`, answering with the
episode-wide phase indices. Goto poses are map-frame and shifted into scenario
coordinates the way an rviz goal handle's are. An empty phase list stops the
robot: `RobotManager.stop_task` drops its unfinished phases and cancels the
in-flight adapter goals.

### Goal progress

[`goal_progress.py`](goal_progress.py) is `EpisodeNode`'s ROS-independent per-robot goal
progress tracking. `GoalProgress` holds a robot's start distance,
running-minimum distance and accumulated path length toward its goal.
`closed_fraction` is `(start - min) / start`, `1.0` when the start distance was
zero. `GoalProgressTracker` keys `GoalProgress` by robot name and exposes
`least_progress()`, the tracked robot with the lowest `closed_fraction`. It is
sampled by `EpisodeNode._sample_goal_progress` and reset once per episode in
`_build_next_record`.

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

After `set_up()`, the active fleet is published latched on `state/robots` as a `task_generator_msgs/RobotFleet` carrying one `RobotState` (`descriptor` `{name, model, ns, frame}`, resolved `caps[]`, resolved `params[]`) per robot. Consumers should subscribe with `TRANSIENT_LOCAL` durability and read `robots[i].descriptor` for identity. (A `robot_names: list[str]` rosparam is also kept up to date on the node for back-compat with external tooling.)

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
| `submit_task(request, start=None)` | hand a typed `TaskRequest` to the adapter, routed across levels from `start` (the live pose when omitted), loop bodies routed and defaulted like top-level phases |
| `move(pose)` | teleport the robot via `EnvironmentManager.move_robot`, blocking until TF shows it at the target |
| `is_done` | whether the current task request is satisfied |
| `accepts` | `frozenset[TaskKind]`: the set of task kinds this robot's adapter handles |
| `set_up_robot(node_paths)` | async; spawn robot, bind adapter, start navigation stack |
| `bring_up_controllers()` | async; drive the robot's ros2_control controllers to active, raises on an explicit controller_manager refusal |
| `destroy()` | tear down adapter and remove robot from sim |

A teleport under a hold (every reset) advances the held sim one physics step at
a time through arena_node's `sim_lifecycle/step` until a TF sample newer than
the step puts the robot within 0.3 m of the target, at most 5 steps per
attempt and 3 attempts, re-sending the move on each. When arena_node refuses
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
