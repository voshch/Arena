# task_generator manager

Runtime managers that own the simulator state: the robot fleet, the map, and
the environment (obstacle lifecycle + simulator adapter).

## `RobotsManager`

[`robot_manager/robots_manager.py:45`](robot_manager/robots_manager.py#L45)

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

[`robot_manager/robot_manager.py:35`](robot_manager/robot_manager.py#L35)

One instance per spawned robot. Owns:

- The bound `Adapter` (resolved from `robot.adapters` via
  `Constants.TaskMode.TM_Robots`).
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
| `submit_task(request)` | hand a typed `TaskRequest` to the adapter |
| `move(pose)` | teleport the robot via `EnvironmentManager.move_robot` |
| `is_done` | whether the current task request is satisfied |
| `accepts` | `frozenset[TaskKind]`: the set of task kinds this robot's adapter handles |
| `set_up_robot(node_paths)` | async; spawn robot, bind adapter, start navigation stack |
| `bring_up_controllers()` | async; drive the robot's ros2_control controllers to active, raises on an explicit controller_manager refusal |
| `destroy()` | tear down adapter and remove robot from sim |

`_launch_robot` order: unpause window, one sim step, tracked launch, adapter
readiness, `bring_up_controllers()`, window close, `adapter.on_controllers_active`.
Controllers are driven here, not by `controller_manager/spawner`: each round
reads `list_controllers` and issues the one call the snapshot needs (load an
absent one, configure an unconfigured one, one grouped STRICT switch for all
inactive ones, nothing during a transient state), so a lost or slow reply costs
a re-read rather than a failure. The wait is unbounded, switches go out in
10 s bursts retried on controller_manager-side timeout, and only an explicit
refusal or a finalized controller raises.

## `WorldManager`

[`world_manager/world_manager.py:16`](world_manager/world_manager.py#L16)

Owns the static map and `WorldDescription`. Provides free-cell sampling for
obstacle and robot placement.

| Method / property | Purpose |
| --- | --- |
| `world` | `WorldDescription`: walls, zones, entities |
| `map` | `WorldMap`: occupancy grid + resolution + origin |
| `update_world(world_map, world_description)` | replace the active world; marks world-entity cells as occupied |
| `get_positions_on_map(n, safe_dist, ...)` | sample `n` free positions with mutual safe distance |
| `get_position_on_map(safe_dist, ...)` | single-position convenience wrapper |
| `forbid(zones)` | mark `PositionRadius` zones as forbidden for subsequent sampling |
| `forbid_clear()` | clear all dynamically forbidden cells |

`WorldManagerROS` ([`world_manager/world_manager_ros.py:96`](world_manager/world_manager_ros.py#L96))
extends `WorldManager` with ROS wiring: subscribes to the `map` topic,
uses `ClientWrapper` to call `map_server/load_map`, and triggers registered
callbacks when the world changes. The `start()` coroutine must be awaited
before the first reset.

`sync(timeout)` blocks until the internal `_map_name` matches `_world_name`,
indicating the map server has processed the latest world change.

## `EnvironmentManager`

[`environment_manager.py:20`](environment_manager.py#L20)

Adapter between the task system and the two simulator layers
([`BaseSim`](../../../arena_runtime/arena_runtime/arena_runtime/sim/README.md) for physics objects,
[`BaseHumanSimulator`](../simulators/human/README.md) for pedestrian logic).
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
| `reset(purge)` | remove all obstacles at or below `purge` layer |
| `before_reset_episode()` | delegates to `BaseSim.before_reset_episode()` (pauses sim) |
| `after_reset_episode()` | delegates to `BaseSim.after_reset_episode()` (unpauses sim) |

**Obstacle/pedestrian removal is split:** `BaseHumanSimulator.unuse_obstacles`
calls `_remove_obstacles_impl` (the human-sim-side pedestrian removal) and
flips `INUSE` layers to `UNUSED`. The subsequent `remove_obstacles(UNUSED)`
then calls `obstacle_delete` / `pedestrian_delete` on the physics simulator.
This two-phase pattern keeps the human-sim state consistent with the physics
sim. See [Reset semantics](../tasks/README.md#reset-semantics) for the
full episode-reset ordering and WORLD-layer invariant.

## `Realizer`

[`realizer.py:21`](realizer.py#L21)

Coordinate-frame transform applied to all entities before they reach the
simulator. Translates by `(x, y)` and prepends a `prefix` to names and sim
paths, supporting multi-robot namespacing in a shared world.

Handles: `str`, `Position`, `Pose`, `Wall`, `Door`, `Floor`, `Elevator`,
`Entity`, `DynamicObstacle`.
