# Simulator dispatch

Entry point: [`sim.launch.py`](sim.launch.py).

Called from `arena_runtime.launch.py` with `simulator`, `use_sim_time`, `world`, and
`headless`. Its sole job is to select and delegate to one simulator backend.

## SelectAction dispatch

`SelectAction` is a key->action registry resolved at launch time. The selector
expression is `LaunchConfiguration('simulator')`. Each registered key maps to
a `GroupAction` or `ExecuteProcess`.

```
simulator key  ->  action
──────────────────────────────────────────────────────────────────
dummy          ->  static_transform_publisher (map -> dummy TF only)
flatsim        ->  ExecuteProcess: `arena feature flatsim launch`
gazebo         ->  ExecuteProcess: `arena feature gazebo launch`
isaac          ->  ExecuteProcess: `arena feature isaac  launch`
mujoco         ->  ExecuteProcess: `arena feature mujoco launch`
```

The `simulator` `LaunchArgument` is declared *after* the `SelectAction` is
built so that `choices` can be derived from `launch_simulator.keys` - the
keys registered above. Passing an unregistered value causes a launch-time
validation error.

## Feature-gated delegation

The real simulators run via their feature `launch` verb: `sim.launch.py`
delegates with an `ExecuteProcess` (`bash -c 'arena feature <sim> launch ...'`,
`on_exit=Shutdown`). The verb refuses unless the feature is installed
(`arena feature <sim> install`), so the dispatcher is the single live entry and
the verb is never a dead path. Forwarded args: gazebo gets
`use_sim_time`/`headless`/`world`, isaac and mujoco get `headless`/`log_level`, flatsim
gets `sim.flatsim.dynamics` as `dynamics` and `sim.flatsim.physics` as `physics`. `sim:=flatsim` without the flatsim
package fails at launch with the install hint.

The real bringup files live wherever the feature owns them:

```
arena_bringup: launch/simulator/sim/gazebo/gazebo.launch.py   (gz-sim 8 + clock bridge)
arena_isaac:   run_isaacsim.launch.py                         (Isaac Sim app)
flatsim:       launch/flatsim.launch.py                       (flatsim server)
arena_mujoco:  run_mujoco.launch.py                           (MuJoCo server)
```

### gazebo.launch.py (arena_bringup)

- Reached via `arena feature gazebo launch`, so it only runs when the gazebo
  feature is installed.
- Stages models via `arena_simulation_setup.staging.stage()` at Python-load time.
- Sets `GZ_SIM_RESOURCE_PATH` and `GAZEBO_MODEL_PATH` from the staging
  directory plus `arena_robots` and any declared deps.
- Resolves the world SDF: looks for
  `arena_simulation_setup/worlds/<world>/worlds/<world>.world`; falls back to
  `arena_bringup/configs/gazebo/empty.sdf` if the file is absent.
- Launches `ros_gz_sim gz_sim.launch.py` (gz-sim 8, dart physics, ogre
  renderer). When `headless=True`, passes `-s` (server-only).
- Starts a `ros_gz_bridge parameter_bridge` for gz's clock on `/gz/clock` and the arena_runtime `clock_relay` that forwards it to `/clock` only when sim time advances (a paused gz loop republishes the same stamp every iteration).

### run_isaacsim.launch.py (arena_isaac)

- Reached via `arena feature isaac launch` (which, in docker, brings up the
  separate `isaac` compose service first).
- In docker the launch first builds `isaacsim_msgs` and `arena_people_msgs` from
  the workspace without symlinks into `install/.isaac_bridge`, which the `isaac`
  service mounts, so Isaac speaks the workspace's message definitions instead of
  the ones baked into the image. Before Isaac Sim starts,
  `arena_isaac/bridge_check.py` loads every type of both packages from that tree
  with Isaac's Python and compares the rclpy, rcl and Fast DDS versions of the
  two containers. Any failure or version mismatch stops the launch with the list.

## Adding a new simulator

1. Add a feature with a `launch` verb that exec's the real bringup
   (in `arena_bringup` or the feature's own package).
2. In `sim.launch.py`, call `launch_simulator.add("<name>", ExecuteProcess(...))`
   delegating to `arena feature <name> launch`.
3. The new key appears in `launch_simulator.keys` automatically, so the
   `simulator` argument's `choices` list updates without extra changes.
4. Add the corresponding `SimSimulator.<NAME>` entry in
   [`arena_bringup/constants.py`](../../../arena_bringup/constants.py).
