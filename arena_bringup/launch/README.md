# arena_bringup launch

Entry point: [`arena_runtime.launch.py`](arena_runtime.launch.py) (runtime: sim + `arena_node`). Task-generator envs are attached via `task_generator.launch.py`; the `arena launch` bash composite orchestrates both.

## Arguments

All arguments are declared with `LaunchArgument` (a thin wrapper around
`DeclareLaunchArgument` that also auto-appends to the description list and
exposes `.substitution` / `.dict` / `.param`).

The table below covers the combined argument surface of `arena launch`. Runtime
args (`sim`, `headless`, `world`, `use_sim_time`, `log_level`) go to
`arena_runtime.launch.py`; the rest go to `task_generator.launch.py` per env.

Old flat names (`tm_robots`, `mobile`, `env_n`, ...) still work with a warning, see [Deprecated launch args](../BRINGUP.md#deprecated-launch-args).

| Name | Type / choices | Default | Meaning |
|---|---|---|---|
| `log_level` | level / `{glob:lvl,…,default}` / yaml path | `warn` | Per-node log level via `NodeLogLevelExtension`. See [Log level](#log-level) below. |
| `robot` | string | `auto` | Robot model; must match a directory under `arena_robots/robots/`. `auto` resolves from the selected planner's `action_type` + `sensor_needs` (canonical defaults: `jackal` for differential_drive, `ridgeback_plus` for omnidirectional, `turtlebot` for image/depth signatures). Composable: `auto`, `auto[2]`, `auto,jackal`. |
| `robot.mobile` | string | derived from `sim` | Mobile adapter kind: `nav2`, `rosnav_rl`, `external`, `none`. Empty = `none` for dummy, `nav2` otherwise. |
| `robot.arm` | string | `moveit` | Arm adapter kind |
| `robot.planner` | string | `` (empty) | Top-level planner selector; resolves to `robot.mobile:=<adapter> robot.mobile.<selector>:=<name>` via `arena_planners.resolver` |
| `robot.train` | bool string | `false` | Training mode: robot adapters route `cmd_vel` from the RL agent |
| `robot.hearing` | `none` \| `bus` \| `srp` \| `seld` | `none` | Robot-side hearing layer ([arena_hearing](../../arena_hearing/README.md)): belief grid and a Nav2 speed-filter mask per fleet robot, merged into each robot's nav2 params, RViz displays. `bus` consumes simulator bus events, `srp` runs the untrained energy-onset + GCC-PHAT front-end on any array, `seld` runs the SELDnet front-end on the array its weights were trained on. Needs `acoustics:=arena` and the hearing feature (`arena feature hearing install`), dispatched by [launch/hearing](../../task_generator/launch/hearing/README.md). |
| `robot.hearing.policy` | `belief` \| `listen` \| `full` | `full` | Hearing mask layers: belief only, plus the corner listen cap, plus creep-yield at blind bends ([arena_hearing](../../arena_hearing/README.md)). |
| `robot.hearing.<param>` | node param | node default | Every other `robot.hearing.<param>:=<val>` reaches the hearing nodes as ROS param `<param>`. The launch-facing ones (`robot.hearing.seld.device`, `robot.hearing.srp.hop_s`, ...) are declared by arena_hearing's `hearing.launch.py`, empty = node default. Never forwarded to the robot adapters. |
| `robot.mobile.<key>:=<val>` | adapter-scoped | - | Override any kwarg the bound mobile adapter accepts. Lands as ROS param `robot.mobile.<key>` and overlays the cap-file YAML. Examples: `robot.mobile.local_planner:=teb`, `robot.mobile.global_planner:=smac`, `robot.mobile.agent:=jackal_pretrained`. |
| `robot.arm.<key>:=<val>` | adapter-scoped | - | Same shape for the arm cap. |
| `sim` | string | `gazebo` | Physics simulator: `dummy`, `gazebo`, `isaac`, or `mujoco`. `dummy` must be explicit. Standalone `arena env` may omit it (adopts the runtime's sim); if given explicitly it must match the running runtime. |
| `headless` | bool string | `False` | `true` = hide sim GUI (server-only). `arena launch` also suppresses rviz unless `rviz:=true` is explicit. |
| `viz` | bool string | `true` | `arena launch` only: run `arena viz --all` after envs are up. Forced `false` when `headless:=true` unless overridden. |
| `human` | string | `dummy` for `dummy` sim, `arena` (arena_humansim) for `gazebo`/`isaac`/`mujoco` | Human-simulator backend |
| `complexity` | string | `1` | `1` map+position known; `2` map known AMCL; `3` SLAM |
| `record.dir` | string | `` (empty) | Directory for data recording; empty disables |
| `record.auto` | bool string | `true` | `false` = do not auto-start the recorder even when `record.dir` is set (the benchmark runner starts its own) |
| `task.robots` | string | `explore` | Robot task mode (legacy single-kind shorthand) |
| `task.config` | string | `` (empty) | Path to a [TaskModeSpec YAML](../configs/tasks/README.md); empty -> synthesize from `task.robots` (wins if both set) |
| `task.obstacles` | string | `random` | Obstacle task mode |
| `task.modules` | string | `rviz_ui` | Comma-separated task modules to load |
| `task.scenario.file` | string | `` (empty) | Scenario for the `scenario` task modes (empty = use the `task.params` default) |
| `task.params` | string | `configs/task_generator.yaml` | Task-generator ROS parameter YAML |
| `task.episode.count` | int string | `-1` | Stop the env after N episodes (`-1` = run forever) |
| `task.episode.fail_on_collision` | bool string | `false` | Abort the episode as FAILED on robot footprint contact |
| `world` | string | `map_empty` | World name; resolved under `arena_simulation_setup/worlds/` |
| `world.lighting` | string | `authored` | `authored` renders the lights the world declares. `auto` also gives every zone with a ceiling and no lights of its own a calibrated ceiling rig, see [AUTHORING.md](../../arena_simulation_setup/AUTHORING.md). |
| `acoustics` | `none` \| `arena` | `none` | Acoustics simulator: `arena` runs arena_auditory (sound propagation, robot and human sound emission, rendering, playback). The `auditory.*` sub-keys below take effect only with `arena`, see [arena_auditory/README.md](../../arena_auditory/README.md) and [launch/acoustics](../../task_generator/launch/acoustics/README.md). `arena` needs the auditory feature (`arena feature auditory install`). |
| `auditory.<param>` | node param | node default | Every `auditory.<param>:=<val>` reaches each auditory node as ROS param `<param>`, coerced to the type of its default in `arena_auditory/params.py`. The rows below are declared by `arena_auditory.launch.py`, an empty value keeps the node default shown. |
| `auditory.output.device` | string | `auto` | PortAudio output device for workstation playback. `auto` tries `pulse`, `pipewire`, `default`, then the PortAudio default. `none` starts no listener renderer. |
| `auditory.output.block_size` | int string | `512` | Workstation audio callback block size. |
| `auditory.output.buffer_s` | float string | `0.04` | Workstation jitter buffer target, raise it on repeated underflows. |
| `auditory.output.motor.enabled` / `auditory.output.ambient.enabled` | bool string | `true` | Play robot motor or environment audio on the workstation. Emission, propagation and robot hearing continue when false. |
| `auditory.viz.enabled` | bool string | `false` | Start the propagation visualizer and publish its markers. |
| `auditory.propagation.backend` | `pyroomacoustics` \| `level3` \| `legacy` | `pyroomacoustics` | Propagation backend. |
| `auditory.portal.multi_hop.enabled` | bool string | `true` | Allow pyroomacoustics RIRs across multi-hop portal routes. |
| `auditory.rir.max_order` | int string | `3` | Image-source reflection order of every RIR. |
| `auditory.pedestrian_listeners.enabled` | bool string | `false` | Pedestrians are propagation listeners and receive sound stimuli through the human simulator. |
| `auditory.motor.enabled` | bool string | `true` | Robots emit drivetrain audio. |
| `auditory.motor.model` | `procedural` \| `wav` | `procedural` | Robot motor audio source. |
| `auditory.motor.trim_db` | float string | `0.0` | Live offset on the motor asset `level_db`, which sets the motor level everywhere (the procedural drivetrain at 1 m/s). |
| `auditory.listener.id` | string | `` (empty) | Microphone listener id of the listener renderer, the RViz auditory panel switches it at run time. |
| `auditory.viewport.height_m` | float string | `1.6` | Listening height of the viewport down-projection microphone. |
| `auditory.array.spec` | `stereo` \| `four_mic` \| `mono` \| path | `stereo`, `four_mic` when `robot.hearing` is `srp` or `seld` | Robot microphone array rendered by `array_renderer`. |
| `auditory.array.mount_frame` | string | `` (empty) | TF frame the robot microphone array is mounted on, `{prefix}` and `{base_frame}` expand, a bare leaf joins the robot prefix, empty uses the robot base frame. |
| `auditory.microphones` | YAML string | `[]` | Robot microphone mappings (owner, robot, placement, frame, index). |
| `auditory.static_sounds` | YAML string | `[]` | World-independent `sound` entities (radios, alarms), as a flat list of the same `Sound` schema used in `world.yaml`. Non-empty adds `sounds` to `task.modules` (already on whenever `acoustics` is not `none`). With `acoustics:=none` nothing hears them and the launch warns once. |
| `use_sim_time` | bool string | `true` | Use sim clock instead of wall clock |
| `env.n` | int string | `1` | Number of task-generator environments `arena launch` will spawn this invocation. Additive: if the runtime already has envs, these add to them rather than replace. |
| `env_d` | float string | `50` | Spacing (metres) between environments on the snail grid |
| `debug` | bool string | `False` | Enable debug features |
| `task.episode.auto_reset` | bool expression | `true` | `true` = standalone: node auto-advances episodes; `false` = managed: external controller drives resets via `lifecycle/reset_episode` |
| `optim` | comma-separated tokens | `$ARENA_OPTIM` or `` (empty) | Strip matching `<sensor>` blocks from each robot's URDF after xacro expansion (affects both Gazebo and Isaac via [`urdf.py`](../../arena_simulation_setup/src/arena_simulation_setup/utils/models/urdf.py)). Tokens: `no_camera` (strips `camera`/`depth`/`rgbd_camera`), `no_lidar` (strips `ray`/`gpu_lidar`). Unknown tokens warn and are ignored. Default reads `$ARENA_OPTIM` so you can set `export ARENA_OPTIM=no_camera,no_lidar` once per shell; CLI `optim:=...` overrides. |

## Log level

The `log_level` arg drives `NodeLogLevelExtension`, which injects `--log-level`
into each `Node` action based on the node's fully-qualified name. Four input
forms are accepted:

| Form | Example | Meaning |
|---|---|---|
| bare scalar | `log_level:=info` | Same level for every node (back-compat). |
| inline rule set | `log_level:='{**/nav2*/**:fatal, /dummy/node:warn, info}'` | Comma-separated `<glob>:<level>` entries inside `{...}`. A bare last entry is the default and expands to `**/*:<level>`. **Replaces** any prior rule set. |
| inline merge | `log_level:='+[/foo:debug, /bar/**:warn]'` (prepend) or `'[<rules>]+'` (append) | Comma-separated `<glob>:<level>` entries inside `[...]`. **Merges** into the current rule set; if the rule set is empty (e.g. when the merge form is used directly from the CLI), the action seeds it with the `base` default first (`warn` unless overridden) so a catch-all is always present. |
| YAML file | `log_level:=/path/to/rules.yaml` with `default: warn` and ordered `rules: [{match, level}, ...]` | Same semantics as the inline rule set. |

Rules match against the node's FQN (`<namespace>/<name>`) with **first-match-wins**
order. Globs are gitignore-style: `**` matches zero or more `/`-separated path
segments, `*` matches within one segment, leading `/` in patterns is stripped so
ROS-style FQNs (`/dummy/node`) match the same as bare paths. Levels are the ROS
canonical set: `debug | info | warn | error | fatal` (no aliases).

`SetGlobalLogLevelAction` is also invoked further down the launch tree (e.g. by
`task_generator`'s robot launcher to silence nav2 nodes by default) - those
later calls can use the merge form to layer rules on top of the user's spec
without clobbering it.

## Simulator dispatch

- [simulator/sim/README.md](simulator/sim/README.md) - physics simulator backends (`dummy`, `gazebo`, `isaac`, `mujoco`).
- Human-simulation backends: see `task_generator/launch/human/README.md`
  (moved alongside their `BaseHumanSimulator` adapters).

## Top-level composition

`arena_runtime.launch.py` assembles the following in order:

1. **`SetGlobalLogLevelAction`** - stores `log_level` in the launch context so
   `NodeLogLevelExtension` can inject `--log-level` into every subsequent `Node`
   action.
2. **`IsolatedGroupAction` -> `sim.launch.py`** - the physics simulator.
3. **`world_generator`** node (`arena_simulation_setup`) - generates world
   assets.
4. **`arena_node`** (`LifecycleNode`) - the multi-env orchestrator.

Task-generator envs are not included here. Each env is started separately via
`task_generator.launch.py` (either manually via `arena env` or orchestrated by
`arena launch`). Each env includes:

- `human.launch.py` - starts the human simulator (if any) for that environment.
- The `task_generator_node` with all forwarded args plus `namespace` and `prefix`.

Environments are positioned on a *snail grid* (`snail_grid(d)`) that spirals
outward from the origin with spacing `d`, so multiple parallel environments do
not overlap.

The simulator is paused during setup and the entire `Task._reset_episode` body -
see [Sim-paused invariant](../../arena_runtime/arena_runtime/arena_runtime/sim/README.md#sim-paused-invariant).

## utils/

| File | Purpose |
|---|---|
| [`utils/fake_localization.launch.py`](utils/fake_localization.launch.py) | Publishes a static `map -> odom` TF (zero transform). Args: `global_frame_id` (default `map`), `odom_frame_id` (default `odom`). Used for `complexity=1` (position known). |
| [`utils/map_server.launch.py`](utils/map_server.launch.py) | Starts `nav2_map_server` with `nav2_lifecycle_manager` (autostart, `bond_timeout=0`). No launch args - callers remap parameters directly. |
