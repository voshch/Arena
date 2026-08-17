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
| `robot.mobile.<key>:=<val>` | adapter-scoped | - | Override any kwarg the bound mobile adapter accepts. Lands as ROS param `robot.mobile.<key>` and overlays the cap-file YAML. Examples: `robot.mobile.local_planner:=teb`, `robot.mobile.global_planner:=smac`, `robot.mobile.agent:=jackal_pretrained`. |
| `robot.arm.<key>:=<val>` | adapter-scoped | - | Same shape for the arm cap. |
| `sim` | string | `gazebo` | Physics simulator: `dummy`, `gazebo`, or `isaac`. `dummy` must be explicit. Standalone `arena env` may omit it (adopts the runtime's sim); if given explicitly it must match the running runtime. |
| `headless` | bool string | `False` | `true` = hide sim GUI (server-only). `arena launch` also suppresses rviz unless `rviz:=true` is explicit. |
| `viz` | bool string | `true` | `arena launch` only: run `arena viz --all` after envs are up. Forced `false` when `headless:=true` unless overridden. |
| `human` | string | `dummy` for `dummy` sim, `arena` for `gazebo`/`isaac` | Human-simulator backend (`none` suppresses it) |
| `complexity` | string | `1` | `1` map+position known; `2` map known AMCL; `3` SLAM |
| `record.dir` | string | `` (empty) | Directory for data recording; empty disables |
| `record.auto` | bool string | `true` | `false` = do not auto-start the recorder even when `record.dir` is set (the benchmark runner starts its own) |
| `task.robots` | string | `explore` | Robot task mode (legacy single-kind shorthand) |
| `task.config` | string | `` (empty) | Path to a [TaskModeSpec YAML](../configs/tasks/README.md); empty -> synthesize from `task.robots` (wins if both set) |
| `task.obstacles` | string | `random` | Obstacle task mode |
| `task.modules` | string | `rviz_ui` | Comma-separated task modules to load |
| `task.scenario` | string | `` (empty) | Sets the `task.scenario.file` ROS param (empty = use the `task.params` default) |
| `task.params` | string | `configs/task_generator.yaml` | Task-generator ROS parameter YAML |
| `task.episodes` | int string | `-1` | Stop the env after N episodes (`-1` = run forever) |
| `task.fail_on_collision` | bool string | `false` | Abort the episode as FAILED on robot footprint contact |
| `world` | string | `map_empty` | World name; resolved under `arena_simulation_setup/worlds/` |
| `auditory` | `none` \| `arena` | `none` | Auditory pipeline: sound propagation, robot hearing, robot and human sound emission. Sub-keys below take effect only when not `none`; see [auditory/README.md](../../task_generator/launch/human/auditory/README.md). |
| `auditory.playback` | string | `auto` | PortAudio output device for workstation playback; `auto` = system default, `none` starts no playback nodes. |
| `auditory.viz` | bool string | `false` | Publish propagation markers. |
| `auditory.static_devices` | YAML string | `[]` | World-independent environment audio systems (radios, alarms); non-empty adds `audio_systems` to `task.modules`. |
| `auditory.motor` | `off` \| `wav` \| `procedural` | `procedural` | Robot motor audio source. |
| `auditory.environment_playback` | bool string | `true` | Play propagated environment audio locally without disabling simulated emission. |
| `auditory.block_size` | int string | `2048` | PortAudio callback size; raise to `4096` on repeated underflows. |
| `auditory.assets` / `auditory.sound_dir` | paths | bundled files | Asset catalog and WAV directory shared by all playback nodes. |
| `use_sim_time` | bool string | `true` | Use sim clock instead of wall clock |
| `env.n` | int string | `1` | Number of task-generator environments `arena launch` will spawn this invocation. Additive: if the runtime already has envs, these add to them rather than replace. |
| `env_d` | float string | `50` | Spacing (metres) between environments on the snail grid |
| `debug` | bool string | `False` | Enable debug features |
| `task.auto_reset` | bool expression | `true` | `true` = standalone: node auto-advances episodes; `false` = managed: external controller drives resets via `lifecycle/reset_episode` |
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

- [simulator/sim/README.md](simulator/sim/README.md) - physics simulator backends (`dummy`, `gazebo`, `isaac`).
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
