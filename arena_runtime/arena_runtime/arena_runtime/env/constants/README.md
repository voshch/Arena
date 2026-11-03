# env constants

`Constants` enum definitions and the `Configuration(server)` factory that
maps them to live ROS parameters.

## `Constants`

[`__init__.py`](__init__.py)

| Name | Type | Values |
| --- | --- | --- |
| `ENV_NODE_NAMESPACE` | `str` | `"arena_env_node"`, the env node's ROS name and the last segment of an env's namespace |
| `DEFAULT_PEDESTRIAN_MODEL` | `str` | `"arenian"` |
| `HumanSimulator` | `Enum` | `dummy`, `none`, `hunav`, `arena`, `replay` |
| `AcousticsSimulator` | `Enum` | `none`, `arena` |

`SimSimulator` is defined in `arena_bringup/constants.py`, not on `Constants`.

The task-mode vocabulary (`Modes.TM_Robots`, `Modes.TM_Obstacles`,
`Modes.TM_Module`) is in
[`task_generator/modes.py`](../../../../../task_generator/task_generator/modes.py).
`TM_Obstacles.default()` and `TM_Robots.default()` return `RANDOM`.
`TM_Module.default()` returns an empty `set`.

## `Configuration(server)`

[`runtime.py`](runtime.py)

Factory function: takes a `ROSParamServer` and returns a `Config` class (not
an instance) whose nested class attributes are live `ROSParamT` descriptors.
Called once during node init; the returned class is stored as `node.conf`.

```python
Config = Configuration(server)
# access via:
Config.Arena.SIM.value        # reads/writes the ROS param
Config.General.RNG.stream("obstacles", "random")   # independent numpy Generator per key
```

### `Config.Arena`

| Attribute | ROS param | Default | Type |
| --- | --- | --- | --- |
| `SIM` | `sim` | `dummy` | `SimSimulator` |
| `HUMAN` | `human` | `dummy` | `Constants.HumanSimulator` |
| `ACOUSTICS` | `acoustics` | `none` | `Constants.AcousticsSimulator` |
| `WORLD` | `world` | *(required)* | `str` |
| `WORLD_LIGHTING` | `world.lighting` | `authored` | `str` |
| `WORLD_LEVELS` | `world.levels` | `auto` | `str` |
| `KG_DERIVE` | `kg.derive` | `auto` | `str` |

`HUMAN`'s `dummy` default is only the bare ROS-param fallback (offline/test contexts). `env.launch.py` always resolves `human` per sim first (`flatsim`/`gazebo`/`isaac` -> `arena`, `dummy` -> `dummy`).

### `Config.General`

| Attribute | ROS param | Default | Notes |
| --- | --- | --- | --- |
| `MAX_RESET_FAIL_TIMES` | `task.episode.reset.max_fails` | `10` | |
| `DESIRED_EPISODES` | `task.episode.count` | `-1` | parsed to `inf` when negative |

`RNG` is not a ROS param: it is an `EpisodeRng` re-rooted each reset on the
per-episode seed (derived from the `run_seed` param via blake2b). Call
`RNG.stream(*key)` for an independent, reproducible generator keyed by a stable
label, so draw order and concurrency cannot affect a given stream.

Every `task.episode.*` param is also a launch arg (empty = node default), listed with its description in `EPISODE_PARAMS`. The pre-`task.episode` names still work: `migrate_deprecated_params` copies each one found at startup onto its replacement and warns, per the `DEPRECATED_PARAMS` table.

### `Config.Obstacles`

| Attribute | ROS param | Default | Notes |
| --- | --- | --- | --- |
| `OBSTACLE_MAX_RADIUS` | `task.episode.spawn.obstacle_max_radius` | `15` | metres; `inf` when negative |

### `Config.Robot`

| Attribute | ROS param | Default | Notes |
| --- | --- | --- | --- |
| `GOAL_TOLERANCE_RADIUS` | `task.episode.goto_pose.tolerance.radius` | `1.0` | meters, default `GoToPhase.tolerance_radius` |
| `GOAL_TOLERANCE_ANGLE` | `task.episode.goto_pose.tolerance.angle` | 30 degrees (in radians) | default `GoToPhase.tolerance_angle` |
| `GOAL_HOLD_TIME` | `task.episode.goto_pose.hold_time` | `0.0` | sim seconds, default `GoToPhase.hold_time` |
| `GOAL_SIGNAL` | `task.episode.goto_pose.signal` | `''` | default `GoToPhase.signal`, empty = Arena judges arrival |
| `SPAWN_ROBOT_SAFE_DIST` | `task.episode.spawn.robot_clearance` | `0.25` | metres |
| `TIMEOUT` | `task.episode.timeout` | `-1` | parsed to `inf` when negative; sim seconds since episode reset, past which the episode ends FAILED with info `"timeout"` |
| `MOBILE_ADAPTER` | `robot.mobile_adapter` | `'nav2'` | default mobile-cap adapter kind, overridden per robot via scenario `mobile:` |
| `ARM_ADAPTER` | `robot.arm_adapter` | `'moveit'` | default arm-cap adapter kind, overridden per robot via scenario `arm:` |

Adapter-specific tunables (planners, RL agent, ...) live in `caps/<cap>.yaml` and can be overridden at launch time via `<cap>.<key>:=<val>` flags. Each `<cap>.<key>:=<val>` lands on the env node as ROS param `robot.<cap>.<key>` and is merged into the adapter's kwargs on top of the cap-file YAML.

### `Config.Human`

| Attribute | ROS param | Default | Notes |
| --- | --- | --- | --- |
| `ROBOTS_VISIBLE` | `human.robots.visible` | `True` | `False` = the human backend leaves the robots out of what it tells its pedestrians, honored by `ArenaHumanSimulator`, read once when the backend is built |

### Node-level runtime params (declared with `ParameterDescriptor`)

Declared directly on `EpisodeNode` at construction time, not via `Configuration`:

| ROS param | Default | Notes |
| --- | --- | --- |
| `task.episode.auto_reset` | `true` | `true` = standalone (node auto-advances); `false` = managed (external controller drives resets via `lifecycle/reset_episode`) |
| `episode_queue_depth` | `10` | Bound of the episode queue |
| `run_seed` | random uuid hex | Hex string for per-episode blake2b seed derivation |

`robot.train` is declared at the launch level and exposed on the env node's param store for robot adapters (`rosnav_rl`, `nav2`) to read, the node itself does not branch on it. For managed (external-controller-driven) resets pass `task.episode.auto_reset:=false` explicitly, there is no auto-derivation from `train_config`.

The `task.<axis>` selection params of the task runtime are not `Configuration`
entries. The runtime declares them with their defaults (for `task_generator` the
`TaskModeParams` in
[`tasks/layer.py`](../../../../../task_generator/task_generator/tasks/layer.py)),
and `env.launch.py` declares none: a `task.<axis>:=` launch arg passes through to
the node's param overrides. The node reaches the selection as `task_params.resolve({})`.
