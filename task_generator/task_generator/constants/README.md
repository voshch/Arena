# task_generator constants

`Constants` enum definitions and the `Configuration(server)` factory that
maps them to live ROS parameters.

## `Constants`

[`__init__.py:7`](__init__.py#L7)

| Name | Type | Values |
| --- | --- | --- |
| `DEFAULT_PEDESTRIAN_MODEL` | `str` | `"arenian"` |
| `TASK_GENERATOR_SERVER_NODE` | `Namespace` | `"task_generator_server"` |
| `SimSimulator` | `Enum` | `dummy`, `flatland`, `gazebo`, `unity`, `isaac` |
| `ArenaType` | `Enum` | `training`, `deployment` |
| `HumanSimulator` | `Enum` | `dummy`, `none`, `hunav`, `arena` |
| `TaskMode.TM_Obstacles` | `Enum` | `parametrized`, `random`, `scenario`, `environment`, `prompt` |
| `TaskMode.TM_Robots` | `Enum` | `guided`, `explore`, `random`, `scenario`, `demo`, `stationary` |
| `TaskMode.TM_Module` | `Enum` | `staged`, `dynamic_map`, `clear_forbidden_zones`, `rviz_ui` |

`TM_Obstacles.default()` returns `RANDOM`. `TM_Robots.default()` returns
`RANDOM`. `TM_Module.default()` returns an empty `set`.

## `Configuration(server)`

[`runtime.py:10`](runtime.py#L10)

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
| `SIM` | `sim` | `dummy` | `Constants.SimSimulator` |
| `HUMAN` | `human` | `dummy` | `Constants.HumanSimulator` |
| `WORLD` | `world` | *(required)* | `str` |

`HUMAN`'s `dummy` default is only the bare ROS-param fallback (offline/test contexts). `task_generator.launch.py` always resolves `human` per sim first (`gazebo`/`isaac` -> `arena`, `dummy` -> `dummy`).

### `Config.General`

| Attribute | ROS param | Default | Notes |
| --- | --- | --- | --- |
| `WAIT_FOR_SERVICE_TIMEOUT` | `timeout_wait_for_service` | `30` | seconds |
| `MAX_RESET_FAIL_TIMES` | `max_reset_fail_times` | `10` | |
| `DESIRED_EPISODES` | `episodes` | `-1` | parsed to `inf` when negative |

`RNG` is not a ROS param: it is an `EpisodeRng` re-rooted each reset on the
per-episode seed (derived from the `run_seed` param via blake2b). Call
`RNG.stream(*key)` for an independent, reproducible generator keyed by a stable
label, so draw order and concurrency cannot affect a given stream.

### `Config.Obstacles`

| Attribute | ROS param | Default | Notes |
| --- | --- | --- | --- |
| `OBSTACLE_MAX_RADIUS` | `obstacle_max_radius` | `15` | metres; `inf` when negative |

### `Config.Robot`

| Attribute | ROS param | Default | Notes |
| --- | --- | --- | --- |
| `GOAL_TOLERANCE_RADIUS` | `goal_tolerance_radius` | `1.0` | metres |
| `GOAL_TOLERANCE_ANGLE` | `goal_tolerance_angle` | 30 degrees (in radians) | |
| `SPAWN_ROBOT_SAFE_DIST` | `robot_safe_dist` | `0.25` | metres |
| `TIMEOUT` | `timeout` | `-1` | parsed to `inf` when negative |
| `RECORD_DATA_DIR` | `record_data_dir` | `''` | `None` when empty |
| `MOBILE_ADAPTER` | `robot.mobile_adapter` | `'nav2'` | default mobile-cap adapter kind, overridden per robot via scenario `mobile:` |
| `ARM_ADAPTER` | `robot.arm_adapter` | `'moveit'` | default arm-cap adapter kind, overridden per robot via scenario `arm:` |

Adapter-specific tunables (planners, RL agent, ...) live in `caps/<cap>.yaml` and can be overridden at launch time via `<cap>.<key>:=<val>` flags. Each `<cap>.<key>:=<val>` lands on the task-generator node as ROS param `robot.<cap>.<key>` and is merged into the adapter's kwargs on top of the cap-file YAML.

### Node-level runtime params (declared with `ParameterDescriptor`)

Declared directly on `TaskGenerator` at construction time, not via `Configuration`:

| ROS param | Default | Notes |
| --- | --- | --- |
| `auto_reset` | `true` | `true` = standalone (node auto-advances); `false` = managed (external controller drives resets via `lifecycle/reset_episode`) |
| `run_seed` | random uuid hex | Hex string for per-episode blake2b seed derivation |
| `episode_history_size` | `10` | Bounded history length for `state/episode` |

`train_mode` is declared at the launch level (`robot.train`) and exposed on the task_generator node's param store for robot adapters (`rosnav_rl`, `nav2`) to read, the node itself does not branch on it. For managed (external-controller-driven) resets pass `task.auto_reset:=false` explicitly, there is no auto-derivation from `train_config`.

### `Config.TaskMode`

| Attribute | ROS param | Default | Type |
| --- | --- | --- | --- |
| `TM_ROBOTS` | `tm_robots` | `random` | `Constants.TaskMode.TM_Robots` |
| `TM_OBSTACLES` | `tm_obstacles` | `random` | `Constants.TaskMode.TM_Obstacles` |
| `TM_MODULES` | `tm_modules` | `''` | `set[Constants.TaskMode.TM_Module]`; comma-separated string |
