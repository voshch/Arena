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

Takes a `ROSParamServer` and returns a `Config` whose `.Arena`, `.General`,
`.Obstacles`, `.Robot` and `.Human` are instances of the module-level
`ParamGroup` classes `ArenaParams`, `GeneralParams`, `ObstaclesParams`,
`RobotParams` and `HumanParams`. Called once during node init; the result is
stored as `node.conf`. A sixth group, `EpisodeParams` (`task.episode.auto_reset`,
`task.episode.fail_on_collision`, `run_seed`, `episode_queue_depth`), is
declared by `EpisodeNode` itself.

```python
conf = Configuration(server)
conf.Arena.SIM.value        # parsed live value of the `sim` parameter
conf.General.RNG.stream("obstacles", "random")   # independent numpy Generator per key
```

Each attribute is a `Param` declared with its description, default and
constraint (see `arena_rclpy_mixins.params`), so `episode_params`,
`episode_config` and `ros2 param describe` list every one with its text.
`WORLD` is the one required parameter. The enum-valued `SIM`, `HUMAN` and
`ACOUSTICS` derive their choices from the enum. A `-1` for the timeouts,
`task.episode.count` and the obstacle radius means unbounded (`inf` after
parsing).

`HUMAN`'s `dummy` default is only the bare ROS-param fallback (offline/test contexts). `env.launch.py` always resolves `human` per sim first (`flatsim`/`gazebo`/`isaac` -> `arena`, `dummy` -> `dummy`).

`General.RNG` is not a ROS param: it is an `EpisodeRng` re-rooted each reset on the
per-episode seed (derived from the `run_seed` param via blake2b). Call
`RNG.stream(*key)` for an independent, reproducible generator keyed by a stable
label, so draw order and concurrency cannot affect a given stream.

`EPISODE_PARAMS` is derived at import from the `task.episode.*` parameters of
`GeneralParams`, `ObstaclesParams`, `RobotParams` and `EpisodeParams`
(name to description). Every `task.episode.*` param is also a launch arg (empty = node default). The pre-`task.episode` names still work: `migrate_deprecated_params` copies each one found at startup onto its replacement and warns, per the `DEPRECATED_PARAMS` table.

Adapter-specific tunables (planners, RL agent, ...) live in `caps/<cap>.yaml` and can be overridden at launch time via `<cap>.<key>:=<val>` flags. Each `<cap>.<key>:=<val>` lands on the env node as ROS param `robot.<cap>.<key>` and is merged into the adapter's kwargs on top of the cap-file YAML. `robot.mobile_adapter`, `robot.arm_adapter`, `robot.lift_adapter` and `robot.gripper_adapter` (`RobotParams`) name the default adapter kind per cap, overridden per robot via the scenario or `adapters: {<cap>: ...}`.

`robot.train` is declared at the launch level and exposed on the env node's param store for robot adapters (`rosnav_rl`, `nav2`) to read, the node itself does not branch on it. For managed (external-controller-driven) resets pass `task.episode.auto_reset:=false` explicitly, there is no auto-derivation from `train_config`.

The `task.<axis>` selection params of the task runtime are not `Configuration`
entries. The runtime declares them with their defaults (for `task_generator` the
`TaskModeParams` in
[`tasks/layer.py`](../../../../../task_generator/task_generator/tasks/layer.py)),
and `env.launch.py` declares none: a `task.<axis>:=` launch arg passes through to
the node's param overrides. The node reaches the selection as `task_params.resolve({})`.
