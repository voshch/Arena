# HuNav configs

This directory contains two distinct things used by the HuNav human simulator:

1. `default.yaml`: default agent template loaded at module init.
2. `behavior_trees/`: shared BTCPP v4 XML behavior tree library.

## `default.yaml`: default agent template

`HunavDynamicObstacle._default` is populated by
`_load_config()` in
[task_generator/simulators/human/hunav/__init__.py](../../../task_generator/task_generator/simulators/human/hunav/__init__.py)
(`_load_config`, `HunavDynamicObstacle.parse`), which reads
`<arena_simulation_setup share>/configs/hunav/default.yaml` at module import
time.

The file sets the default field values for every `HunavDynamicObstacle` that
does not override them explicitly. `HunavDynamicObstacle.parse` reads these
top-level keys: `id`, `group_id`, `skin`, `type` (agent kind, see below),
`max_vel` (-> `desired_velocity`), `radius`, `goal_radius`, `cyclic_goals`,
`behavior_tree`, `init_pose: {x, y, z, h}`, `goals` (see Goals below), and
`behavior:` (see Behavior below):

```yaml
id: 1
group_id: -1
skin: 0
max_vel: 0.8
radius: 0.3
goal_radius: 2.0
cyclic_goals: true
init_pose:
  x: 0.0
  y: 0.0
  z: 1.250        # spawn height in Gazebo
  h: 0.0
behavior:
  type: 1         # hunav_msgs::msg::BEH_REGULAR
  configuration: 0
  goal_force_factor: 5.0
  obstacle_force_factor: 20.0
  social_force_factor: 20.0
  other_force_factor: 20.0
```

`HunavDynamicObstacle.Behavior._default` is then set from `_default.behavior`.

### `type` (Agent.msg)

`hunav_msgs/Agent.type`: `1` = `PERSON`, `2` = `ROBOT`, `3` = `OTHER`. Defaults to `1` if omitted or not an int.

### `behavior:` block

Maps to `hunav_msgs/AgentBehavior`. Accepted keys: `type`, `state`, `configuration`, `duration`, `once` (bool), `vel`, `dist`, `social_force_factor`, `goal_force_factor`, `obstacle_force_factor`, `other_force_factor`. Enum values (from `hunav_msgs/AgentBehavior.msg`, upstream in `deps/hunav`):

| `type` | Behavior | | `state` | | `configuration` |
|---|---|---|---|---|---|
| 1 | `BEH_REGULAR` | | 0 | `BEH_NO_ACTIVE` | 0 = `BEH_CONF_DEFAULT` |
| 2 | `BEH_IMPASSIVE` | | 1 | `BEH_ACTIVE_1` | 1 = `BEH_CONF_CUSTOM` |
| 3 | `BEH_SURPRISED` | | 2 | `BEH_ACTIVE_2` | 2 = `BEH_CONF_RANDOM_NORMAL` |
| 4 | `BEH_SCARED` | | | | 3 = `BEH_CONF_RANDOM_UNIFORM` |
| 5 | `BEH_CURIOUS` | | | | |
| 6 | `BEH_THREATENING` | | | | |

All keys are forwarded as authored. A `dynamic:` entry without a
`behavior:` block gets `goal_force_factor: 5.0` and every other factor `0.0`.

## Per-instance overrides on a scenario `dynamic:` entry

For HuNav (`human:=hunav`), `HunavDynamicObstacle.from_dynamic_obstacle` builds
one agent per `dynamic:` entry from the entry's `extra:` map (scenario-level
`pose:`/`model:`/`waypoints:` still supply the base pose and mesh). Every
`default.yaml` key above is a valid `extra:` key and overrides that agent's
default. A few keys have HuNav-specific meaning:

| `extra:` key | Meaning |
|---|---|
| `position: {x, y, z, h}` | Overrides `init_pose`. Falls back to the entry's own `pose:` for `x`/`y` if omitted. |
| `velocity` | Raw `Twist`-shaped seed velocity, rarely set. |
| `desired_velocity` | Overrides `max_vel` for this instance. |
| `linear_vel` / `angular_vel` | Initial linear/angular velocity. |
| `goals` | List of waypoint-name strings. Each name is looked up as a sibling key in the same `extra:` map, e.g. `extra: {goals: [wp1, wp2], wp1: {x: 1.0, y: 2.0}, wp2: {x: 3.0, y: 0.0}}`. Overrides the entry's scenario-level `waypoints:` list when present. |
| `behavior_tree` | Path (`./name.xml`, resolved relative to the scenario directory) or a name from `behavior_trees/` below. Falls back to `default.yaml`'s `behavior_tree` (itself defaulting to `default.xml`) if omitted. |
| `behavior` | See `behavior:` block above. |
| `cyclic_goals`, `goal_radius`, `id`, `type`, `skin`, `group_id`, `radius` | Same meaning as in `default.yaml`. |

See [task_generator human README](../../../task_generator/task_generator/simulators/human/README.md) for where `dynamic:` entries and their `extra:` map come from in `scenario.yaml`.

## `behavior_trees/`: BT library

Shared BTCPP v4 XML files defining reusable pedestrian behaviors. These files
are referenced by name from scenario `behavior_tree:` fields and from
per-agent `hunav_<N>_behavior_tree.xml` files in scenario directories.

| File | Behavior |
|---|---|
| `BTRegularNav.xml` | Standard goal-directed navigation |
| `BTCuriousNav.xml` | Curious pedestrian, deviates toward the robot |
| `BTScaredNav.xml` | Scared pedestrian, avoids the robot |
| `BTSurprisedNav.xml` | Surprised pedestrian, brief stop then reroutes |
| `BTThreateningNav.xml` | Threatening pedestrian, approaches the robot |
| `default.xml` | Alias used as the fallback in `HunavDynamicObstacle._default.behavior_tree` |

Per-scenario BT files (`hunav_<N>_behavior_tree.xml` in a scenario directory)
are instance-specific overrides that can reference or extend these shared trees.
The `behavior_tree` field in `scenario.yaml` accepts a path relative to the
scenario directory (e.g. `./hunav_1_behavior_tree.xml`) or a name from this
library (e.g. `BTRegularNav.xml`).
