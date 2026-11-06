# USAR generator

The USAR generator builds the worlds and tasks of the search study: a manifest names the [`rooms` world](../world/README.md) configurations and the task axes. Every world is a building of rooms joined by doors from an entrance room. Each of its tasks hides one calling person a set number of corners from the robot's start, and some worlds are held out for testing.

```
manifest --tasks--> worlds x tasks --rooms generator--> worlds --placement--> search scenarios + index
```

## CLI

```bash
ros2 run arena_simulation_setup generate_usar configs/usar/mvp.yaml               # dry run
ros2 run arena_simulation_setup generate_usar configs/usar/mvp.yaml --write --out data/usar_mvp
```

| Option | Effect |
|---|---|
| `--write` | write the worlds and the index, a dry run without it |
| `--out DIR` | write every world to `DIR/<world>` and the index to `DIR/<name>.index.yaml`. Without it worlds go to the writable world root, and the index next to them |
| `--limit N` | only the first N tasks |

It prints every rejected task with its reason and exits 1 when no task was placed. Tasks on the same world configuration and seed share one world, which holds all their scenarios.

Run a task with the `search` robots mode. It sends no goal and only judges the robot on reaching the person, so a search driver has to move it:

```bash
export ARENA_WORLD_PATH=/opt/arena_ws/data/usar_mvp
arena launch world:=<world> task.robots:=search task.obstacles:=scenario task.scenario.file:=<task> acoustics:=arena robot.mobile.mapping:=slam robot.hearing:=srp robot.hearing.driver.enabled:=true
```

## Manifest

[`configs/usar/mvp.yaml`](../../../../configs/usar/mvp.yaml) is the MVP study: 10 worlds, 5 tasks each, 2 worlds held out.

| Key | Meaning |
|---|---|
| `name`, `seed` | index name; world seeds are `seed * 1000 + variant` |
| `world` | `rooms` configuration axes, each a value, a list or a stepped span, gridded |
| `world_variants` | world seeds per world configuration |
| `tasks_per_world` | tasks on each world |
| `test_worlds` | worlds held out from tuning and training, the first in world key order |
| `corners` | turns on the route from the robot's start to the person |
| `call_levels` | `weak`, `loud`, each with its `call_db` volume |
| `hiding` | `random`, or `adversarial`: the spot whose call points furthest from the route's first leg |
| `call_asset`, `call_period_s`, `call_on_s` | the looped sound the person calls with, on for `call_on_s` every `call_period_s` |
| `agent_type` | pedestrian profile of the person |

The task axes `corners`, `call_levels` and `hiding` are taken in turn across all tasks, corners fastest, so a small study still spans every corner count.

## Placement

The robot starts at the centre of the `entrance` zone, facing the building's centre. Candidate spots are drawn inside every room, clear of its walls and at least 3 m from the start. The route to a spot is the shortest chain of doors from [`shared/route.py`](../../shared/route.py), and its corners are the turns sharper than `TURN_RAD` along start, door middles and spot. A task is rejected when no spot has its corner count.

## Output

Each task writes `scenarios/<task>/` into its world:

| File | Content |
|---|---|
| `scenario.yaml` | the standing person `caller`, the looped sound `call` at its position, toggled by two `every:` timeline entries, and the robot's `start` without phases |
| `types/` | the person's standing behavior tree |
| `provenance.yaml` | the task, its split, the route, corners, doors, misdirection and call settings, and the launch keys that run it |

`<name>.index.yaml` lists the worlds of each split and every task with its status.
