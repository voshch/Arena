# arena_fleet

Fleet dispatcher for robots that share doors with pedestrians. It runs as its own node next to an Arena env whose robots
are parked by a scenario (`task.robots:=scenario`, robots without phases), and drives them only through existing interfaces:

- reads robot odometry (`<ns>/arena_env_node/<robot>/odom`), pedestrians (`<ns>/arena_peds`) and door leaves (`/flatsim/poses`);
- assigns goals and stops robots through the env's `runtime/run_task` action;
- publishes one global plan per robot on `<ns>/arena_env_node/<robot>/plan` (local planner drlvo, started with
  `robot.mobile.global_planner:=none`), or sends it as a Nav2 `follow_path` goal (`local:=dwb`).

## What it decides

1. **Goals**: `n` goals are drawn from the seed in the goal zone and assigned to the robots by minimum total cost `J`
   (Hungarian). Fleets of 2, 4 and 8 with one seed share their first goals.
2. **Door and plan**: each robot gets one plan to its goal through the door with the lowest `J`, re-planned once a second.
3. **Rules**, switched on per profile in [configs/rules.yaml](configs/rules.yaml):

| Rule | What it does |
| --- | --- |
| `capacity` | at most `k` robots inside a door; a robot holds its door until it is through |
| `queue` | robots wait for a door one at a time, at zig-zag slots beside the door axis |
| `yield_people` | no entry while people are in the door; stop or step aside before a person predicted on the plan |
| `plan_around` | the plan bends around cells the people map marks occupied |
| `slot_by_people` | queue slots on the side of the door the people leave free |
| `reassign_door` | switch door while moving when another is clearly cheaper |
| `reassign_goal` | two robots swap goals when that lowers their total `J` |

Profiles `base`, `capacity`, `queue`, `queue_yield` and `full` add the rules one by one for ablations; `hata` is the
HATA baseline (arXiv 2508.19731).

## Cost

`J = w_time * T + w_encounter * H + w_facility * F + w_wait * W`, weights and terms in [configs/cost.yaml](configs/cost.yaml):
`T` plan time at `v_nom`, `H` people the plan is predicted to meet, `F` door use (closed door, other robots routed there,
people at the door on arrival), `W` time standing at the door behind other robots. Every decision is logged with all terms.

## What the dispatcher knows about people (`info`)

| `info` | People map built from |
| --- | --- |
| `none` | nothing, people cost nothing |
| `oracle` | every pedestrian, constant-velocity forecast stopped at walls |
| `perceived` | only pedestrians a robot sees (range, field of view, line of sight), kept for `memory_s` |
| `flow` | live state of everyone blended into a map of dynamics (`mod`), thresholded |
| `mod` | the map of dynamics alone (HATA baseline) |

Every rule reads only this map, so the levels differ only in how it is made. The map of dynamics (`mod`) is an `.npz`
with `rho[t, y, x]` (pedestrians per m^2 per time window and 0.25 m cell, from robot-free runs) and `x_edges`,
`y_edges`, `t_edges`.

## Run the doorway example

Terminal 1, the env with the example scenario (4 parked robots, pedestrian waves through `main_door`):

```bash
arena launch sim:=flatsim world:=mechanisms_doorway 'robot:=jackal[4]' \
  robot.mobile:=drl robot.mobile.planner:=drlvo robot.mobile.global_planner:=none \
  task.robots:=scenario task.obstacles:=scenario \
  task.scenario.file:=$(ros2 pkg prefix arena_fleet)/share/arena_fleet/examples/doorway/scenario run_seed:=fb-6 headless:=true
```

Terminal 2, the dispatcher:

```bash
EX=$(ros2 pkg prefix arena_fleet)/share/arena_fleet/examples/doorway
ros2 run arena_fleet dispatcher --ros-args \
  -p world:=$ARENA_DIR/arena_simulation_setup/worlds/mechanisms_doorway/world.yaml \
  -p resources:=$EX/resources.yaml -p scenario:=$EX/scenario/scenario.yaml \
  -p profile:=full -p info:=oracle -p n:=4 -p seed:=6 -p out:=/tmp/arena_fleet_run
```

The node ends the episode `tail_s` after the last robot arrives, or after `duration_s`. With Nav2 DWB instead of drlvo,
launch with `robot.mobile.local_planner:=dwb` and pass `-p local:=dwb`.

## Output (`out`)

| File | Content |
| --- | --- |
| `events.jsonl` | every decision: goal assignment with the cost matrix, door options with all terms, holds, queue, yields, re-assignments, trips, end with makespan |
| `state.jsonl` | 4 Hz snapshot: phase, door, goal, queue slot, wait reason per robot, doors, known people |
| `samples.jsonl` | every 0.5 s per robot: input features ([arena_fleet/features.py](arena_fleet/features.py)) and the decision |
| `heat.jsonl` | the people map now and 5 s ahead |
| `dispatch.json` | profile, rules, cost, goals and seed of the run |

## Tests

```bash
python3 -m pytest arena_fleet/test
```

They drive the dispatcher on the `mechanisms_doorway` world with the example files; robots walk their plan at 0.4 m/s.
