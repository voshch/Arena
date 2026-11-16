# arena_fleet

Fleet dispatcher for robots that share doors with pedestrians. It runs as its own node next to an Arena env whose robots
are parked by a scenario (`task.robots:=scenario`, robots without phases), and drives them only through existing interfaces:

- finds the env `env_id` in the runtime's registry (`/arena/state/envs`), takes the env's map offset from its
  `reference` there and drives every robot its fleet (`<env node>/state/robots`) lists;
- loads the world the env runs (`world` of its semantics) and manages what the `resources` file names for it: the world
  doors (their level is the level that holds them all, each door's area the smallest world zone holding its span) and
  the `goal_zones`;
- reads each robot's odometry (`<robot ns>/odom`), the env's pedestrians (`<env ns>/arena_peds`) and the state of the
  world doors from the env's semantics (`<env node>/state/semantics`); a door the env never reports counts as open;
- assigns goals and stops robots through the env's `<env node>/runtime/run_task` action;
- publishes one global plan per robot on `<robot ns>/plan` (local planner drlvo, started with
  `robot.mobile.global_planner:=none`), or sends it as a Nav2 `<robot ns>/follow_path` goal (`local:=dwb`).

## What it decides

1. **Goals**: `n` goals are drawn from the seed in the world zones the `goal_zones` of the resources file name, and assigned to the robots by minimum total cost `J`
   (Hungarian). A smaller fleet with the same seed takes the first goals of a larger one.
2. **Door and plan**: each robot gets one plan to its goal through the door with the lowest `J`, re-planned every `episode.replan_period_s`.
3. **Rules**, each profile in [configs/rules.yaml](configs/rules.yaml) lists the ones it switches on:

| Rule | What it does |
| --- | --- |
| `capacity` | at most `k` robots inside a door; a robot holds its door until it is through |
| `queue` | robots wait for a door one at a time, at zig-zag slots beside the door axis, or where they stand when no slots fit |
| `yield_people` | no entry while people are in the door; stop or step aside before a person predicted on the plan |
| `plan_around` | the plan bends around cells the people map marks occupied |
| `slot_by_people` | queue slots on the side of the door the people leave free |
| `reassign_door` | switch door while moving when another is clearly cheaper |
| `reassign_goal` | two robots swap goals when that lowers their total `J` |

Profiles `base`, `capacity`, `queue`, `queue_yield` and `full` add the rules one by one for ablations; `hata` is the
HATA baseline (arXiv 2508.19731).

## Cost

`J = w_time * T + w_encounter * H + w_facility * F + w_wait * W`, weights and terms in [configs/cost.yaml](configs/cost.yaml):
`T` plan time at `v_nom` (`time.speed_fraction` of the slowest fleet robot's top speed), `H` people the plan is predicted to meet, `F` door use (closed door, other robots routed there,
people at the door on arrival), `W` time standing at the door behind other robots. Every decision is logged with all terms.
[configs/hata_cost.yaml](configs/hata_cost.yaml) names `cost.yaml` as its `base` and sets only the HATA bid and its
planning and flow values.

## What the dispatcher knows about people (`info`)

| `info` | People map built from |
| --- | --- |
| `none` | nothing, people cost nothing |
| `oracle` | every pedestrian, constant-velocity forecast stopped at walls |
| `perceived` | only pedestrians within a robot's laser range and field of view and in its line of sight, kept for `memory_s` |
| `flow` | live state of everyone blended into a map of dynamics (`mod`), thresholded |
| `mod` | the map of dynamics alone (HATA baseline) |
| `dmod` | known people forecast onto the walking directions of a directional map of dynamics (`rho_dir`, turning with `flow.advect_tau_s`), plus the people that map expects on the plan |

Every rule reads only this map, so the levels differ only in how it is made. The map of dynamics (`mod`) is an `.npz`
with `rho[t, y, x]` (pedestrians per m^2 per time window and cell, from robot-free runs) and `x_edges`, `y_edges`,
`t_edges`, and for `dmod` also `rho_dir[t, y, x, bin]`, the density split by walking direction. The people map spans the
bounding box of the world's walls in cells of `people_view.grid.cell_m`. The map of dynamics must start at the same
corner, have the same cell size and cover that box, otherwise the dispatcher refuses it.

## Robot

The dispatcher takes the robot from the fleet's models (`arena_robots`, `caps/mobile.yaml`): the body is the largest
`radius` plus `robot.margin_m`, `v_nom` is `time.speed_fraction` of the slowest top speed (`velocity_limits.linear`),
and perception is the shortest and narrowest `laser` range and field of view.

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
ros2 run arena_fleet dispatcher --ros-args -p resources:=$EX/resources.yaml \
  -p profile:=full -p info:=oracle -p seed:=6 -p out:=/tmp/arena_fleet_run
```

The node ends the episode `tail_s` after the last robot arrives, or after `duration_s`. With Nav2 DWB instead of drlvo,
launch with `robot.mobile:=nav2 robot.mobile.local_planner:=dwb` in place of the three `robot.mobile` arguments and
pass `-p local:=dwb`.

## Output (`out`)

Every run writes these files afresh.

| File | Content |
| --- | --- |
| `events.jsonl` | every decision: goal assignment with the cost matrix, door options with all terms, holds, queue, yields, re-assignments, trips, end with makespan |
| `state.jsonl` | a snapshot per tick (`tick_s`): phase, door, goal, queue slot, wait reason per robot, doors, known people |
| `samples.jsonl` | every `episode.sample_period_s` per robot: input features ([arena_fleet/features.py](arena_fleet/features.py)) and the decision |
| `heat.jsonl` | the people map `people_view.heat_s` seconds ahead |
| `dispatch.json` | profile, rules, cost, goals and seed of the run |

## Tests

```bash
python3 -m pytest arena_fleet/test
```

They drive the dispatcher on the `mechanisms_doorway` world with the example files, robots parked at the example
scenario's starts; robots walk their plan at `v_nom`.
