# EgoBench generator

The EgoBench generator builds the encounter dataset: a manifest names the world types, the encounter types and the value ranges, every sample of it becomes one world from the [`corridor` or `open` world generator](../world/README.md) with one robot and one pedestrian scenario beside it, the pedestrian on a fixed or a reactive route, and a sample whose encounter does not fit its world is rejected with the reason.

```
manifest --sample--> samples --world generator--> worlds --scenario generator--> scenarios + index
```

## CLI

```bash
ros2 run arena_simulation_setup generate_egobench configs/egobench/mvp.yaml               # dry run
ros2 run arena_simulation_setup generate_egobench configs/egobench/mvp.yaml --write --out data/egobench_mvp
```

| Option | Effect |
|---|---|
| `--write` | write the worlds and the index, a dry run without it |
| `--out DIR` | write every world to `DIR/<world>` and the index to `DIR/<name>.index.yaml`. Without it worlds go to the writable world root, where `world:=<world>` finds them, and the index next to them |
| `--limit N` | only the first N samples |

It prints every rejected sample with its reason and exits 1 when no sample was placed. Samples on the same world configuration and seed share one world, which then holds all their scenarios. The `egobench` world generator does the same for one sample, so `generate_world`, the World Generator panel and the MCP `worldgen` tools reach it too.

## Manifest

Every axis is a value, a list of values or a span `{min, max, step}`. A grid takes the product of all axes and needs a `step` on every span, `random` draws `samples` samples and a span without `step` uniformly.

| Key | Meaning |
|---|---|
| `name` | dataset name, names the index |
| `seed` | seeds the random draws and the world seeds |
| `sampling` | `grid` (default) or `random` |
| `samples` | draws for `random`, a cap on the grid otherwise, 0 = the whole grid |
| `world_variants` | worlds per world configuration, each from its own seed |
| `worlds` | per world type (`corridor`, `open`) its generator's configuration fields as axes, checked against the generator |
| `encounters` | `oncoming`, `crossing_left`, `crossing_right`, `overtake` (the robot passes), `overtaken` (the pedestrian passes), `follow` (the robot walks behind), `lead` (the pedestrian walks behind), `standing` |
| `pedestrians` | `fixed` (default): the pedestrian ignores the robot and starts on the clock, so it walks the same route under every planner. `reactive`: it avoids the robot under the social force model and starts as the robot nears |
| `lateral_offset` | m. Along a run the distance between the robot's and the pedestrian's lane, `+` puts the pedestrian on the robot's left. In a crossing how far ahead of the robot the pedestrian crosses its route |
| `robot_speed`, `ped_speed` | m/s, the speeds the release is timed for |
| `agent_type` | pedestrian profile: `adult`, `child`, `elder`, `hurried`, `distracted`, `nurse`, `doctor` |

```yaml
name: egobench_mvp
world_variants: 2
worlds:
  corridor: {corridor_width: 3.0, branches: 10}
  open: {width: 20.0, height: 20.0, obstacles: 0}
encounters: [oncoming, crossing_left, crossing_right]
pedestrians: [fixed, reactive]
lateral_offset: {min: 0.6, max: 1.4, step: 0.2}
```

## Placement

The robot drives one straight leg, and the encounter is designed at its middle.

- **Corridor** ([`scenario_corridor.py`](scenario_corridor.py)): the legs are the straight runs of the corridor tree, longest first and in both directions, cut short at both ends by the robot's radius. The two lanes lie symmetric about the centreline, so a corridor holds a lateral offset of at most its width less 0.9 m. A crossing comes out of a branch square to the run, at a T or a cross junction; at a T the pedestrian turns towards the robot after crossing.
- **Open** ([`scenario_open.py`](scenario_open.py)): the legs are chords through the free space in eight headings, longest and most central first, kept clear of the obstacles.

Both check that the robot's leg and the pedestrian's route stay inside the free space with the bodies' radii ([`encounter.py`](encounter.py)). The pedestrian is a HumanSim behavior tree that stands until the timeline raises `go` on it. The release lies as much robot travel before the encounter as the pedestrian needs for its 5 m lead-in. A followed or leading pedestrian walks from the start, a standing one is never released. Both kinds walk the same route:

| | `fixed` | `reactive` |
|---|---|---|
| release | `at` the time a robot at the nominal speed reaches it | `when` the robot comes within 1.5 m of it |
| local planner | `straight`: constant speed, no avoidance of the robot or walls | the profile's own, avoiding the robot |
| differs between planners | when the robot meets it, not the route | the route and the realized offset |

## Outputs

Each placed sample writes `scenarios/<scenario>/` into its world:

| File | Content |
|---|---|
| `scenario.yaml` | `arena-scenario/2`: the pedestrian, the robot's start and goal and the release, carrying `GENERATED_MARKER` |
| `types/*.yaml` | the pedestrian's behavior tree |
| `provenance.yaml` | the sample and the designed geometry: robot start and goal, pedestrian route, encounter point, release arc and its nominal time |

The world binds its episode to the `scenario` task modes on both axes. To run one scenario:

```bash
arena launch world:=<world> task.robots:=scenario task.obstacles:=scenario task.scenario.file:=<scenario> human:=arena
```

The index lists every sample with its world, scenario, `ok` or `rejected` and the reason.
