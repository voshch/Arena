# `edge_case` obstacle task mode

Runs the **effects** a scenario's `edge_case:` block declares, on top of the population the
scenario file already carries. The design is `new_plan_2.md` at the workspace root; the offline
generator that writes these scenarios is
[`arena_benchmark/promptgen`](../../../../../arena_benchmark/README.md#prompt-driven-generation).

Who is present is baked into `scenario.yaml` by the generator; until a case runs, the
population behaves exactly as the base scenario wrote it. Everything the case *changes* -
where people go, how they are tuned, where a formation stands - is an effect with an onset,
and what depends on the live robot is solved at reset:

| Effect | What it does | Runtime |
|---|---|---|
| `intercept` | one pedestrian placed so it meets the robot at a designed point and time on its **planned** route | [`geometry.py`](geometry.py), [`pathing.py`](pathing.py) |
| `rally` | send the agents in scope to a zone or point, once | [`waypoint_driver.py`](waypoint_driver.py) |
| `shuffle` / `scatter` | throw each agent a new point near where it started, every `period` s / once | same |
| `track_robot` | re-issue the robot's position as the goal | same |
| `hold` | pin the agents in scope, then release them to their own goal | same |
| `retune` | change the agents in scope to another `profile` and/or `speed` / `speed_scale`, once, in place (HumanSim's `UpdateAgents`: position, velocity and route continue; a backend that refuses counts as failures on the plan's outcome) | same, [`impl.py`](impl.py) `_retune_agents` |
| `objects:` | a hand-authored object timeline (not written by the generator) | [`object_driver.py`](object_driver.py) |

Every effect carries a `when`: `{at: s}`, `{robot_within: m, of: zone|[x,y]}`, or
`{route_fraction: f}` (fires when the robot comes within 1.5 m of that point of its route).

Requires `human:=arena`; registered from `ArenaHumanSimulator._register_task_modes`.

---

## Quick start

```bash
arena launch sim:=gazebo world:=hospital_1 robot:=jackal human:=arena \
      task.robots:=edge_case task.obstacles:=edge_case headless:=true \
      task.auto_reset:=true task.scenario.file:=normal__u_004
```

Write `task.scenario.file:=`, not `task.scenario:=` - the latter is silently overridden.
A scenario without a block runs the base population alone (the control arm). Records:

```bash
cat $ARENA_DATA_DIR/edge_case/cases.jsonl | python3 -m json.tool     # what was built
cat $ARENA_DATA_DIR/edge_case/scores.jsonl | python3 -m json.tool    # what happened
```

Newly generated scenario directories need `colcon build --packages-select
arena_simulation_setup --symlink-install` before the runtime sees them.

---

## The block

```yaml
edge_case:
  id: normal__u_004
  prompt_id: u_004
  steps: [A:busy, D:interaction_unbroken, C:approach]
  note: It is visiting hours and the place is packed. ...
  effects:
  - {type: intercept, label: C:approach, profile: cart, model: male_adult_construction_01,
     angle: 180.0, route_fraction: 0.3, speed: 0.9, after: continue, waypoint_mode: once}
  - {type: rally, label: A:evacuation, scope: [normal_1, normal_2], target: central_hallway, when: {at: 8.0}}
  - {type: scatter, label: D:formation_break, scope: [d_1, d_2, d_3], radius: 2.0, when: {robot_within: 3.0, of: [12.1, 4.0]}}
  - {type: retune, label: A:blackout, scope: [normal_1, normal_3], profile: ./types/adult__vision_range_low.yaml, speed_scale: 0.545, when: {at: 10.0}}
  formation: {centre: [12.1, 4.0], radius: 1.8, variant: circle, at: {at: 8.0}}   # where Step D's group forms
  decisions: {...}      # the generator's typed decisions, for review
  provenance: {...}     # prompt hash, model, response hashes, generator version
```

Grammar: [`effects.py`](effects.py). Reader: [`scenario_block.py`](scenario_block.py). Both are
strict - an unknown key, an unknown effect type or a scope naming an agent the scenario did
not place **aborts the episode with the reason recorded**, because a case that silently did
nothing is indistinguishable from a case with no effect.

### `intercept`

| Field | Default | Meaning |
|---|---|---|
| `profile` | adopt from the population | agent type: a builtin (`adult`, `elder`, …) or `./types/x.yaml` beside the scenario |
| `model` | adopt from the population | mesh; `arenian` when the population is empty |
| `angle` | 180 | travel direction relative to the robot's heading at the encounter: 180 head-on, 90 crossing from its right, 0 overtaking |
| `route_fraction` | 0.5 | where on the robot's planned route the meeting is designed |
| `speed` | the profile's nominal | m/s |
| `offset` | 0 | seconds the pedestrian arrives after the robot |
| `lead_out` | 3.0 | metres it continues past the encounter |
| `search` | 90 | max angular deviation when hunting free space; 0 aborts instead |
| `after` | `continue` | `stop_facing` (halt at the encounter for `duration`), `follow` (track the robot for `duration`), `veer` (turn off to `veer_to`), `stand` (placed *on* the route, stationary, released after `duration`) |
| `waypoint_mode` | `reverse` | `once` for exactly one designed crossing |

The geometry is solved at reset against the robot's route as planned through the occupancy
grid ([`pathing.py`](pathing.py)); requested and achieved angle/fraction are both recorded.
Injected agents are named `edge_0`, `edge_1`, … in block order.

### Scopes

`all` (every placed agent), `injected` (the `edge_*` agents), or an explicit list of names.

---

## Parameters

Namespace `task.edge_case.*`, declared in [`__init__.py`](__init__.py).

| Parameter | Default | Meaning |
|---|---|---|
| `robot_speed` | 1.0 | m/s the robot is assumed to drive when interceptions, holds and object events are timed. jackal covers hospital_1's 40 m leg in 33-45 s; at the old 0.5 the designed encounter came after the robot had arrived (2026-08-29). |
| `read_scenario_block` | `true` | Off runs the base population alone. |
| `prompt` | `""` | **The RViz input.** A natural-language situation; when set, the scenario is generated from it at reset by the same generator as `arena_bench promptgen` (model answers cached by content, so a repeated prompt is instant), written into the world's installed `scenarios/` as `<base>__rviz_<hash>` and run instead of `task.scenario.file`. The RViz task-mode panel shows this as a text field under `edge_case`. |
| `prompt_base` | `""` | Base the prompt is realised on; empty uses `task.scenario.file` (or its base when that is itself a prompt scenario). |
| `record_dir` | `""` | Where `cases.jsonl` / `scores.jsonl` go. Empty → `$ARENA_DATA_DIR/edge_case`. |
| `score`, `score_rate_hz`, `score_scope` | `true`, `10`, `auto` | Criticality panel per episode; `auto` scores the injected agents when the block injects any, everyone otherwise. |
| `objects`, `objects_rate_hz` | `""`, `5` | Object timeline override. |

There is no knob surface any more: a case is its scenario file. To sweep, list scenarios.

---

## Records

`cases.jsonl` (written at build): identity (`scenario`, `case_id`, `prompt_id`, `steps`), the
effects as designed, the resolved plans (owned agents, targets), the first intercept's
designed and achieved geometry (`spawn_pose`, `ped_goal`, `encounter_point`, `t_encounter`,
`approach_angle_requested/achieved`, …), `depart_at` (seconds the walker waits at its spawn because the full lead-in did not fit in free space - or would start inside the discs reserved around the robot's endpoints (`spawn_valid`) - it starts on the longest clear stretch, at least 3 m, and that much later), `robot_hold` (sim seconds `tm_robots:=edge_case` holds the robot at its start - informational: every plan's clock, and so every onset and interception time, starts when the robot is **under way**, `ROBOT_DEPART_M` = 0.3 m from where it stood at arming, so neither the hold nor nav2's first-plan delay shifts a case), `invalid_spawns`, and `status` (`ok` / `baseline` /
`aborted` with `reason`).

`scores.jsonl` (written when the episode ends, at the next reset or at teardown): the panel
from [`criticality.py`](criticality.py) plus `waypoint_outcomes` - per plan, when the gate
fired (`t_trigger`; `None` with a non-empty `trigger` means the robot never came near, i.e.
the effect **did not run**), when the first rewrite went out, and when a hold released. When the
block names the agents Step D owns (`owned: {D: [...]}`), the row also carries `formation`: whether
those members were seen, when every one of them was first within the block's `formation.radius`
of its `formation.centre` (`formed_at`; None if they never all arrived; the centre is realised into the map frame first), how long that lasted
(`held_s`), when the first one left (`broke_at`), and `arrived` - the most members within the radius at
once (a line five of six reached is not "never formed") - the Type D success criterion of
`new_plan_2.md` §10, positional because the cast is plain profiles walked into place by a rally.

Pair with [`tm_robots:=edge_case`](../../robots/edge_case/README.md): it assigns the goal
during reset (so the route exists when effects are built), traverses repeatedly, and reports
*blocked* as an outcome.

---

## Layout

| File | Role |
|---|---|
| [`impl.py`](impl.py) | `TM_EdgeCase` - population → effects → arm → record |
| [`effects.py`](effects.py) | the block grammar |
| [`scenario_block.py`](scenario_block.py) | reading the block from the active scenario |
| [`waypoints.py`](waypoints.py), [`waypoint_driver.py`](waypoint_driver.py) | resolved route-rewrite plans and the sim-time node that fires them |
| [`triggers.py`](triggers.py) | proximity gates |
| [`geometry.py`](geometry.py), [`pathing.py`](pathing.py) | the interception solver and the route planner |
| [`objects.py`](objects.py), [`object_driver.py`](object_driver.py), [`timelines/`](timelines/README.md) | object timelines |
| [`agent_types.py`](agent_types.py) | reading a profile's nominal speed |
| [`scoring.py`](scoring.py), [`criticality.py`](criticality.py) | the per-episode panel |
| [`provenance.py`](provenance.py) | `cases.jsonl` |

Tests: `tests/test_edge_case_*.py` (pure) and `tests/ros/test_edge_case_*.py`
(`test_edge_case_effects_apply.py` drives a whole reset with fakes; `edge_case_fixtures.py`
is the shared bare-mode builder).
