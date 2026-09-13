# Object timelines

Things that appear and disappear while the robot is driving. Selected with
`task.edge_case.objects:=<name or path>`.

```bash
arena launch ... world:=arena_arena_002 task.edge_case.objects:=corridor_pinch
arena launch ... world:=arena_arena_002 task.edge_case.objects:=trap_behind
arena launch ... world:=hospital_1 task.scenario.file:=edge_e_objects   # hand-authored; the
                                                                        # scenario's own block
                                                                        # names ./objects.yaml
```

## Why not a scenario `static:` block

A scenario places objects at t=0, and **an object present from the start is just furniture** —
the robot plans around it and nothing is learned. Every failure mode worth measuring here
needs the world to change *after* the plan exists:

| Case | Question |
|---|---|
| a corridor narrows once the robot is inside it | does it replan, or freeze? |
| a blocked route later **clears** | does a robot that gave up ever retry? |
| an object closes in behind it | is backtracking attempted at all? |

The middle one is the reason `TM_Obstacles.retract` had to exist: before it, an object could
appear mid-episode but never disappear.

## Two coordinate conventions

`route_fraction`
: Position along the robot's *planned* route, by arc length. **Portable** — it names no
coordinate, so the same file is a valid case in every one of the twelve generated office
worlds. This is the only way `arena_arena_002` gets object cases at all: those worlds share
only `open_work_area` and `reception`, so a hand-authored coordinate would be right in one
world and silently wrong in eleven.

`pose`
: Absolute map-frame coordinates, for a hand-authored world where the point is a *specific*
doorway. See
[`hospital_1/scenarios/edge_e_objects/objects.yaml`](../../../../../../arena_simulation_setup/worlds/hospital_1/scenarios/edge_e_objects/objects.yaml).

*hospital_1 is where geometry is authored; generated worlds are where it is solved.*

## Format

```yaml
events:
  - action: spawn            # or `despawn`
    entity: blocking_cart    # timeline-local name; a despawn refers to its spawn by it
    model: Office/Object/SM_BookcaseA
    at: {route_fraction: 0.33}       # WHEN — or {t: 12.0} in sim seconds
    place: {route_fraction: 0.66, lateral_offset: 0.55}   # WHERE — or {pose: [x, y, yaw]}
    note: free text, carried into the case record

  - action: despawn
    entity: blocking_cart
    at: {t: 45.0}
```

`at` and `place` are **separate on purpose**. Firing when the robot *arrives* at a point puts
the object on top of it; leading the placement ahead of the firing point is what makes it
appear in front. Omitting `place` defaults it to the `at` fraction — which is a legitimate
case (an object dropped in the robot's path) and is exactly why the achieved
`reveal_distance_m` is always recorded rather than assumed.

`lateral_offset` shifts left of the direction of travel. Default yaw faces back down the
route, i.e. toward an oncoming robot.

Parsing and validation: [`../objects.py`](../objects.py). **Unknown keys are errors** — a
silently dropped `latteral_offset` is a case that measured something else. A despawn of
something never spawned is caught at parse time, because at runtime it is indistinguishable
from an object that vanished on its own.

## Shipped timelines

| Timeline | Events | What it does |
|---|---|---|
| [`corridor_pinch`](corridor_pinch.yaml) | 1 | A bookcase narrows the lane two thirds along, revealed a third of the way in |
| [`doorway_block_clear`](doorway_block_clear.yaml) | 2 | Blocks the route at 60 %, clears it at t=45 |
| [`trap_behind`](trap_behind.yaml) | 2 | Closes in behind the robot at the halfway point, clears at t=60 |

Names resolve against the installed share directory, then this directory; a path is taken as
given, and a scenario-relative name (`./objects.yaml`) resolves beside the active scenario —
the same convention `agent_type: ./talker.yaml` already uses.

## What is recorded, and where

The same split as `designed_ttc` versus `min_ttc_s`:

| File | Field | Contents |
|---|---|---|
| `cases.jsonl` | `object_timeline`, `object_events` | the timeline as **designed** — entity, action, model, fire time, pose, reveal distance |
| `scores.jsonl` | `object_events`, `objects_pending` | what actually **fired** — `t_fired` against `t_designed`, the `sim_path` each spawn returned, and per-event status |

`objects_pending` non-zero means the episode ended before the timeline did.

## Two things that will look like bugs

**A mid-episode object is not in nav2's static layer.** It exists only in the map the lidar
builds, so nothing happens until it is *seen*. `reveal_distance_m` is what says how far away
that was — an object revealed beyond sensor range is a case that measured nothing, and one
revealed at 0.5 m is a different experiment from one revealed at 5 m. Neither is wrong; both
have to be known.

**A despawned object may leave costmap residue** until the clearing raycast catches up. "The
robot still avoids a cart that is no longer there" is a real outcome — arguably the more
interesting one. Report it; do not tune it away.

## Failure discipline

A **spawn** that fails aborts the case: the object never appeared, so the case did not run,
and an unperturbed episode reported as a result is the worst thing this pipeline can produce.
A **despawn** that fails does not: the case ran, and an object left behind is recorded as an
anomaly on it. A despawn whose spawn failed is `skipped`, not `failed` — nothing is there,
which is what it wanted.

The driver ([`../object_driver.py`](../object_driver.py)) runs on **sim time**, so a paused
episode does not race through its timeline, and it disables itself rather than taking an
episode down with it.
