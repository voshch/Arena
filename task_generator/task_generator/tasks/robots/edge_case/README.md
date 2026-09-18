# `edge_case` robot task mode

Routed traversals with obstruction as a first-class outcome. Built to pair with
[`tm_obstacles:=edge_case`](../../obstacles/edge_case/README.md).

```bash
arena launch sim:=dummy world:=arena_arena_002 robot:=jackal human:=arena \
      tm_robots:=edge_case tm_obstacles:=edge_case headless:=true
```

Unlike its obstacle-side namesake this mode has **no HumanSim dependency**, so it registers
at import and works under any `human:=` backend. It is useful on its own wherever you want
repeated traversals or blocked-detection, not only for edge cases.

---

## Why not `explore`

`tm_robots:=explore` actively fights the edge-case design in three ways:

| | `explore` | `edge_case` |
|---|---|---|
| Goal set | lazily, in `done` | during `reset()`, before obstacles are generated |
| Episode end | never (`done` is always `False`) | on completion, obstruction, or timeout |
| Route | random | the scenario's declared route, deterministic |

The first matters most. `Task._reset_episode` awaits `tm_robots.reset()` **before**
`tm_obstacles.reset()`, which is what lets the obstacle mode pick the agent whose path most
nearly meets the robot's. Under `explore` the goal is still unset at that moment, so the
obstacle mode records `robot_goal_degraded: true` and falls back to start-pose-only
selection — the ego-relative half of the design is off.

`tm_robots:=scenario` fixes the first two and is a reasonable fallback. What it does not do
is repeat the crossing or treat obstruction as an outcome.

---

## What it does

**Route** — inherited from the robots scenario mode: the `robots:` block of whichever
scenario `task.scenario.file` names. Deterministic, and authored next to the crowd it has
to cross. Zone names work (`start: reception`), as do coordinates.

**Traversals** — the single start→goal can be expanded into `2N-1` alternating legs (out,
back, out, …) so the robot passes through the population repeatedly within one episode.
The default is 1 since 2026-08-29: every prompt case is designed against the single
start→goal leg (onsets, interceptions, formations along that route), a return leg only
replays the population with the design behind it, and the recordings want the episode to
end when the robot arrives. `traversals: 1` is exactly what the scenario mode does.

**Obstruction watchdog** — tracks each robot's closest-ever approach to its current leg's
goal. If that high-water mark does not improve by `blocked_distance` for `blocked_timeout`
seconds, the robot counts as obstructed and the episode ends as FAILED with a reason.

Distance-to-goal rather than raw speed, deliberately: a robot legitimately pauses, rotates
in place, or backs out of a doorway, and none of that is obstruction. Failing to get any
closer for a sustained window is.

---

## Outcomes

Three terminal states, each mapping to a real episode result:

| Situation | Outcome | Reason |
|---|---|---|
| every traversal completed | `SUCCESS` | — |
| obstructed past the timeout | `FAILED` | `robot 'jackal' blocked: no progress toward (…) for 15s, still 2.31m away` |
| overall timeout, work left | `FAILED` | `timed out after 120s with traversals incomplete` |

The base `TM_Robots.done` returns `True` on timeout without aborting, which the node reports
as SUCCESS. For edge-case work a robot that ran out of clock without finishing is a
failure, so this mode distinguishes the two.

Because episodes now end on their own, an external driver can wait for the episode outcome —
it waits for the task mode's own outcome instead of a wall-clock window.

---

## Parameters

Namespace `task.edge_case_robot.*`.

| Parameter | Default | Meaning |
|---|---|---|
| `traversals` | 1 | Round trips through the population. N traversals is `2N-1` legs; 1 = the single start→goal leg. |
| `retries` | 1 | Attempts per leg. nav2 gives up on a leg it cannot make progress on and the phase policy is `continue`, so with 1 a blocked leg is skipped; with 3 the same goal is submitted three times before the route moves on - what a recording wants. |
| `blocked_distance` | 0.25 | Metres of improvement that count as progress. Jitter below this is not progress. |
| `blocked_timeout` | 15.0 | Seconds without progress before the robot counts as obstructed. |
| `hold` | 0 | Sim seconds the robot stands at its start before the first leg (`done` reports neither blocked nor finished meanwhile). Rarely needed since 2026-08-29: the obstacles mode's case clock starts when the robot is under way (0.3 m from its start), so onsets count from the drive, not the reset; recordings run with 0. |
| `accept_within` | 0.6 | Metres from a leg's goal at which a robot that has stopped getting closer counts as arrived (`near_goal_accepted`). nav2 refuses to plan into a goal cell inside the robot's inflation - a goal pinned 0.35 m from a desk left mpo700 "blocked" 0.23 m short for a whole recording - and the robot is as good as there. 0 = off. |
| `accept_after` | 5 | Seconds without `blocked_distance` of progress inside `accept_within` before the leg is accepted. |
| `on_blocked` | `abort` | `abort` ends the episode as FAILED; `continue` logs and carries on. |

The **scenario file is read from `task.scenario.file`**, not from this namespace — so the
route and the crowd always come from the same scenario. See the note in `impl.__init__`:
this is why `TM_Scenario.__init__` is deliberately skipped.

---

## Verifying the pairing

The acceptance criterion is already instrumented by the obstacle mode. Run the pair and
read its provenance:

```bash
python3 -c "
import json; d=[json.loads(l) for l in open('\$ARENA_DATA_DIR/edge_case/cases.jsonl')][-1]
print(d['robot_goal_degraded'], d['selector'], d['target_agent'])"
```

`robot_goal_degraded: false` is the direct proof the ordering problem is fixed — it is
`true` under `explore`.

---

## Gotchas

**The route endpoints are `forbid`den.** Inherited from the scenario mode: a disc of
`robot.safe_distance` around the start and every goal is excluded from obstacle placement,
so the crowd cannot spawn on top of the robot. On a small zone this is significant — it is
what pushed the generated worlds' `default` scenario to stop spawning anyone in
`reception`, which is only 20 m² and holds up to 8 pieces of furniture.

**Nearest-approach follows the robot.** Because the route starts in reception, the obstacle
mode will prefer agents near reception. If you want the perturbation to land on a specific
group, put the group where the route actually goes — or pin it with
`task.edge_case.select <agent name>`.

**`traversals` interacts with `Robot.TIMEOUT`.** Three traversals of a long route may not
fit in the default timeout, which surfaces as a `timed out` failure rather than a blocked
one. Raise the timeout or lower `traversals` if you see that.

---

## Layout

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | registration + `declare_schema` |
| [`impl.py`](impl.py) | `TM_EdgeCase(TM_Scenario)` — route expansion, watchdog, outcomes |

Tests: `tests/test_edge_case_robot.py` (pure: phase expansion, watchdog) and
`tests/ros/test_edge_case_robot_registry.py` (registration, schema, param binding).
