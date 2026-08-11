---
name: project-drl-sim-vs-offline-gap
description: OPEN - DRL planners drive the s-bend offline in under 60s but time out in sim at 180s; five explanations refuted
metadata: 
  node_type: memory
  type: project
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-07T18:15:54.175Z
---

As of 2026-08-07, on world `sbend_corridor` (robot auto -> jackal, `nav2/theta_star` global
plan, carrot shim): 9 of 14 registry DRL planners complete the real benchmark route in a
closed-loop offline rollout in 56-59 s (cadrl, crowdsurfer, scope, her-drl, height, kdma,
rlrvo, navrep, navistar), yet in the benchmark only her-drl solves (25.3 s) and the rest time
out against a 180 s budget. See [[project-drl-planner-offline-probe]] for the harness.

height is the sharpest case: offline it reaches in 59.5 s; in sim it locks at v=0.5, w=+1.0
and drives a fixed 0.5 m circle, 87 m travelled for 2.7 m net displacement.

Explanations tested and REFUTED - do not re-raise without new evidence:

- **Holonomic->diff-drive projection.** `project_holonomic_to_diff_drive` uses gain 1/dt,
  which looks undamped but is deadbeat: it nulls heading error in one step. All 7
  omnidirectional planners reach the goal through it with w at the rail ~2% of the time.
- **Collision monitor halting the robot.** Across 5 planners' bags, zero cmd_vel samples are
  exactly zero and StopPolygon never fires. Only crowdsurfer is meaningfully slowed.
- **Carrot collapsing at corners.** Measured the real `lookahead_clear_on_path` against a
  costmap built from the world's map.png with nav2 inflation: the carrot holds at 1.28-1.80 m
  along the whole route, minimum 1.28 m at each corner. height only fails below 0.4 m.
- **attngraph missing `ob_rms`.** The mirrored checkpoint is a bare OrderedDict with no
  normaliser, and the Arena fork drops VecNormalize deliberately as training-only.
- **SRNN-family convention drift.** The four sibling wrappers disagree on absolute vs
  recentred coords and on body vs world frame for `temporal_edges`. Patching dsrnn and
  gensafenav to agree with the others changed nothing; both patches were reverted.

Still untested (needs a live run with the console attached, since benchmark runs discard node
output - see [[project-benchmark-discards-node-logs]]): the actual theta_star plan geometry,
the actual jackal laser scan, and sim dynamics/acceleration limits.

Separately genuine per-planner defects, failing offline too: dsrnn (into the south wall at
3.2 s), gensafenav and sonic (identical trajectory, wall at 5.0 s - shared GST/talk2Env
lineage, treat as one), attngraph (near-constant saturated action independent of heading and
goal). crowdnav is unevaluable, see [[project-crowdnav-no-weights]].

RESOLVED 2026-08-09: three causes, none planner-side - mpo700 belly-below-wheels chassis collision (born broken at arena_robots d42cfac), measurement window missing episode 1, and the paused-teleport reset-pin [[gz-paused-teleport-pin]]. Post-fix: her-drl/cadrl/kdma/crowdsurfer/drlvo all reach the s-bend goal.
