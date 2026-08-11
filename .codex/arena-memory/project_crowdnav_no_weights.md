---
name: project-crowdnav-no-weights
description: The crowdnav planner has no checkpoint anywhere and silently runs a randomly initialised SARL net
metadata:
  type: project
---

`arena_planners/planners/crowdnav` has no `model/` directory and its `weights.yaml` is
`files: []`. Its own README states upstream (vita-epfl/CrowdNav) publishes no pretrained
weights. `_build_policy` guards the load with `if weights.exists():`, so with none present
SARL runs at random initialisation and the planner produces meaningless actions without
any error. Benchmark results for crowdnav measure nothing.

Sibling `cd-sarl` is the same architecture with a real checkpoint (arena-rosnav/cd-sarl,
392 KB) and is the working SARL representative. Its loader now raises when the file is
missing instead of falling back to random init.

Only crowdnav is affected: an audit of all 17 registry planners on 2026-08-07 found every
other declared weight file present (sicnav is correctly weightless, being pure CasADi MPC).

**Why:** a 0% success rate looked like a policy result; it was an unprovisioned planner.

**How to apply:** crowdnav is a dead end until someone trains a checkpoint. Do not report
its benchmark numbers as a CrowdNav result. See [[project-drl-planner-offline-probe]].
