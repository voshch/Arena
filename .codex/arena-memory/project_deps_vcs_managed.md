---
name: project_deps_vcs_managed
description: "src/deps is vcstool-managed via _meta/repos/arena.repos, NOT git submodules; arena_robots/arena_planners ARE Arena submodules tracking jazzy"
metadata: 
  node_type: memory
  type: project
  originSessionId: dc29dc24-d16d-4742-ac55-6889e637be94
---

`src/deps/*` (hunav, nav2, hri, slam_toolbox, etc.) is populated by `vcs import --input _meta/repos/arena.repos` (run by `arena update`/`arena deps`, see `_meta/tools/source:79-83`), NOT git submodules. Deps lacking jazzy support are forked under `voshch/<name>` on a `jazzy` branch and pinned by `version:` in arena.repos.

By contrast, `arena_robots`, `arena_planners`, `arena_tools`, `arena_evaluation`, `arena_isaac`, `arena_training`, `humansim` ARE git submodules of Arena (Arena `.gitmodules`); arena_robots and arena_planners track `branch = jazzy`. So controller configs / launch / robot code live in the arena_robots submodule repo, and bridge-planner submodules nest under arena_planners.

`ARENA_ROS_DISTRO` defaults to `jazzy` (`_meta/tools/source:38`). Active dev is the jazzy line; `feature/drl-planners` has jazzy as ancestor (jazzy + DRL-planner commits). Don't assume deps are submodules and don't edit deps expecting an Arena-tracked pin. See [[feedback_verify_before_claiming_absent]].
