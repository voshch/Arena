---
name: exec_depend doesn't cause build cycles
description: exec_depend doesn't feed colcon's build graph, but colcon TEST does topo-sort exec_depends — a cycle breaks `colcon test`
type: feedback
originSessionId: 81f33989-0b2a-4e98-bd24-decd319f333b
---
Adding a `<exec_depend>` to package.xml cannot create a colcon *build* cycle. Only `<build_depend>` / `<buildtool_depend>` / `<depend>` feed the build graph; `<exec_depend>` is runtime ordering.

However `colcon test` topologically orders packages using run dependencies too: an exec_depend cycle makes it fail with "Unable to order packages topologically" (observed 2026-06-12 when adding `<exec_depend>arena_simulation_setup</exec_depend>` to arena_bringup while arena_simulation_setup already exec_depends on arena_bringup).

**Why:** User corrected a "verify no cycle" warning on a build-graph basis, but later a real exec_depend cycle broke `colcon test`.

**How to apply:** Don't warn about build cycles for exec_depend, but don't *create* exec_depend cycles either — colcon test chokes on them. Prefer leaving the dep undeclared when the reverse exec_depend already exists.
