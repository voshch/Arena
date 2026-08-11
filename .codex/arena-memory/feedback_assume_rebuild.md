---
name: Assume rebuilds
description: Do not ask the user to verify a rebuild happened; assume edits are live.
type: feedback
originSessionId: 55c84bef-4608-4339-805b-db8e06fd2757
modified: 2026-08-06T04:48:59.256Z
---
Do not ask "did you rebuild?" or propose "make sure the build picked up the edits" as a debugging step.

**Why:** User handles their own build lifecycle and treats the question as condescending. Raised explicitly after I suggested a stale-build verdict for why `[ART_REINIT]` prints weren't visible.

**How to apply:** When diagnostic output isn't appearing, skip "check the build" and go straight to the other hypotheses (wrong import path, wrong event, swallowed exception, wrong stream, etc.). Also applies to any other "did you restart / reinstall / re-source" variant.

The build lifecycle being theirs also means: never mutate `build/` or `install/`. Do not delete stale artifacts colcon left behind after a file move or rename (dead console scripts, orphaned share dirs) - report them and let the user rebuild. Said after I removed a stale `install/arena_runtime/lib/arena_runtime/cam` script post-move: accepted, but they would have rebuilt themselves. Building (`colcon build --packages-select ...`) to verify your own work is fine; hand-editing the trees is not.
