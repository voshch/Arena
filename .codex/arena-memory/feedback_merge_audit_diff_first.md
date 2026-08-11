---
name: merge-audit-diff-first
description: "When auditing a merge resolution, measure with diff and wc before reading; mental diffs flip sides"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e9084c01-4343-4026-993c-da6a9b5268dd
---

When auditing an in-progress merge resolution, the FIRST tool calls must be quantitative size/diff measurements, before reading any file content:

```
wc -l ours theirs resolved
diff resolved ours   | wc -l
diff resolved theirs | wc -l
```

The smaller diff tells you which side the resolution mostly tracked. Only then read the small diff to confirm the additions/subtractions are sensible.

**Why:** in a real session (jazzy → jazzy-humansim merge, 2026-05-15) two parallel Sonnet sub-agents systematically inverted ours/theirs on 4 of 9 files they reviewed (`gazebo_simulator.py`, `World.py`, `world_manager.py`, `robot_manager.py`). They claimed "ours was kept, theirs was dropped" on files where the resolved bytes were essentially `theirs + small additions`. The branch with more code (humansim) anchored their mental model, so a slim resolved file looked like a regression from `theirs`. A single `wc -l` would have caught it: world_manager.py was ours=317, theirs=185, resolved=186, clearly tracking theirs, no chance the agent reasoned about it correctly without measuring.

**How to apply:** any merge-audit task (mine or delegated to a sub-agent) must front-load the diff/wc step. When prompting an audit agent, give the exact commands above and require the size table in its report. Avoid prompt phrasing that primes a side-preservation bias (e.g. "did the humansim-side additions survive?"); phrase neutrally as "which side does the resolution track per file, and is the delta sensible?". See also [[feedback_no_senseless_diffs]], merge audits are the explicit exception where diffing IS the work.
