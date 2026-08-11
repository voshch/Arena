---
name: feedback-dds-cold-cache-double-run
description: First ros2 CLI call in a session returns empty (DDS cold cache); always run discovery commands twice and trust the second
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 81fbf01f-be53-42ec-80ae-8071c28f3efc
---

The first `ros2 node list` / `ros2 topic list` in a fresh shell against the running container returns empty or partial output. This is a known DDS cold-cache bug, not a sign the sim is down.

**Why:** discovery cache is cold on the first invocation; the daemon warms up only after the first query.

**How to apply:** run the discovery command twice (or discard the first result) via [[feedback-run-via-arena-source]]. Never conclude "no nodes / topic missing" from a single first call.
