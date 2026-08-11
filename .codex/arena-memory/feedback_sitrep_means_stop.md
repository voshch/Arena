---
name: sitrep-means-stop
description: "\"sitrep\" is a halt order: report immediately and stop launching/running anything until told to continue"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c4774e32-1971-447e-aac9-2a457703fef1
  modified: 2026-07-29T11:31:21.704Z
---

When the user asks for a sitrep (or any mid-task status request), stop all further tool runs, give the status report, and wait for direction. Do not continue the pipeline after reporting.

**Why:** 2026-07-29 during the gz rgbd tearing investigation, two sitrep requests were answered with a report followed immediately by more launches/reruns. The user interrupted: "if i tell you sitrep, you do it now. you don't run more and more stuff."

**How to apply:** Treat "sitrep" as an interrupt, not a side query. Finish at most the currently in-flight command, report, end turn. Resume only on explicit go-ahead. Related: [[questions-not-tasks]].
