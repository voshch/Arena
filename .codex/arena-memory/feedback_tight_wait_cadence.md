---
name: tight-wait-cadence
description: "Poll/wakeup cadence during launches and long waits must be <=5 minutes, never 20-minute fallbacks"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 26189c30-8f48-467e-852a-a6dc9b883705
---

While waiting on sim bringups, benches, or background tasks, check back at most every ~5 minutes (prefer ~4.5 min to stay in prompt cache), never 20-minute fallback wakeups.

**Why:** user watched a wedged env sit unnoticed for ~20 min because my fallback wakeup was 1200 s and the readiness poller was watching for something that could never happen ("wtf are these 20 minute timings??", "keep it tight, 5 minutes max unless absolutely necessary", 2026-07-03).

**How to apply:** ScheduleWakeup delays <=270 s during active launch supervision; pair every "wait for X" poller with a failure-path watcher (process died, traceback in log) so a crash fires immediately instead of timing out; check logs proactively when a step exceeds its expected duration. Related: [[no-concurrent-isaac-launch]].
