---
name: memory-headroom-2gb
description: Isaac launch memory gate; box OOMed 2026-07-03 and again 2026-07-21 (needed reboot)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 26189c30-8f48-467e-852a-a6dc9b883705
  modified: 2026-07-21T12:14:21.885Z
---

The box (31GB) went OOM on 2026-07-03 while my go1 Newton verification launch ran alongside the user's own work. It happened AGAIN on 2026-07-21: I launched Isaac headless with a 394MB collected world while `free -h` showed 10GB available and **13GB swap already in use**; the host wedged hard enough that the user had to reboot. 10GB available is NOT enough for Isaac, and swap-in-use means the box is already over-committed.

**Why:** Isaac + Newton runs push past 28GB used; heavy asset worlds (collected scenes, big textures) add several GB more. Swap-in-use at launch time means there is no real headroom regardless of what "available" says.

**How to apply:** Gate every sim launch on `free -h`: abort/defer if swap used > ~1GB OR available < ~15GB for Isaac (6GB for gazebo/dummy). When launching anyway, arm a watchdog alongside the poller that kills the launch when available drops below ~3GB. After every run, do the verified kill sweep ([[kill-container-processes-after-runs]]) including stale `ros2 service call` and `world_generator` processes in the arena container. Never run two sims concurrently ([[no-concurrent-isaac-launch]]).
