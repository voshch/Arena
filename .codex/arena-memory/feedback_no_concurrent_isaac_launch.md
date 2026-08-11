---
name: feedback_no_concurrent_isaac_launch
description: "Never spawn a heavy sim launch (Isaac) without checking the user isn't already running one; duplicates stomp and OOM the box"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: af69523e-6daf-4ccb-ae8a-c988ec14047a
---

Never start `arena feature isaac launch` (or any heavy sim) when the user might already be running it by hand. On 2026-06-21 I launched a second headless Isaac while the user had one running; the box OOM-crashed.

**Why:** the isaac compose service is a single named container (`arena-arena_ws-isaac-1`). My `docker rm -f <it>` + `arena_docker_compose up isaac` killed the user's hand-run instance AND stacked a second Isaac, exhausting the ~31GB host (Isaac 6.0 alone spikes >10GB, plus firefox ~7GB). The user had explicitly said "forget about memory" right before, but the real fault was launching a duplicate at all, not the memory math.

**How to apply:** before driving any heavy/exclusive launch, check for an existing instance (`docker ps --filter name=isaac`, `pgrep -af run_isaacsim`) and ask. Default to letting the user drive launches and me diagnosing from pasted output. Don't `docker rm -f` / recreate a single-named container the user may be using. Relates to [[feedback_kill_container_processes_after_runs]] and [[feedback_run_via_arena_source]].
