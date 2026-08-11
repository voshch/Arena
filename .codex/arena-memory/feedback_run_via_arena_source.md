---
name: Run commands via `source arena -c '...'`
description: One-shot ROS/Gazebo/colcon commands run via `cd ~/arena_ws; source arena -c '...'`, which docker-execs into the container; the inner command must source the arena env itself
type: feedback
originSessionId: 19bb518d-f698-4aa5-b69f-346001c41a10
modified: 2026-07-19T17:19:00.573Z
---
To run a one-shot command in the arena container from the host:

```
cd ~/arena_ws; source arena -c 'source /opt/arena_ws/source > /dev/null && <command>'
```

`~/arena_ws/arena` is a symlink to `src/Arena/_meta/docker/source`. Sourcing it runs `do_run "$@"`, which forwards args to `docker compose exec ... bash --norc "$@"` inside the `arena` service container. The `-c '...'` flag is bash-passthrough.

**Why the inner `source /opt/arena_ws/source` is required:** the non-TTY path uses `--norc`, so the rcfile that auto-sources the arena env (ROS overlay, venv, PATH, `arena` function) is *not* loaded. Without sourcing inside, `ros2`/`gz`/`colcon`/`arena` are missing inside the container.

**How to apply:** for any single ros2/gz/colcon/arena invocation, wrap as `cd ~/arena_ws; source arena -c 'source /opt/arena_ws/source > /dev/null && <command>'`. Don't use `docker exec arena-arena_ws-arena-1 ...` directly (it skips the arena env), and don't expect `bash -c` to pick up env from the host.

**Tests too (user, 2026-07-19):** pytest runs also belong in the container venv, never system python or scratch venvs. System python (3.14) mismatching the venv wheels is irrelevant noise, not a defect: "system python3.14 has no business here." Run suites per package (separate pytest invocations, own rootdir): combining test paths across packages breaks conftest/sys.path collection. arena_evaluation tests need the evaluation feature container (submodule not mounted in the main arena container); the user's full sweep runs from /opt/arena_ws.

**EXECUTION IS CONTAINER-ONLY (user, 2026-06-23):** the user explicitly restricted: only authorized to run things via `cd arena_ws; source arena -c '...'`. Do not execute sims/heavy processes on the host. The container has its OWN PID namespace: host `pgrep -af` SEES container procs (with host-side PIDs) but host `kill`/`pkill` CANNOT signal them. To kill a launched sim, do it inside the container: `source arena -c 'pkill -9 -f "gz sim"; pkill -9 -f arena_node; pkill -9 -f world_generator; pkill -9 -f parameter_bridge; pkill -9 -f arena_runtime.launch'` then verify with an in-container pgrep. Always tear the sim down before ending the turn (see [[feedback_kill_container_processes_after_runs]]).
