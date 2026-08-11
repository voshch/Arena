---
name: feedback-kill-container-processes-after-runs
description: How to sweep arena/gz processes after runs without killing the Claude session or failing silently
metadata: 
  node_type: memory
  type: feedback
  originSessionId: eae9a72e-c1b6-4ab0-b430-744ad06444d4
  modified: 2026-08-07T07:38:28.456Z
---

Arena/gz processes run in the `arena-arena_ws-arena-1` container, which has its **own PID namespace**. Host-side `ps aux` shows them (with host PIDs), but `kill -9` from the host returns `Permission denied` even at matching uid. Sweeps MUST run inside the container via `source arena -c '...'` from `~/arena_ws`.

Two traps, both hit for real:

1. **`pkill -f <pattern>` matches the Claude session itself.** The `claude` binary's cmdline carries the full `--add-dir` list, so patterns like `arena_evaluation`, `arena_planners`, `task_generator`, `arena_bringup`, `arena_runtime` all match it and kill the session. The wrapper shell running the `-c` string matches too.
2. **Host-side kills fail silently.** `kill -9` in a loop with `2>/dev/null` reports nothing and leaves every process standing; it looks like a successful sweep.

**How to apply:** build a protected PID set from `$$` walking up via `ps -o ppid=` first, then `pgrep -f "$pat"` and skip any protected PID before `kill -9`. Always print survivors + `free -g` afterward and read them — never assume the sweep worked. See [[feedback_memory_headroom]] for the pre-launch gate and [[feedback_run_via_arena_source]] for the invocation form.
