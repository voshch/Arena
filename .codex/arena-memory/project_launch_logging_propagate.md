---
name: project_launch_logging_propagate
description: "ros2 launch's logging init makes later-created loggers non-propagating, silently dropping records"
metadata: 
  node_type: memory
  type: project
  originSessionId: f0263fb2-f253-42ec-9ecf-e6795abbd2de
---

ros2 `launch.logging` (pulled in transitively by `arena_rclpy_mixins`/`task_generator` → `launch`) reconfigures Python `logging` so that loggers created *after* it is imported default to `propagate=False`. A module-level `_log = logging.getLogger(__name__)` in such a process is created during import (after launch) and inherits `propagate=False` with no handler of its own, so every `_log.*` record dead-ends before any console/file handler. INFO then silently falls to `logging.lastResort` (WARNING level) and is dropped, so WARNING/ERROR still show but INFO/progress vanishes.

Symptom found in `arena_evaluation` benchmark runner: per-episode `[i/N]` progress lines never appeared on console or in runner.log, while the startup `asyncio` DEBUG (logger created before launch import) did. Fix: set `_log.propagate = True` in `cli_main` after `basicConfig`/`setLevel` (runner.py). Any launch-importing CLI with a module-level logger is exposed to this; restore propagate (or attach a handler directly).
