---
name: Parallelize implementation with sonnet agents
description: For multi-part implementation work, default to spawning parallel Agent tasks on Sonnet rather than executing sequentially myself
type: feedback
originSessionId: a73b9394-cae4-4480-8533-e820d621bc6d
---
For multi-part implementation work, default to spawning parallel Agent calls (subagent_type=general-purpose, model=sonnet) in a single message instead of working serially in the main thread.

**Why:** parallel sonnet is meaningfully faster for independent work; user trades orchestration complexity for wall-clock speed.

**How to apply:** identify independent units, send parallel Agent calls with fully self-contained prompts (exact paths, content). Keep truly sequential steps (git submodule adds, .gitmodules edits) in main thread. Skip parallelism if fewer than ~2 independent units or shared-state mutation.
