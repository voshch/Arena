---
name: In planning, hold off implementing until told
description: While the user is actively planning/discussing architecture, don't touch files; wait for explicit go-ahead even under auto mode
type: feedback
originSessionId: 74f7497b-8718-44e3-ac29-cfdfe07c5d8f
---
When the conversation is in a planning/discussion phase (architecture comparisons, "what if we..." questions, plan reviews), do not start editing files even if auto mode is active. Wait for an explicit instruction to begin (e.g. "go", "start", "implement now", "do it"). Auto mode's "prefer action over planning" does NOT override an in-flight planning conversation.

**Why:** The user got burned when I jumped to implementation mid-plan-revision and they had to interrupt and roll back partial edits. Premature implementation also locks in an architecture that's still being negotiated, which costs more to undo than to defer.

**How to apply:** Signals that the conversation is still in planning: the user is asking comparative/exploratory questions ("X vs Y advantages"), making plan revisions ("what about doing Z instead"), questioning assumptions in a recent plan ("why does it need W?"), or saying things like "let's rethink", "reassess", "we're still planning". In these states, respond with reasoning + revised plan diffs only. Do not call Edit/Write/Bash-that-mutates. Wait for an unambiguous green light. When in doubt, ask once rather than act.
