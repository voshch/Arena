---
name: No automatic commits
description: Don't commit changes without explicit per-commit approval; stage and wait
type: feedback
originSessionId: 755cb100-9181-414e-b144-ca250fc3de8a
---
Don't run `git commit` on your own. Stage changes, show a diff, and wait for explicit approval before each commit — even if the earlier turn in the same session authorized one commit, that permission does not extend to subsequent commits.

**Why:** User pushed back after I committed a formatting change and a perf change back-to-back without asking, having only been authorized for one earlier commit in the same session. Authorization for one commit is not a blanket license.

**How to apply:** After editing, leave the working tree modified/staged, summarize the change, and ask. Only commit when the user explicitly says so for that specific change.
Refinement 2026-07-31: do not even STAGE without an explicit instruction or an active autonomous grant. Interactive sessions: worktree edits only, user controls the index entirely.
