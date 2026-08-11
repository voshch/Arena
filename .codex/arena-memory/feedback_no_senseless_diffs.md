---
name: No senseless git diffing
description: Don't run git diff unless actually reviewing history/changes; look at current file state instead
type: feedback
originSessionId: 0c8fbab8-b35f-4e7f-b26f-4c2a9fc1bf54
---
Don't invoke `git diff` / `git status` as a first step for debugging. Read the current file state directly.

**Why:** The user wants investigation grounded in what exists now, not in what changed. Diffs distract from root-cause analysis of runtime behavior (e.g. a TF double-rotation bug doesn't care which lines were touched last).

**How to apply:** Only use git diff/log/status when the task is explicitly about reviewing changes, writing commit messages, or understanding history. For bugs and behavior questions, open the relevant files with Read/Grep and reason from their current contents.
