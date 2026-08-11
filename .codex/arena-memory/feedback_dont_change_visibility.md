---
name: dont-change-visibility
description: "Don't change attribute/method visibility (rename `_foo` ↔ `foo`) when the user asked for an unrelated fix"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 726dc8b0-287b-4262-a725-a3684e3fdb16
---

When refactoring an attribute (e.g., cached field → property), keep the existing name and visibility. Don't simultaneously promote `_foo` to `foo` (or vice versa) just because the refactor "lets you" drop a `# noqa: SLF001` or tidy up callers.

**Why:** the user called this out after I converted `_goal_tolerance_distance` → public `goal_tolerance_distance` while fixing a separate staleness bug. Visibility was an intentional design choice; flipping it expanded scope past the bug and rewrote a call site (request.py) and tests for no reason the user asked for.

**How to apply:** treat the leading underscore as load-bearing — it signals "callers are internal-only by convention". Don't change it unless the user explicitly requests a visibility change. Same goes for the reverse: don't add an underscore to demote a public name during a refactor. Keep the diff scoped to the bug.
