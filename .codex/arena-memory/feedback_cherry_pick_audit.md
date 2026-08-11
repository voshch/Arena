---
name: cherry-pick-audit
description: "When cherry-picking across branches, check both branches' HEAD state for each commit, not just the patch in isolation"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1bace4a7-3711-443c-ab35-ec4fe08c6fff
---

When asked which commits to cherry-pick from branch A to branch B, verify two things per commit before recommending:

1. **Is the change already on the destination?** A `git cherry-pick --no-commit <sha>` that returns rc=0 with an empty `git diff --cached --stat` is a no-op — the patch resolves to nothing because the change is already present (often hand-applied earlier). Do not call that "applies cleanly"; it means "skip, already there."
2. **Is the change still current on the source?** A commit may be a transient workaround for a then-broken state. Check `<file>` on source HEAD, not just the commit's own diff. If source HEAD reverted the surrounding context, the fix is obsolete on both ends and porting it in isolation reintroduces a problem neither branch has.

**Why:** I once recommended 7 "clean" cherry-picks; 5 were already applied and 1 (`397def1`) was a workaround for a realizer change (`50f679f`) that humansim itself later reverted. Picking it onto jazzy broke rviz/navigation by introducing a fix for a problem jazzy never had.

**How to apply:** For each candidate commit, run the cherry-pick dry-run and inspect `git diff --cached --stat` explicitly (not just rc); then `git show jazzy-humansim:<path>` (or other source HEAD) to confirm the change reflects current intent, not a stale intermediate. Related: [[no-senseless-diffs]].
