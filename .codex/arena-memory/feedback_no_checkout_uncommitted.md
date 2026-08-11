---
name: no-checkout-uncommitted
description: "Never `git checkout HEAD -- <file>` to revert in this workspace; much PR work is uncommitted and not in HEAD"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 45bf66a1-d89a-41f9-9b4a-f0eca94c96e9
---

Never use `git checkout HEAD -- <file>` (or `git restore`) to "revert" a file in this workspace. Large parts of in-progress PRs live as **uncommitted working-tree changes that are NOT in HEAD** (HEAD lags the actual work). Checking out HEAD silently destroys that uncommitted work.

**Why:** During the multi-level-world PR cleanup I ran `git checkout HEAD -- <5 files>` to undo out-of-scope edits and clobbered uncommitted `_active_level_id`/`level_id` wiring in environment/parametrized/explore/prompt impl.py (it existed in the working tree but not HEAD). Recovered only because the content had been staged at some point, leaving dangling blobs findable via `git fsck --unreachable` + `git cat-file -p`.

**How to apply:** To undo my OWN edits made this session, re-edit the specific lines back (I know what I changed) or read the prior content first. Don't reach for checkout/restore. To verify whether code is PR-added vs pre-existing, compare against the PR base commit (`git show <base>:<file>`), NOT HEAD. If a revert truly needs git, prefer `git stash` (recoverable) and confirm with the user first. If uncommitted work is ever lost, recover via `git fsck --lost-found`/`--unreachable`, match blobs to files by content, and verify byte-identity before restoring. Related: [[feedback-merge-audit-diff-first]].
