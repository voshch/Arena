---
name: recheck-git-state-before-status-claims
description: "User commits/pushes staged work himself mid-session, re-check git state before any staged/unpushed/pending claim"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: ada36411-a804-4885-9c26-7d2cc39a2ccf
  modified: 2026-07-29T18:25:57.519Z
---

The user works in the same tree concurrently and routinely commits and pushes work Claude staged, bumps submodule pins, and amends commits mid-session ("they're pushed. you should check this before you lie", 2026-07-29).

**Why:** Status claims ("staged, awaiting commit", "unpushed", "pin bump pending") go stale within minutes. Repeating them unverified reads as lying.

**How to apply:** Before stating anything about staged, committed, pushed, or pinned state, run `git status -sb` / `git log --oneline` (and `git fetch` + compare for push claims) in every affected repo. A moved HEAD noticed in passing is a signal to re-check everything, not to ignore. Related: [[verify-before-claiming-absent]].
