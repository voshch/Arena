---
name: feedback_precommit_checklist
description: pre-commit checklist; never stage unless explicitly asked
metadata: 
  node_type: memory
  type: feedback
  originSessionId: c9cf54ae-209f-488f-8b24-a173db45bc7b
  modified: 2026-08-01T11:22:07.116Z
---

Wrap-up checklist after edits: (1) remove introduced AI artifacts (justifying/narration comments, attribution trailers, change-tombstones, anything that reads as machine-generated), (2) format, (3) check (`ruff check .` from Arena root). Do NOT stage — leave changes unstaged in the working tree. The user stages and commits himself, and the index may already hold his curated in-progress commit that a `git add` would contaminate (happened 2026-08-01, user objected).

**Why:** keeps commits clean and reviewable, and keeps the index solely under the user's control.

**How to apply:** stage only when the current message explicitly asks for it. Artifact removal builds on [[feedback_no_justifying_comments]] and [[feedback_useless_comments_scope]]; check step is [[feedback_ruff_check_whole_tree]]; committing follows [[feedback_no_auto_commits]] and [[feedback_commit_messages_verbatim]].
