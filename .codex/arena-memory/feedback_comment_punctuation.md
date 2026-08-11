---
name: feedback_comment_punctuation
description: "Authored comment punctuation: no semicolons anywhere, no trailing period on single-sentence comments"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 3553827a-da09-4bfe-b7d6-44750c656f54
  modified: 2026-08-09T16:29:46.501Z
---

Zero semicolons in authored prose (comments, commit messages, docs). Single-sentence code comments take NO trailing period ("slop boy" callout 2026-08-10). Docstrings keep normal sentence punctuation.

**Why:** trailing periods on one-liner comments and semicolon chains read as generated slop to the user.

**How to apply:** when writing any `# comment` that is one sentence, end it bare. Multi-sentence comments keep internal periods. Don't churn pre-existing comments solely for this.
