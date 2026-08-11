---
name: feedback-pr-hygiene-ok
description: "Don't downrate PRs for being big; don't propose splitting. User prefers single large PRs."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 4ff60925-c6b9-407a-a92b-9dd0eb233ee2
---

PR hygiene / size is not a concern for this user. Don't flag big PRs as a quality issue, don't propose splitting into smaller PRs, don't lower a "hygiene" rating for high LOC or many files.

**Why:** User explicitly likes big PRs ("fuck pr hygiene, i don't care about that axis. i even like big prs"). The merge dance and bisect-difficulty tradeoffs of a single comprehensive PR are fine by them.

**How to apply:** When estimating PR quality across axes, drop "PR hygiene" or rate it 10/10 unconditionally. Don't suggest splitting agents into separate PRs. Don't surface "this PR is getting big" as a concern in planning discussions.
