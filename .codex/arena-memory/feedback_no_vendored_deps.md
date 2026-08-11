---
name: no-vendored-deps
description: "Vendoring third-party packages into the repo is unacceptable, prefer stdlib rewrites"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: edfe0404-d61a-4251-94a5-f6104c1f85f1
  modified: 2026-08-04T14:41:19.181Z
---

User rejected a vendored copy of click ("ugly") in favor of a stdlib rewrite, even at the cost of porting 12 files.

**Why:** vendored blobs pollute the repo and need manual maintenance.

**How to apply:** when a dependency becomes a liability, rewrite against stdlib instead of vendoring ([[arena-cli-no-uv]]).
