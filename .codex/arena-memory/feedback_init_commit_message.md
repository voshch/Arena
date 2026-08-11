---
name: feedback-init-commit-message
description: "First commit of vendored mesh/asset mirror repos should be messaged 'init'"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7e3405d0-3812-4b3e-94dc-10216c6fa11c
---

For the **first commit** of a vendored asset-mirror repo (the per-robot `arena-robots/<name>` mesh repos — license + meshes squashed into one commit), the user wants the message to be just `init`.

**Why:** these are dumb asset mirrors; a descriptive first-commit message adds nothing the bundled `NOTICE` file doesn't already carry.

**How to apply:** seed the repo with license + assets in a single commit messaged `init` (amend + force-push if it was already pushed under another message — safe on a fresh, un-pulled repo). Distinct from [[feedback-commit-messages-verbatim]] (which governs messages the user dictates for real code commits). Mesh-repo attribution + the `init` convention also live in the integrate-robot skill's Phase 4.
