---
name: ssot-audit-on-new-facts
description: "When introducing an authoritative fact, sweep for hand-authored duplicates and propose derivation; broken findings always go on the debt list"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1b5c52cf-7d20-4b92-bf27-92ed227b52de
  modified: 2026-08-03T06:07:35.508Z
---

When a design introduces an authoritative value (a polygon, radius, dimension, path), sweep the codebase for existing hand-authored encodings of the same physical fact and either propose deriving them from the new SSOT or explicitly justify their independence. One-way derivation (fact feeds policy default, override wins) usually satisfies decoupling concerns that one-directional "isolation" framing hides.

**Why:** In the collision_polygon design the user had to manually flag that StopPolygon should derive from it. I had measured StopPolygon as uselessly undersized in the same session but had filed the fact under "rejected alternative" with a "different semantics" label, so it never joined the under-declaration debt list (footprint, radius) it belonged to.

**How to apply:** After designing a new SSOT, grep for numeric/geometric siblings (polygons_dict entries, radius, safe_distance, padding). Anything measured broken goes on the named debt list regardless of its semantic label. Surfacing the connection is not scope creep, it is the "propose extras separately" half of [[no-speculative-scope]].
