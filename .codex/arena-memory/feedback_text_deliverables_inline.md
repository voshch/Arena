---
name: feedback-text-deliverables-inline
description: "Plans and text documents go inline in chat or to a local file, never Artifacts; Artifacts only for genuinely visual pages"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 5ba49527-47d6-4a50-9f6e-500945128630
---

Plans, reports, and other text deliverables must be printed inline in the reply (and/or written to a local file the user can open), not published as Artifacts. Artifacts are reserved for genuinely visual content where rendering matters (UI mockups, diagrams).

**Why:** User asked "why is that an artifact and not printed inline/locally?" after the implementation plan was published as an Artifact (2026-07-18); the extra hop to claude.ai added nothing over text they can read in place or open in the editor.

**How to apply:** Markdown plan/doc deliverable = paste inline + optionally keep the scratchpad file and link its path. Only reach for Artifact when the deliverable is a rendered page (like the [[project-human-manual-steering-plan]] mockup).
