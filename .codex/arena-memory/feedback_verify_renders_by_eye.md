---
name: verify-renders-by-eye
description: Extract and view a frame of any render before delivering claims about it
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d1058176-4dfa-4e43-b9e3-c10599875612
---

Delivered walk.gif captioned "title removed" while the title was still baked in: the removal patch ran in a background shell whose cwd had reset, failed silently, and the claim was never checked against pixels (2026-07-18, user caught it).

**Why:** file-write success and render-script exit codes do not prove what the image shows; cwd resets make relative-path patches silently no-op.

**How to apply:** before sending any image/GIF with a claim about its content, extract a frame (PIL seek(0)) and Read it; patch scripts with absolute paths or a leading cd inside the SAME command; never caption a render with an unverified property. See also [[verify-before-claiming-absent]].
