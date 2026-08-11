---
name: Match surrounding comment density
description: Before adding a comment, check how densely the surrounding file/codebase comments and match that
type: feedback
originSessionId: ca5aef96-31aa-4470-883e-f85ba184e27b
---
Match the file's comment style. If surrounding code is sparse or bare, don't drop a paragraph-long explanation in, even a legitimate "why" one.

**Why:** "too many damn comments. read the room" — transplanting a 4-line rationale into a comment-free file was jarring.

**How to apply:** glance at neighbors before commenting. Default to no comment if neighbors have none. Applies to docstrings too: cap at one short line unless the file runs longer.
