---
name: Use commit messages verbatim
description: When the user provides a commit message, use it exactly — no body, no summary, no Co-Authored-By line unless they ask
type: feedback
originSessionId: 0c8fbab8-b35f-4e7f-b26f-4c2a9fc1bf54
---
When the user says "commit as '<message>'", use that string verbatim as the commit message. No body, no bullet-list summary, no Co-Authored-By line.

**Why:** The user prefers terse commit messages; they explicitly rejected an elaborated version ("only use the commit message i gave you").

**How to apply:** Pass `git commit -m "<exact message>"`. Don't add a body even if the change is large. If a Co-Authored-By line is ever wanted, they'll ask.
