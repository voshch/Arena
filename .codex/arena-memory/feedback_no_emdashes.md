---
name: No emdashes
description: No fancy typography anywhere: no emdashes, no Unicode ellipsis, ASCII only in code and chat
type: feedback
originSessionId: 798a2e85-6513-46d7-8434-6bc4f7f5b1fa
modified: 2026-07-29T19:10:23.633Z
---
Don't use emdashes (—) anywhere: code, docstrings, comments, commit messages, or chat responses. Use a colon, period, comma, or parentheses instead. Same for the Unicode ellipsis character: use three ASCII dots if an ellipsis is needed at all (user, 2026-07-29: "did you actually put an ellipsis in a fixed-width terminal?").

**Why:** User explicitly called this out after I dropped emdashes into rewritten docstrings, and again for a Unicode ellipsis in a chat message. Fixed-width terminal rendering; keep output ASCII.

**How to apply:** Default to colons or sentence breaks for the kind of pause an emdash would cover. Even when the rewrite "reads better" with one, don't. No typographic Unicode substitutes for ASCII punctuation anywhere.
