---
name: Questions are questions, not tasks
description: When the user asks "shouldn't X..." / "isn't it odd that..." / "why does X...", answer the question; don't jump to implementing a fix
type: feedback
originSessionId: 8887a667-8183-4177-83bc-b1e6be83ad1a
---
When the user phrases something as a question ("shouldn't X share Y semantics?", "why does this...", "isn't it weird that..."), they are asking for analysis/confirmation, not authorizing a change. Answer the question and stop.

**Why:** User said "it was just a damn question" after I answered + immediately edited the file. Being in Auto mode does not convert rhetorical/exploratory questions into work orders.

**How to apply:** If the message is a question (ends in `?`, or starts with shouldn't/isn't/why/does/can/what), default to answering only. Wait for an imperative ("fix it", "change it", "go ahead") before editing. Auto mode speeds up agreed work; it doesn't widen what counts as agreed.
