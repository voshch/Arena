---
name: Planning is not authorization
description: spec confirmation ("all sound?", "yes", "agreed") is not a go-ahead to implement
type: feedback
originSessionId: 306f1ed2-6939-4cb4-8dd2-41f33491651d
---
When the user is iterating on a spec or design, treat their replies as planning input, not implementation orders. Confirmations like "yes", "all sound?", "agreed", "sounds good" lock in design decisions; they do **not** authorize touching files.

**Why:** user explicitly said "we'll put that into the plan later" earlier in the same conversation, then later answered four spec questions and asked "all sound?" — I treated the auto-mode prompt as license to start writing files. They interrupted with "i didn't tell you to implement. we're still planning!!" and made me revert.

**How to apply:** wait for an explicit imperative ("implement", "do it", "go ahead", "write it", "let's build it") before editing or creating files. Auto mode does not override an active planning conversation. Even after locking the spec, summarize the plan and await go-ahead.
