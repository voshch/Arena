---
name: No arena_humansim modifications without approval
description: deps/arena_humansim is upstream like HuNav; workaround on Arena side first, only patch with explicit approval
type: feedback
originSessionId: 74d3ad74-eb35-4967-a58d-011de9f5c02d
---
`deps/arena_humansim` is upstream and treated the same way as `deps/hunav`: do not modify it without an explicit go-ahead, even when the user owns the upstream and even when the bug clearly lives there.

**Why:** the user wants Arena-side and upstream-side fixes scoped separately. Patching upstream from inside an Arena task implicitly fans the change out to other consumers of arena_humansim, which the user wants to control deliberately.

**How to apply:** when a bug traces to `deps/arena_humansim`, diagnose, propose the upstream fix in chat, but do not edit any file under `deps/arena_humansim/` until the user explicitly approves. Default to proposing an Arena-side workaround first; if no clean workaround exists, say so and ask for upstream approval explicitly.
