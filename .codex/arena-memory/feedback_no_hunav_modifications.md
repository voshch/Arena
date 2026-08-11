---
name: Don't modify deps/hunav without explicit permission
description: HuNav (deps/hunav, the hunav_sim packages) is treated as upstream — workarounds belong on the task_generator/Arena side unless the user explicitly OKs a HuNav patch
type: feedback
originSessionId: 8a90b351-41a8-4fd7-804a-484b539f6b4d
---
When fixing humansim/HuNav-related bugs, default to working around them on the
Arena/task_generator side. Do not edit files under `deps/hunav/` or
`hunav_sim/` unless the user has explicitly approved the patch first.

**Why:** the user has rejected a "patch HuNav to add a service" fix and
preferred a workaround using only existing HuNav APIs (clear_agents +
re-send with cached full Agent records overlaid with live pose). They
treat HuNav as a maintained upstream and don't want Arena-specific
behavior bolted onto it without a discussion.

**How to apply:** if the natural fix touches `deps/hunav/...`, propose
the workaround variant first (even if uglier) and only modify HuNav
after explicit go-ahead. The Arena fork already has narrow Arena-only
extensions in `_arena_hunav_node.{hpp,cpp}` — adding to that file is
still a HuNav modification and needs the same approval.
