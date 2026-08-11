---
name: Terse comments in IDL files
description: Don't write multi-line comment blocks in .srv/.msg/.action files; one short line per field max
type: feedback
originSessionId: 71d3c70d-8cf7-4c7f-8501-c09b12eb9864
---
Don't write multi-line `# ...` blocks above fields in `.srv` / `.msg` / `.action` files. One short line per field is the cap; if a field needs an essay, the design is wrong.

**Why:** services.md initially asked for verbose per-field docstrings ("these become MCP tool descriptions"), and I leaned into that — multi-paragraph notes about value spaces, edge cases, and TM-mode caveats inline in IDL files. The user explicitly pushed back. The IDL is meant to be readable at a glance; verbose prose belongs in the package README or MCP tool descriptions hand-written elsewhere.

**How to apply:** when authoring or editing `.srv` / `.msg` / `.action` files in this repo (`utils/msgs/task_generator_msgs/` etc.):
- One `#` line above each field, terse, ≤80 chars.
- No multi-line blocks. No "typically copied from..." / "only meaningful for..." prose.
- Constants and types speak for themselves.
- If a field's semantics are non-obvious, document it in the package README, not inline.
