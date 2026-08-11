---
name: Uncommitted is fully malleable
description: Treat uncommitted/staged changes as freely modifiable; no soft-deprecation, no backward-compat shims for them
type: feedback
originSessionId: 74f7497b-8718-44e3-ac29-cfdfe07c5d8f
---
When refactoring or replanning around code that exists only in uncommitted (staged or modified) state, treat it as fully changeable: rename, restructure, or delete outright. Do not propose stub aliases, deprecation shims, transition layers, or "keep the old name for back-compat" for files/symbols that have not yet been committed.

**Why:** Uncommitted code has zero external consumers by definition. The user is the only producer and consumer. Backward-compat layers add code without serving anyone, and they obscure the intended end state.

**How to apply:** Before suggesting a soft-deprecation or stub-passthrough for any file/launch/symbol, check `git status` / `git log` to see whether it's actually been committed. If it's only in the index or working tree, recommend the clean cut. Reserve transition layers for symbols that downstream users (committed callers, external repos, published packages) actually depend on.
