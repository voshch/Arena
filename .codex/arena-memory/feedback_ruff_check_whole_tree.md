---
name: ruff-check-whole-tree
description: "Verify lint with `ruff check .` from src/Arena root; never pass explicit file paths (bypasses excludes → false errors)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 789b1d5a-74d0-457c-a7ac-943a0805e11d
---

The bar is: `ruff check` must pass the **entirety** of src/Arena. Verify by running `ruff check .` (and `ruff format --check .`) from `/home/arena/arena_ws/src/Arena`.

**Why:** `tests/` and `**/launch` are in ruff's `extend-exclude` (pyproject.toml), along with `_meta`, `arena_tools`, `arena_isaac`, `arena_evaluation`, `arena_training`, `arena_robots/deps`. Passing explicit file paths to `ruff check <files>` bypasses excludes (ruff only auto-excludes during directory traversal), so excluded files report false errors (ANN missing-annotations, F401 unused-import, B017). The pre-commit/CI invocation uses the canonical traversal with `--force-exclude`, so it passes. I once surfaced "24 errors" that were all in excluded test/launch files and nearly acted on them.

**How to apply:** after edits, check lint with `ruff check .` from the Arena root, not a file list. Don't run `ruff format` on excluded test/launch files even when asked to "run ruff format" — canonical `ruff format .` skips them, so there is nothing to format there. See also [[feedback_no_hasattr]] (ruff check must be clean), [[feedback_no_smoke_tests]] (static analysis only).
