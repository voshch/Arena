---
name: arena CLI lives in _meta/tools/source
description: The `arena` command is a thin bash shim in Arena/_meta/tools/source dispatching to a click CLI in _meta/tools/arena_cli/; add verbs in cli.py, not the case statement.
type: project
originSessionId: fa20c73a-02fb-488d-9a75-d5fb541e2ae3
modified: 2026-07-21T18:53:09.873Z
---
The `arena` command is split (since 2026-07-21): `Arena/_meta/tools/source` keeps env setup, venv bootstrap, feature source hooks, and a thin `arena()` shim whose case handles only `build`/`rebuild` (re-source after), `repair` (in-container uv venv+sync, /.dockerenv-guarded), `deactivate`, `resource`. Everything else goes through `_arena_cli` → `python -E -S _meta/tools/arena_cli/__main__.py` (click 8, pinned in root pyproject).

**Why:** Shell can't be replaced for env-mutating verbs; everything else lives in Python for maintainable help and logic. `-E -S` + lazy stdlib imports keep dispatch ~50ms; `__main__.py` exits 113 on ImportError so the shim can hint `arena repair`.

**How to apply:** New verbs = new `@main.command` in `_meta/tools/arena_cli/cli.py` with `PASSTHROUGH | HELP_NAMES` context and `click.UNPROCESSED` args for `key:=value` forwarding. Shared plumbing in `common.py`, in-process ports of the old tools in `viz.py`/`human.py`/`robot.py`/`build.py`/`pull.py`/`testsum.py` (originals under `_meta/tools/` are dead code kept as fallback authority). Features: `arena_cli/features/<name>.py` exports `group` + `SCRIPT_SHA256` of the bash script it ports; `features.load()` uses the Python group only while the deployed script's hash matches, else falls back to `FeatureSubGroup` subshell dispatch (this protects container-local script variants like isaac-with-exec, docker, vllm). Don't add verbs to the bash case unless they must mutate the calling shell.
