---
name: arena-cli-no-uv
description: "arena CLI is pure stdlib on bare python3, uv must NEVER be in the verb path (cache lock contention)"
metadata: 
  node_type: memory
  type: project
  originSessionId: edfe0404-d61a-4251-94a5-f6104c1f85f1
  modified: 2026-08-04T21:33:38.878Z
---

The arena CLI shim runs `python3 _meta/tools/arena_cli/__main__.py`, pure stdlib (click was vendored briefly, then rejected per [[no-vendored-deps]] and replaced by the Verb/make_verb/CLIError framework in common.py with the dispatcher in cli.py). PYTHONPATH passes through untouched, no stash and no extend_ros_path: the CLI execs children (supervisor, ros2 launch) that need the workspace overlay, and clearing PYTHONPATH for the CLI broke them (no module named arena_bringup, 2026-08-05). The CLI's local imports are safe because the script dir precedes PYTHONPATH on sys.path. uv appears ONLY in venv provisioning (`_arena_venv_provision`, `arena update`, `arena repair`). Never route CLI verbs through `uv run`: uv takes an exclusive UV_CACHE_DIR lock even for warm zero-dep script envs, and on fresh installs the container's first-boot block runs a multi-GB `uv sync` in the background holding that lock, so any uv-based verb hangs silently for the whole sync (reproduced 2026-08-04, this was the "arena hangs on fresh install" bug). Also: compose masks the image-baked /opt/venv with the host's empty .venv on fresh installs, so nothing baked at image build is visible at runtime, and runtime pypi fetches hang unboundedly on stalled networks.

**Why:** three independent failure modes (lock contention, masked baked venv, network stalls) all funnel through runtime uv/venv dependence.

**How to apply:** keep the CLI stdlib-only at runtime, no third-party imports anywhere in _meta/tools/arena_cli. The ONLY provisioning serialization is the entrypoint first-boot flock (/tmp/arena-first-boot.lock), `_arena_venv_provision` itself is lock-free by user decision (manual double-repair degrades to uv's internal serialization). If a lock is ever reintroduced: never inside the venv (uv venv clobbers foreign files at creation), never under $TMPDIR (in-container TMPDIR=/tmp/shared is a root-owned cross-service volume, writes fail) ([[host-python-314-zombie-venv]], [[shared-venv-provisioning]]).
