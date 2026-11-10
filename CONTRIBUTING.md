# Contributing

This page explains how to get a change into Arena and the conventions we follow. They apply to code, worlds and scenarios alike.

## Workflow

Every change goes into `jazzy` through a pull request, small fixes and docs included. Name your branch after the kind of change:

| Branch | For |
|---|---|
| `feature/<name>` | a new capability |
| `enhancement/<name>` | better existing behavior: tuning, performance, refactor, cleanup, docs |
| `bugfix/<name>` | a defect fixed |

If you change a submodule:

1. Commit in the submodule, on a branch with the same name, and open a PR there.
2. Commit the new submodule pin in the metarepo and open a metarepo PR.
3. Link the submodule PR to the metarepo PR.

Keep to one change/feature/system per PR.

Talk to a maintainer before you change an upstream dependency (the vcstool-managed repos under `src/deps`, HuNav).

Don't force-push or rewrite history that is already on GitHub.

Keep logs, notes and plans out of the repo.

## Before you push

Run these in the metarepo and make sure they pass:

```bash
pre-commit run --all-files
pre-commit run basedpyright --hook-stage manual
arena test
```

## Code

- Must pass `pre-commit run --all-files` (lint, format and annotation checks). Annotate all types. Avoid `Any` and `object` except for generic containers.
    - Add `# noqa` only with a specific reason.
- Access attributes directly. Instead of `hasattr` or `getattr(obj, "name", default)`, fix the type annotation. `getattr` is only for attribute names computed at runtime.
- For rclpy and asyncio glue (namespaces, lifecycle, parameters, time), use/extend [arena_rclpy_mixins](utils/arena_rclpy_mixins/arena_rclpy_mixins/README.md) rather than writing your own.
- Declare every node parameter as a `Param` (`declare` or a `ParamGroup`) and every launch argument as a `LaunchArgument`, generated with `LaunchArgument.from_param(param)` when a node parameter backs it. Each takes a description. The `param-census` hook rejects anything else.
- Declare every pip package your package imports in its own `pyproject.toml`. The commit hook updates the lock files.
- Namespace launch arguments and CLI keys with periods: `lockstep.rtf:=5`, `viz.view:=robot3p`.
- Packages must not depend on each other in a cycle at runtime (except via `exec_depend`).
- Derive repeated values (a footprint polygon, a radius, a path) from one source instead of copying them by hand.

## Comments and docs

- Write only necessary comments. Leave out why the code changed, what it used to do and how the math works. Keep comments for real constraints (`uint32 on the wire, so normalize into [0, 1e9)`) and sentinel values (`# 0 = auto`). CI checks the comment density of every PR.
- YAML, configs and tests can have more explanatory comments, as long as they are necessary.
- A docstring is one short line saying what the thing is or does, without `Parameters` or `Returns` blocks.
- In `.msg`, `.srv` and `.action` files, give each field one short line of at most 80 characters.
- If your change makes a doc out of date, update it in the same commit. Docs stay free of TODO lists. The `doc-links` hook checks relative links.
- Prefer ASCII and keep it human-written.

## Tests

- Test real code on real inputs, with `tmp_path` and real files. We don't use `unittest.mock`, `patch` or `monkeypatch`. If something can only be tested with a fake, leave the test out and mention the gap in the PR.
- Name a test after what it checks, not `test_smoke`, `test_basic` or `test_simple`.
- When a signature changes, update its tests rather than deleting them.