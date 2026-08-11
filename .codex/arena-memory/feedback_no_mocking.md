---
name: No mocking in tests
description: Don't write tests that use unittest.mock (Mock/MagicMock/AsyncMock/patch) or pytest monkeypatch — use real files, real callables, real inputs
type: feedback
originSessionId: 601e1eb5-74fe-4cff-96b2-ba038308d5b2
---
Never use `unittest.mock` (`Mock`, `MagicMock`, `AsyncMock`, `patch`, `patch.object`) or pytest's `monkeypatch` fixture in tests. If a test target genuinely requires a mock to exercise (e.g. xacro subprocess, Isaac/pxr, remote ROS node), skip writing that test entirely — don't fake coverage with mock-asserting tests that only verify the mock was called.

**Why:** user pushed back with "ehh no mocking" when I wrote heavily-mocked tests for URDF path resolution, Collada parsing, ModelWrapper loaders, and Material tint. Mock-asserting tests restate the implementation and break on every refactor while proving nothing about real behavior. The existing tests in `arena_simulation_setup/tests/unit/` use tmp_path with real files and invoke production code end-to-end — that's the expected shape.

**How to apply:** before writing a test, ask whether it can run against real production code with real inputs on tmp_path. If yes, write it. If no — if the only way to hit a code path is to patch a symbol or inject a Mock — skip the test and report the gap. Don't ship mock-heavy replacements. Also applies to patching module attributes via `monkeypatch.setattr` to stand in for paths, config, or globals — if the production code doesn't already accept a path parameter, the test doesn't belong.
