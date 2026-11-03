from __future__ import annotations

import ast
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parents[2] / "task_generator"
_ALLOWED = {"arena_runtime.env.task_layer", "arena_runtime.env.task_api", "arena_runtime.shared", "arena_runtime_msgs"}


def _allowed(module: str) -> bool:
    return any(module == allowed or module.startswith(f"{allowed}.") for allowed in _ALLOWED)


def _runtime_imports(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.lineno, node.module


def test_task_generator_reaches_runtime_only_through_seam_modules():
    sources = sorted(_PACKAGE.rglob("*.py"))
    assert sources
    offenders = [f"{path.relative_to(_PACKAGE.parent)}:{line} imports {module}" for path in sources for line, module in _runtime_imports(ast.parse(path.read_text(), filename=str(path))) if (module == "arena_runtime" or module.startswith("arena_runtime.")) and not _allowed(module)]
    assert not offenders, "\n".join(offenders)
