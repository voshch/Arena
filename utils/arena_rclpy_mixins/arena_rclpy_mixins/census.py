"""Every first-party node parameter is declared through params.py and every launch argument is a described LaunchArgument."""

from __future__ import annotations

import ast
import pathlib
import sys
from collections.abc import Iterator

_SKIP_DIRS = frozenset({"deps", "planners", "build", "install", "log", "__pycache__", "test", "tests", "node_modules"})
_MECHANISM = ("arena_rclpy_mixins/params.py", "arena_rclpy_mixins/ROSParamServer.py")
_EXEMPT_FUNCTIONS = frozenset({"migrate_deprecated_params"})
_DECLARING_METHODS = frozenset({"declare_parameter", "declare_parameters", "set_descriptor", "declare_safe", "declare_forward"})


def _sources(root: pathlib.Path) -> Iterator[pathlib.Path]:
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        directories = relative.parts[:-1]
        if _SKIP_DIRS & set(directories) or any(part.startswith(".") for part in directories) or path.name.startswith("test_"):
            continue
        if str(relative).endswith(_MECHANISM):
            continue
        yield path


def _callee(call: ast.Call) -> tuple[str | None, ast.expr | None]:
    func = call.func
    if isinstance(func, ast.Subscript):
        func = func.value
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        return func.attr, func.value
    return None, None


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _empty_text(node: ast.expr | None) -> bool:
    return node is None or (isinstance(node, ast.Constant) and (not isinstance(node.value, str) or not node.value.strip()))


def _is_rosparam(receiver: ast.expr | None) -> bool:
    if isinstance(receiver, ast.Subscript):
        receiver = receiver.value
    return isinstance(receiver, ast.Attribute) and receiver.attr == "rosparam" or isinstance(receiver, ast.Name) and receiver.id == "rosparam"


class _Census(ast.NodeVisitor):
    def __init__(self, path: pathlib.Path, root: pathlib.Path) -> None:
        self.relative = path.relative_to(root)
        self.offenders: list[str] = []

    def _flag(self, node: ast.Call, what: str) -> None:
        self.offenders.append(f"{self.relative}:{node.lineno} {what}")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name not in _EXEMPT_FUNCTIONS:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name, receiver = _callee(node)
        if name in _DECLARING_METHODS and receiver is not None:
            self._flag(node, f"{name}() outside params.py")
        elif name == "get" and _is_rosparam(receiver) and len(node.args) >= 2:
            self._flag(node, "rosparam.get(name, default) declares without a description")
        elif name == "ROSParam" and receiver is not None:
            self._flag(node, "ROSParam outside params.py, declare a Param")
        elif name == "ParameterDescriptor":
            self._flag(node, "hand-built ParameterDescriptor, declare a Param")
        elif name == "Param" and isinstance(node.func, (ast.Subscript, ast.Name)):
            description = node.args[2] if len(node.args) > 2 else _keyword(node, "description")
            if _empty_text(description):
                self._flag(node, "Param without a description")
        elif name == "DeclareLaunchArgument":
            self._flag(node, "DeclareLaunchArgument, declare a LaunchArgument")
        elif name == "LaunchArgument" and _empty_text(_keyword(node, "description")):
            self._flag(node, "LaunchArgument without a description")
        self.generic_visit(node)


def offenders(root: pathlib.Path) -> list[str]:
    """Every declaration under root outside params.py or without a description, as path:line what."""
    found: list[str] = []
    for path in _sources(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        census = _Census(path, root)
        census.visit(tree)
        found.extend(census.offenders)
    return sorted(found)


def main() -> int:
    found = offenders(pathlib.Path.cwd())
    for offender in found:
        print(offender)
    if found:
        print(f"{len(found)} declarations outside params.py or without a description", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
