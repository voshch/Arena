"""Every first-party node parameter and launch argument is declared through params.py or a described launch argument."""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterator

_SKIP_DIRS = frozenset({"deps", "planners", "build", "install", "log", ".uv-workspace", "__pycache__", "test", "tests", ".git", ".claude", ".venv", "node_modules"})
_MECHANISM = ("arena_rclpy_mixins/params.py", "arena_rclpy_mixins/ROSParamServer.py")
_EXEMPT_FUNCTIONS = frozenset({"migrate_deprecated_params"})
_DECLARING_METHODS = frozenset({"declare_parameter", "declare_parameters", "set_descriptor", "declare_safe", "declare_forward"})
_LAUNCH_ARGUMENTS = frozenset({"LaunchArgument", "DeclareLaunchArgument"})


def _arena_root() -> pathlib.Path:
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "_meta").is_dir():
            return parent
    raise RuntimeError("the Arena tree (a parent holding _meta/) was not found above this test")


def _sources(root: pathlib.Path) -> Iterator[pathlib.Path]:
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if _SKIP_DIRS & set(relative.parts[:-1]) or path.name.startswith("test_"):
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

    def _flag(self, node: ast.AST, what: str) -> None:
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
        elif name in _LAUNCH_ARGUMENTS and _empty_text(_keyword(node, "description")):
            self._flag(node, f"{name} without a description")
        self.generic_visit(node)


def test_every_parameter_and_launch_argument_is_declared_and_described() -> None:
    root = _arena_root()
    offenders: list[str] = []
    for path in _sources(root):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        census = _Census(path, root)
        census.visit(tree)
        offenders.extend(census.offenders)
    assert not offenders, f"{len(offenders)} declarations outside params.py or without a description:\n" + "\n".join(sorted(offenders))
