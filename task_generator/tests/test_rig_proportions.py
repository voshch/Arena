"""The pointing skeleton's bone lengths must match human_description/urdf/human-tpl.xacro."""

from __future__ import annotations

import ast
import operator
import re
from pathlib import Path

import pytest
from task_generator.simulators.human.pointing.skeleton import Body

_PROPERTY = re.compile(r'<xacro:property name="(\w+)" value="(.*?)"\s*/>')
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

# Body attribute -> xacro property
PROPORTIONS = {
    "size_unit": "size_unit",
    "head_radius": "head_radius",
    "neck_length": "neck_length",
    "neck_shoulder": "neck_shoulder_length",
    "upperarm": "upperarm_length",
    "forearm": "forearm_length",
    "torso_height": "torso_height",
    "spine_segment": "spine_segment",
    "waist_length": "waist_length",
    "thigh": "tight_length",
    "tibia": "tibia_length",
    "limb_radius": "limb_radius",
}


def _xacro_path() -> Path | None:
    try:
        from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

        try:
            return Path(get_package_share_directory("human_description")) / "urdf" / "human-tpl.xacro"
        except PackageNotFoundError:
            pass
    except ImportError:
        pass
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "deps" / "human_description" / "urdf" / "human-tpl.xacro"
        if candidate.is_file():
            return candidate
    return None


def _evaluate(expr: str, names: dict[str, float]) -> float:
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError(f"unsupported xacro expression {expr!r}")

    return walk(ast.parse(expr, mode="eval"))


def xacro_proportions(path: Path, height: float) -> dict[str, float]:
    names: dict[str, float] = {"height": height}
    for name, raw in _PROPERTY.findall(path.read_text()):
        if name == "height":
            continue
        expr = raw[2:-1] if raw.startswith("${") and raw.endswith("}") else raw
        try:
            names[name] = _evaluate(expr, names)
        except (ValueError, KeyError, SyntaxError):
            continue
    return names


@pytest.mark.parametrize("height", [1.65, 1.80])
def test_body_matches_xacro(height: float) -> None:
    path = _xacro_path()
    if path is None:
        pytest.skip("human_description xacro not found")
    props = xacro_proportions(path, height)
    body = Body(height)
    for attr, prop in PROPORTIONS.items():
        assert getattr(body, attr) == pytest.approx(props[prop], abs=1e-9), f"{attr} drifted from xacro {prop}"
