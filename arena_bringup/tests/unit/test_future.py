from __future__ import annotations

import pytest

_SKIP_REASON = "launch package not available (source ROS)"


def _launch_available() -> bool:
    try:
        import launch.actions  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


_skip = pytest.mark.skipif(not _launch_available(), reason=_SKIP_REASON)


def _make_context():
    import launch

    return launch.LaunchContext()


@_skip
def test_python_expression_parse_one_arg() -> None:
    from arena_bringup.future import PythonExpression

    cls, kwargs = PythonExpression.parse(["1 + 1"])
    assert cls is PythonExpression
    assert "expression" in kwargs


@_skip
def test_python_expression_parse_two_args() -> None:
    import launch.substitutions
    from arena_bringup.future import PythonExpression

    module_sub = launch.substitutions.TextSubstitution(text="math")
    cls, kwargs = PythonExpression.parse(["sqrt(4)", [module_sub]])
    assert cls is PythonExpression
    assert "python_modules" in kwargs


@_skip
@pytest.mark.parametrize("args", [[], ["a", "b", "c"]])
def test_python_expression_parse_wrong_arity_raises(args: list[str]) -> None:
    from arena_bringup.future import PythonExpression

    with pytest.raises(TypeError):
        PythonExpression.parse(args)


@_skip
@pytest.mark.parametrize(
    ("expression", "kwargs", "expected"),
    [
        ("2 + 2", {}, "4"),
        ("math.sqrt(4)", {"python_modules": ["math"]}, "2.0"),
        ("sqrt(4)", {}, "2.0"),
        ("'hello'", {}, "hello"),
    ],
)
def test_python_expression_perform(expression: str, kwargs: dict, expected: str) -> None:
    from arena_bringup.future import PythonExpression

    assert PythonExpression(expression, **kwargs).perform(_make_context()) == expected


@_skip
def test_python_expression_normalizes_to_substitutions() -> None:
    from arena_bringup.future import PythonExpression
    import launch.substitutions

    expr = PythonExpression("1 + 2", python_modules=["math"])
    assert all(isinstance(s, launch.substitutions.TextSubstitution) for s in [*expr.expression, *expr.python_modules])
    assert expr.expression and expr.python_modules


@_skip
def test_python_expression_describe() -> None:
    from arena_bringup.future import PythonExpression

    expr = PythonExpression("1 + 1")
    desc = expr.describe()
    assert "PythonExpr" in desc


@_skip
def test_if_else_substitution_both_empty_raises() -> None:
    from arena_bringup.future import IfElseSubstitution

    with pytest.raises(RuntimeError):
        IfElseSubstitution(condition="true", if_value="", else_value="")


@_skip
@pytest.mark.parametrize(
    ("condition", "expected"),
    [("true", "yes"), ("True", "yes"), ("1", "yes"), ("false", "no"), ("False", "no"), ("0", "no")],
)
def test_if_else_substitution_picks_the_branch_of_the_condition(condition: str, expected: str) -> None:
    from arena_bringup.future import IfElseSubstitution

    assert IfElseSubstitution(condition=condition, if_value="yes", else_value="no").perform(_make_context()) == expected


@_skip
@pytest.mark.parametrize(
    ("condition", "kwargs", "expected"),
    [("true", {"if_value": "only_if"}, "only_if"), ("false", {"else_value": "only_else"}, "only_else")],
)
def test_if_else_substitution_with_one_value_specified(condition: str, kwargs: dict, expected: str) -> None:
    from arena_bringup.future import IfElseSubstitution

    assert IfElseSubstitution(condition=condition, **kwargs).perform(_make_context()) == expected


@_skip
def test_if_else_substitution_normalizes_to_lists() -> None:
    from arena_bringup.future import IfElseSubstitution

    sub = IfElseSubstitution(condition="true", if_value="a", else_value="b")
    assert isinstance(sub.condition, list)
    assert isinstance(sub.if_value, list)
    assert isinstance(sub.else_value, list)
