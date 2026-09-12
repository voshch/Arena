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
def test_python_expression_parse_zero_args_raises() -> None:
    from arena_bringup.future import PythonExpression

    with pytest.raises(TypeError):
        PythonExpression.parse([])


@_skip
def test_python_expression_parse_three_args_raises() -> None:
    from arena_bringup.future import PythonExpression

    with pytest.raises(TypeError):
        PythonExpression.parse(["a", "b", "c"])


@_skip
def test_python_expression_perform_simple_arithmetic() -> None:
    from arena_bringup.future import PythonExpression

    ctx = _make_context()
    expr = PythonExpression("2 + 2")
    result = expr.perform(ctx)
    assert result == "4"


@_skip
def test_python_expression_perform_math_module() -> None:
    from arena_bringup.future import PythonExpression

    ctx = _make_context()
    expr = PythonExpression("math.sqrt(4)", python_modules=["math"])
    result = expr.perform(ctx)
    assert result == "2.0"


@_skip
def test_python_expression_perform_math_implicit() -> None:
    from arena_bringup.future import PythonExpression

    ctx = _make_context()
    expr = PythonExpression("sqrt(4)")
    result = expr.perform(ctx)
    assert result == "2.0"


@_skip
def test_python_expression_perform_string_result() -> None:
    from arena_bringup.future import PythonExpression

    ctx = _make_context()
    expr = PythonExpression("'hello'")
    result = expr.perform(ctx)
    assert result == "hello"


@_skip
def test_python_expression_expression_property() -> None:
    from arena_bringup.future import PythonExpression
    import launch.substitutions

    expr = PythonExpression("1 + 2")
    assert isinstance(expr.expression, list)
    assert all(isinstance(s, launch.substitutions.TextSubstitution) for s in expr.expression)


@_skip
def test_python_expression_python_modules_property() -> None:
    from arena_bringup.future import PythonExpression
    import launch.substitutions

    expr = PythonExpression("1", python_modules=["math"])
    assert isinstance(expr.python_modules, list)
    assert all(isinstance(s, launch.substitutions.TextSubstitution) for s in expr.python_modules)


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
def test_if_else_substitution_true_condition_returns_if_value() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="true", if_value="yes", else_value="no")
    assert sub.perform(ctx) == "yes"


@_skip
def test_if_else_substitution_false_condition_returns_else_value() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="false", if_value="yes", else_value="no")
    assert sub.perform(ctx) == "no"


@_skip
def test_if_else_substitution_1_is_truthy() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="1", if_value="yes", else_value="no")
    assert sub.perform(ctx) == "yes"


@_skip
def test_if_else_substitution_0_is_falsy() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="0", if_value="yes", else_value="no")
    assert sub.perform(ctx) == "no"


@_skip
def test_if_else_substitution_true_uppercase() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="True", if_value="yes", else_value="no")
    assert sub.perform(ctx) == "yes"


@_skip
def test_if_else_substitution_false_uppercase() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="False", if_value="yes", else_value="no")
    assert sub.perform(ctx) == "no"


@_skip
def test_if_else_substitution_only_if_value_specified() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="true", if_value="only_if")
    assert sub.perform(ctx) == "only_if"


@_skip
def test_if_else_substitution_only_else_value_specified() -> None:
    from arena_bringup.future import IfElseSubstitution

    ctx = _make_context()
    sub = IfElseSubstitution(condition="false", else_value="only_else")
    assert sub.perform(ctx) == "only_else"


@_skip
def test_if_else_substitution_condition_property() -> None:
    from arena_bringup.future import IfElseSubstitution

    sub = IfElseSubstitution(condition="true", if_value="a")
    assert isinstance(sub.condition, list)


@_skip
def test_if_else_substitution_if_value_property() -> None:
    from arena_bringup.future import IfElseSubstitution

    sub = IfElseSubstitution(condition="true", if_value="a")
    assert isinstance(sub.if_value, list)


@_skip
def test_if_else_substitution_else_value_property() -> None:
    from arena_bringup.future import IfElseSubstitution

    sub = IfElseSubstitution(condition="true", if_value="a", else_value="b")
    assert isinstance(sub.else_value, list)
