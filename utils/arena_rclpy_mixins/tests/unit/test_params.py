from __future__ import annotations

import enum
import math

import attrs
import pytest
from arena_rclpy_mixins.params import Catalog, Choices, Pair, Param, ParamGroup, PerRole, Range, Text, configure, constraint_json, floats, names, pair


class _Mode(enum.StrEnum):
    FAST = "fast"
    SLOW = "slow"


class _MotorGroup(ParamGroup):
    TRIM_DB = Param[float]("motor.trim_db", 0.0, "Offset on the motor level.")
    RATE_HZ = Param[float]("motor.rate_hz", 20.0, "Update rate.", constraint=Range(lo=0.0, lo_open=True))
    GAIN = Param[float]("motor.gain", PerRole(array=0.75, listener=1.0), "Gain per renderer role.")
    KINDS = Param[tuple[str, ...]]("motor.kinds", ["Wheel", " "], "Sound kinds.", parse=names)
    MODE = Param[_Mode]("motor.mode", _Mode.FAST, "Playback mode.")


@attrs.frozen
class _MotorConfig:
    trim_db: float
    rate_hz: float
    kinds: tuple[str, ...]


def test_param_needs_a_description() -> None:
    with pytest.raises(ValueError, match="needs a description"):
        Param[int]("x", 1, "")
    with pytest.raises(ValueError, match="needs a description"):
        Param[int]("x", 1, "   ")


def test_kind_follows_the_default() -> None:
    assert Param[bool]("b", True, "d").kind == "bool"
    assert Param[int]("i", 1, "d").kind == "int"
    assert Param[float]("f", 1.0, "d").kind == "float"
    assert Param[str]("s", "a", "d").kind == "str"
    assert Param[list[int]]("li", [1, 2], "d").kind == "int[]"
    assert Param[list[float]]("lf", [1, 2.5], "d").kind == "float[]"
    assert Param[list[str]]("ls", ("a",), "d").kind == "str[]"


def test_empty_list_and_none_defaults_need_a_type() -> None:
    rclpy = pytest.importorskip("rclpy")

    with pytest.raises(ValueError, match="pass type_"):
        Param[list[str]]("models", [], "d")
    with pytest.raises(ValueError, match="pass type_"):
        Param[str]("world", None, "d")
    assert Param[list[str]]("models", [], "d", type_=rclpy.Parameter.Type.STRING_ARRAY).kind == "str[]"
    required = Param[str]("world", None, "d", type_=rclpy.Parameter.Type.STRING)
    assert required.required
    assert required.ros_default() is None


def test_enum_default_gives_choices_and_parse() -> None:
    param = Param[_Mode]("mode", _Mode.SLOW, "d")

    assert param.kind == "str"
    assert param.constraint == Choices("fast", "slow")
    assert param.ros_default() == "slow"
    assert param.parsed_default() is _Mode.SLOW
    param.check("fast")
    with pytest.raises(ValueError, match="not in"):
        param.check("medium")


def test_constraint_must_fit_the_kind() -> None:
    with pytest.raises(ValueError, match="does not apply"):
        Param[str]("s", "a", "d", constraint=Range(0, 1))
    with pytest.raises(ValueError, match="does not apply"):
        Param[int]("i", 1, "d", constraint=Choices("a"))
    with pytest.raises(ValueError, match="does not apply"):
        Param[float]("f", 1.0, "d", constraint=Pair())


def test_range_checks_bounds_and_open_ends() -> None:
    closed = Param[float]("f", 0.5, "d", constraint=Range(0.0, 1.0))
    closed.check(0.0)
    closed.check(1.0)
    with pytest.raises(ValueError, match="outside"):
        closed.check(1.5)
    open_low = Param[float]("f", 0.5, "d", constraint=Range(0.0, lo_open=True))
    open_low.check(1e-9)
    with pytest.raises(ValueError, match=r"outside \(0.0, inf\)"):
        open_low.check(0.0)
    with pytest.raises(ValueError, match="is not int"):
        Param[int]("i", 1, "d", constraint=Range(0, 9)).check(True)
    with pytest.raises(ValueError, match="not a number"):
        Range(0, 9).check("5")


def test_floats_reject_nan_and_inf_unless_the_default_or_a_bound_is_infinite() -> None:
    plain = Param[float]("f", 1.0, "d")
    with pytest.raises(ValueError, match="not finite"):
        plain.check(math.nan)
    with pytest.raises(ValueError, match="not finite"):
        plain.check(math.inf)
    with pytest.raises(ValueError, match="not finite"):
        Param[list[float]]("lf", [1.0], "d").check([1.0, math.nan])
    sentinel = Param[float]("f", math.nan, "d")
    sentinel.check(math.nan)
    sentinel.check(2.0)
    to_infinity = Param[float]("f", 1.0, "d", constraint=Range(0.0, math.inf))
    to_infinity.check(math.inf)
    with pytest.raises(ValueError, match="outside"):
        to_infinity.check(-1.0)


def test_choices_single_and_many() -> None:
    single = Param[str]("s", "a", "d", constraint=Choices("a", "b"))
    single.check("b")
    with pytest.raises(ValueError, match="not in"):
        single.check("a,b")
    many = Param[str]("s", "a,b", "d", constraint=Choices("a", "b", many=True))
    many.check("a,b")
    many.check("")
    with pytest.raises(ValueError, match=r"\['c'\] not in"):
        many.check("a,c")
    with pytest.raises(ValueError):
        Choices()


def test_pair_accepts_one_or_two_numbers_within_bounds() -> None:
    param = Param[tuple[int, int]]("n", [0, 0], "d", constraint=Pair(lo=0, hi=10), parse=pair)
    param.check([3])
    param.check([5, 2])
    with pytest.raises(ValueError, match=r"not a \[min, max\] pair"):
        param.check([1, 2, 3])
    with pytest.raises(ValueError, match="outside"):
        param.check([0, 11])
    assert pair([5, 2]) == (2, 5)
    assert pair([3]) == (3, 3)


def test_type_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="is not int"):
        Param[int]("i", 1, "d").check(1.5)
    with pytest.raises(ValueError, match="is not str"):
        Param[str]("s", "a", "d").check(1)
    Param[float]("f", 1.0, "d").check(2)


def test_constraint_json_shapes() -> None:
    assert constraint_json(Range(0, None, 2, lo_open=True)) == {"kind": "range", "lo": 0, "hi": None, "step": 2, "lo_open": True, "hi_open": False}
    assert constraint_json(Range(0.0, math.inf)) == {"kind": "range", "lo": 0.0, "hi": None, "step": 0.0, "lo_open": False, "hi_open": False}
    assert constraint_json(Choices("a", "b", many=True)) == {"kind": "choices", "values": ["a", "b"], "many": True}
    assert constraint_json(Catalog("worlds")) == {"kind": "catalog", "name": "worlds"}
    assert constraint_json(Pair(lo=1)) == {"kind": "pair", "lo": 1, "hi": None}
    assert constraint_json(Text(sketch=True)) == {"kind": "text", "sketch": True}


def test_coerce_turns_launch_strings_into_ros_values() -> None:
    assert Param[bool]("b", False, "d").coerce("true") is True
    assert Param[int]("i", 0, "d").coerce("7") == 7
    assert Param[float]("f", 0.0, "d").coerce("0.5") == 0.5
    assert Param[str]("s", "", "d").coerce("[a]") == "[a]"
    assert Param[list[float]]("lf", [1.0], "d").coerce("[1, 2.5]") == [1.0, 2.5]
    assert Param[list[str]]("ls", ["x"], "d").coerce("a") == ["a"]
    with pytest.raises(ValueError, match="expects true or false"):
        Param[bool]("b", False, "d").coerce("yes please")
    with pytest.raises(ValueError, match="empty list"):
        Param[list[str]]("ls", ["x"], "d").coerce("[]")


def test_per_role_picks_by_role_name() -> None:
    gain = PerRole(array=0.75, listener=1.0)

    assert gain.pick("array") == 0.75
    assert gain.first == 0.75
    with pytest.raises(KeyError):
        gain.pick("other")
    with pytest.raises(ValueError):
        PerRole()
    assert _MotorGroup.GAIN.ros_default("listener") == 1.0
    with pytest.raises(TypeError, match="needs a role"):
        _MotorGroup.GAIN.ros_default()


def test_at_prefixes_the_name_and_keeps_the_declaration() -> None:
    param = _MotorGroup.RATE_HZ.at("task.random")

    assert param.name == "task.random.motor.rate_hz"
    assert param.constraint == _MotorGroup.RATE_HZ.constraint
    assert param.attr == "RATE_HZ"
    assert _MotorGroup.RATE_HZ.at("") is _MotorGroup.RATE_HZ
    assert _MotorGroup.RATE_HZ.at(None) is _MotorGroup.RATE_HZ


def test_group_class_lists_params_defaults_and_names() -> None:
    assert [param.attr for param in _MotorGroup.params()] == ["TRIM_DB", "RATE_HZ", "GAIN", "KINDS", "MODE"]
    assert _MotorGroup.defaults() == {"trim_db": 0.0, "rate_hz": 20.0, "kinds": ("wheel",), "mode": _Mode.FAST}
    assert list(_MotorGroup.by_name("ns")) == ["ns.motor.trim_db", "ns.motor.rate_hz", "ns.motor.gain", "ns.motor.kinds", "ns.motor.mode"]
    assert configure(_MotorConfig, _MotorGroup) == _MotorConfig(trim_db=0.0, rate_hz=20.0, kinds=("wheel",))
    assert configure(_MotorConfig, _MotorGroup, rate_hz=5.0).rate_hz == 5.0


def test_parse_helpers() -> None:
    assert floats([1, 2.5]) == (1.0, 2.5)
    assert names(["Wheel", " ", " Fan "]) == ("wheel", "fan")


def test_declare_launch_arguments_emits_every_param_with_choices() -> None:
    pytest.importorskip("launch")
    from arena_rclpy_mixins.params import declare_launch_arguments

    arguments = declare_launch_arguments(_MotorGroup.by_name(), prefix="motor_", reserved=("motor.gain",))

    assert [argument.name for argument in arguments] == ["motor_motor.trim_db", "motor_motor.rate_hz", "motor_motor.kinds", "motor_motor.mode"]
    assert arguments[0].description == "Offset on the motor level. Empty = node default."
    assert arguments[-1].choices == ["", "fast", "slow"]
    assert arguments[0].choices is None
