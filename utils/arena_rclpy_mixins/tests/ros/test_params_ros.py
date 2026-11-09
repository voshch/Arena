from __future__ import annotations

import enum
import math
import typing

import pytest
from arena_rclpy_mixins.params import Catalog, Choices, Pair, Param, ParamGroup, Range, Text, declare, pair, parse_constraint

if typing.TYPE_CHECKING:
    import rclpy.node
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer


class _Side(enum.StrEnum):
    LEFT = "left"
    RIGHT = "right"


class _Group(ParamGroup):
    N = Param[tuple[int, int]]("n", [1, 3], "Count range.", constraint=Pair(lo=0, hi=10), parse=pair)
    GAIN = Param[float]("gain", 0.5, "Gain.", constraint=Range(0.0, 1.0))
    SIDE = Param[_Side]("side", _Side.LEFT, "Side.")


def _server(**overrides: object) -> ROSParamServer:
    import rclpy
    from arena_rclpy_mixins.ROSParamServer import ROSParamServer

    return ROSParamServer("params_probe", parameter_overrides=[rclpy.Parameter(name, value=value) for name, value in overrides.items()], automatically_declare_parameters_from_overrides=True)


def _plain(**overrides: object) -> rclpy.node.Node:
    import rclpy
    import rclpy.node

    return rclpy.node.Node("params_plain_probe", parameter_overrides=[rclpy.Parameter(name, value=value) for name, value in overrides.items()], automatically_declare_parameters_from_overrides=True)


@pytest.mark.parametrize(
    ("param", "ros_type", "tokens"),
    [
        (Param[tuple[int, int]]("n", [5, 15], "d", constraint=Pair()), "INTEGER_ARRAY", "range:int_pair"),
        (Param[tuple[float, float]]("r", [0.0, 1.0], "d", constraint=Pair(lo=0.0)), "DOUBLE_ARRAY", "range:float_pair;lo:0.0"),
        (Param[str]("f", "default", "d", constraint=Catalog("scenarios")), "STRING", "catalog:scenarios"),
        (Param[list[str]]("m", ["a"], "d", constraint=Catalog("objects")), "STRING_ARRAY", "catalog:objects"),
        (Param[str]("g", "arena", "d", constraint=Choices("arena", "behavior_tree")), "STRING", "enum:arena,behavior_tree"),
        (Param[str]("axes", "a,b", "d", constraint=Choices("a", "b", many=True)), "STRING", "enum:a,b;many"),
        (Param[_Side]("side", _Side.LEFT, "d"), "STRING", "enum:left,right"),
        (Param[str]("p", "x", "d", constraint=Text()), "STRING", "text"),
        (Param[str]("s", "x", "d", constraint=Text(sketch=True)), "STRING", "sketch"),
        (Param[str]("plain", "x", "d"), "STRING", ""),
        (Param[bool]("b", True, "d"), "BOOL", ""),
        (Param[float]("open", 1.0, "d", constraint=Range(0.0, lo_open=True)), "DOUBLE", "open:lo"),
    ],
)
def test_descriptor_type_description_and_tokens(param: Param[object], ros_type: str, tokens: str) -> None:
    import rclpy

    descriptor = param.descriptor()

    assert descriptor.type == rclpy.Parameter.Type[ros_type].value
    assert descriptor.description == "d"
    assert descriptor.additional_constraints == tokens


def test_descriptor_ranges_follow_the_kind_and_unbounded_ends() -> None:
    integer = Param[int]("c", 5, "d", constraint=Range(0, 10, 2)).descriptor()
    assert [(r.from_value, r.to_value, r.step) for r in integer.integer_range] == [(0, 10, 2)]
    assert integer.floating_point_range == []
    double = Param[float]("g", 0.5, "d", constraint=Range(0.0, 1.0, 0.1)).descriptor()
    assert [(r.from_value, r.to_value, r.step) for r in double.floating_point_range] == [(0.0, 1.0, 0.1)]
    half_open = Param[float]("h", 1.0, "d", constraint=Range(lo=0.0)).descriptor()
    assert half_open.floating_point_range[0].to_value == math.inf
    lower_only = Param[int]("k", 1, "d", constraint=Range(hi=9)).descriptor()
    assert lower_only.integer_range[0].from_value == -(2**63)
    assert Param[int]("free", 1, "d").descriptor().integer_range == []


@pytest.mark.parametrize(
    "param",
    [
        Param[tuple[int, int]]("n", [5, 15], "d", constraint=Pair(lo=0, hi=20)),
        Param[tuple[float, float]]("r", [0.0, 1.0], "d", constraint=Pair()),
        Param[str]("f", "default", "d", constraint=Catalog("scenarios")),
        Param[str]("g", "arena", "d", constraint=Choices("arena", "behavior_tree")),
        Param[str]("axes", "a,b", "d", constraint=Choices("a", "b", many=True)),
        Param[str]("p", "x", "d", constraint=Text()),
        Param[str]("s", "x", "d", constraint=Text(sketch=True)),
        Param[int]("c", 5, "d", constraint=Range(0, 10, 2)),
        Param[int]("k", 1, "d", constraint=Range(hi=9)),
        Param[float]("g", 0.5, "d", constraint=Range(0.0, 1.0, 0.1, lo_open=True, hi_open=True)),
        Param[float]("h", 1.0, "d", constraint=Range(lo=0.0)),
        Param[str]("plain", "x", "d"),
    ],
)
def test_parse_constraint_round_trips_every_encoding(param: Param[object]) -> None:
    assert parse_constraint(param.descriptor()) == param.constraint


def test_declare_on_a_plain_node_describes_and_validates() -> None:
    from rclpy import Parameter

    node = _plain()
    try:
        gain = declare(node, Param[float]("gain", 0.5, "Gain.", constraint=Range(0.0, 1.0)))
        side = declare(node, Param[_Side]("side", _Side.LEFT, "Side."))
        described = node.describe_parameters(["gain", "side"])
        assert described[0].description == "Gain."
        assert described[0].floating_point_range[0].to_value == 1.0
        assert described[1].additional_constraints == "enum:left,right"
        assert gain.value == 0.5
        assert side.value is _Side.LEFT
        assert node.set_parameters([Parameter("side", value="right")])[0].successful
        assert side.value is _Side.RIGHT
        rejected = node.set_parameters([Parameter("side", value="up")])[0]
        assert not rejected.successful
        assert "side" in rejected.reason
        assert node.get_parameter("side").value == "right"
        assert not node.set_parameters([Parameter("gain", value=math.nan)])[0].successful
        gain.destroy()
        assert not node.has_parameter("gain")
    finally:
        node.destroy_node()


def test_declare_on_a_server_gives_a_live_handle_that_parses_lazily() -> None:
    calls: list[object] = []

    def parse(value: object) -> int:
        calls.append(value)
        return int(value) * 2  # pyright: ignore[reportArgumentType]

    node = _server()
    try:
        handle = declare(node, Param[int]("twice", 2, "Doubled.", parse=parse))
        assert calls == []
        assert handle.value == 4
        assert handle.value == 4
        assert len(calls) == 1
        handle.value = 5
        assert node.get_parameter("twice").value == 5
        assert handle.value == 10
        assert len(calls) == 2
    finally:
        node.destroy_node()


def test_declaring_twice_returns_the_same_handle_and_one_validator() -> None:
    from rclpy import Parameter

    node = _server()
    try:
        first = declare(node, Param[int]("once", 1, "d", constraint=Range(0, 5)))
        second = declare(node, Param[int]("once", 1, "d", constraint=Range(0, 5)))
        assert first is second
        rejected = node.set_parameters([Parameter("once", value=9)])[0]
        assert not rejected.successful
        assert "\n" not in rejected.reason
        rejected = node.set_parameters([Parameter("once", value=-1)])[0]
        assert not rejected.successful
    finally:
        node.destroy_node()


def test_integer_override_of_a_double_param_is_coerced_and_kept() -> None:
    import rclpy

    node = _server(pos_x=10, pos_y=2.5)
    try:
        x = declare(node, Param[float]("pos_x", math.nan, "d"))
        y = declare(node, Param[float]("pos_y", math.nan, "d"))
        theta = declare(node, Param[float]("pos_theta", 0.75, "d"))
        assert node.get_parameter("pos_x").type_ == rclpy.Parameter.Type.DOUBLE
        assert x.value == 10.0
        assert y.value == 2.5
        assert theta.value == 0.75
        assert node.describe_parameters(["pos_x"])[0].description == "d"
    finally:
        node.destroy_node()


def test_an_override_that_violates_the_constraint_fails_at_declare() -> None:
    node = _server(gain=7.0)
    try:
        with pytest.raises(RuntimeError, match=r"parameter gain: .*7.0"):
            declare(node, Param[float]("gain", 0.5, "d", constraint=Range(0.0, 1.0)))
    finally:
        node.destroy_node()
    node = _server(side="up")
    try:
        with pytest.raises(RuntimeError, match=r"parameter side: 'up' not in \['left', 'right'\]"):
            declare(node, Param[_Side]("side", _Side.LEFT, "d"))
    finally:
        node.destroy_node()


def test_required_param_reads_its_override_and_raises_while_unset() -> None:
    import rclpy

    with_override = _server(world="map_empty")
    try:
        world = declare(with_override, Param[str]("world", None, "d", type_=rclpy.Parameter.Type.STRING))
        assert world.value == "map_empty"
    finally:
        with_override.destroy_node()
    without = _server()
    try:
        world = declare(without, Param[str]("world", None, "d", type_=rclpy.Parameter.Type.STRING))
        with pytest.raises(ValueError, match="unset"):
            _ = world.value
        assert without.describe_parameters(["world"])[0].description == "d"
    finally:
        without.destroy_node()


def test_array_values_arrive_as_arrays_from_the_wire_and_still_validate() -> None:
    from rcl_interfaces.msg import Parameter as ParameterMsg
    from rcl_interfaces.msg import ParameterValue
    from rclpy import Parameter

    node = _server()
    try:
        counts = declare(node, Param[tuple[int, int]]("n", [1, 3], "d", constraint=Pair(lo=0, hi=10), parse=pair))
        gains = declare(node, Param[list[float]]("gains", [0.5], "d"))
        wire = Parameter.from_parameter_msg(ParameterMsg(name="n", value=ParameterValue(type=7, integer_array_value=[9, 2])))
        assert node.set_parameters([wire])[0].successful
        assert counts.value == (2, 9)
        wire = Parameter.from_parameter_msg(ParameterMsg(name="n", value=ParameterValue(type=7, integer_array_value=[1, 2, 3])))
        rejected = node.set_parameters([wire])[0]
        assert not rejected.successful
        assert "pair" in rejected.reason
        wire = Parameter.from_parameter_msg(ParameterMsg(name="gains", value=ParameterValue(type=8, double_array_value=[0.25, math.nan])))
        assert not node.set_parameters([wire])[0].successful
        wire = Parameter.from_parameter_msg(ParameterMsg(name="gains", value=ParameterValue(type=8, double_array_value=[0.25, 0.75])))
        assert node.set_parameters([wire])[0].successful
        assert list(gains.value) == [0.25, 0.75]
    finally:
        node.destroy_node()


def test_read_only_param_keeps_its_default_and_refuses_sets() -> None:
    from rclpy import Parameter

    node = _server()
    try:
        side = declare(node, Param[_Side]("side", _Side.RIGHT, "d", read_only=True))
        assert side.value is _Side.RIGHT
        assert node.describe_parameters(["side"])[0].read_only
        assert not node.set_parameters([Parameter("side", value="left")])[0].successful
        rate = declare(node, Param[float]("rate", 15, "d", read_only=True))
        assert rate.value == 15.0
    finally:
        node.destroy_node()


def test_empty_array_default_declares_its_type() -> None:
    import rclpy

    node = _server()
    try:
        models = declare(node, Param[list[str]]("models", [], "d", type_=rclpy.Parameter.Type.STRING_ARRAY, constraint=Catalog("objects")))
        parameter = node.get_parameter("models")
        assert parameter.type_ == rclpy.Parameter.Type.STRING_ARRAY
        assert list(parameter.value) == []
        assert models.value == []
    finally:
        node.destroy_node()


def test_group_declares_under_its_namespace_and_is_one_instance_per_node() -> None:
    from rclpy import Parameter

    node = _server(**{"task.random.n": [4]})
    try:
        group = _Group(node, namespace="task.random")
        again = _Group(node, namespace="task.random")
        other = _Group(node, namespace="task.other")
        assert group is again
        assert other is not group
        assert sorted(node.list_parameters(["task.random"], 10).names) == ["task.random.gain", "task.random.n", "task.random.side"]
        assert group.N.value == (4, 4)
        assert group.values() == {"n": (4, 4), "gain": 0.5, "side": _Side.LEFT}
        assert node.set_parameters([Parameter("task.random.n", value=[7, 2])])[0].successful
        assert group.N.value == (2, 7)
        assert not node.set_parameters([Parameter("task.random.n", value=[1, 2, 3])])[0].successful
        assert node.describe_parameters(["task.other.gain"])[0].description == "Gain."
    finally:
        node.destroy_node()


def test_rclpy_enforces_ranges_itself_after_set_descriptor() -> None:
    from rclpy import Parameter

    node = _server(level=3)
    try:
        declare(node, Param[int]("level", 1, "d", constraint=Range(0, 5)))
        node._arena_params.params.clear()
        for callbacks in node._ROSParamServer__callbacks.values():
            callbacks.clear()
        assert not node.set_parameters([Parameter("level", value=9)])[0].successful
        assert node.set_parameters([Parameter("level", value=4)])[0].successful
    finally:
        node.destroy_node()
