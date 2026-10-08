from __future__ import annotations

import math

import attrs
import pytest
from arena_rclpy_mixins.param_groups import Param, ParamGroup, PerRole, configure, declare_launch_arguments, floats, names, positive, positive_count, within


class _MotorGroup(ParamGroup):
    TRIM_DB = Param[float]("motor.trim_db", 0.0, description="Offset on the motor level.")
    RATE_HZ = Param[float]("motor.rate_hz", 20.0, parse=positive)
    GAIN = Param[float]("motor.gain", PerRole(array=0.75, listener=1.0), description="Gain per renderer role.")
    KINDS = Param[tuple[str, ...]]("motor.kinds", ["Wheel", " "], parse=names)


@attrs.frozen
class _MotorConfig:
    trim_db: float
    rate_hz: float
    kinds: tuple[str, ...]


def test_per_role_picks_by_role_name_and_coerces_on_the_first_value() -> None:
    gain = PerRole(array=0.75, listener=1.0)

    assert gain.pick("array") == 0.75
    assert gain.pick("listener") == 1.0
    assert gain.first == 0.75
    assert gain == PerRole(array=0.75, listener=1.0)
    with pytest.raises(KeyError, match="no default for role 'viewport'"):
        gain.pick("viewport")
    with pytest.raises(ValueError, match="at least one role"):
        PerRole()
    assert _MotorGroup.GAIN.coerce("0.5") == 0.5


def test_coercers_reject_out_of_range_values() -> None:
    assert within(0.0, 1.0)(1.0) == 1.0
    with pytest.raises(ValueError, match=r"outside \(0.0, 1.0\]"):
        within(0.0, 1.0, lo_open=True)(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        positive_count(0)
    with pytest.raises(ValueError, match="not finite"):
        floats([1.0, math.inf])


def test_list_param_refuses_an_empty_list_from_the_command_line() -> None:
    param = Param[tuple[str, ...]]("belief.kinds", ["footstep"], parse=names)
    assert param.coerce("[speech, onset]") == ["speech", "onset"]
    with pytest.raises(ValueError, match="belief.kinds cannot be an empty list"):
        param.coerce("[]")


def test_configure_from_group_defaults_leaves_role_dependent_ones_out() -> None:
    config = configure(_MotorConfig, _MotorGroup, rate_hz=10.0)

    assert config == _MotorConfig(trim_db=0.0, rate_hz=10.0, kinds=("wheel",))
    assert "gain" not in _MotorGroup.defaults()


def test_declare_launch_arguments_covers_described_params_only() -> None:
    pytest.importorskip("launch")
    params = {param.name: param for param in _MotorGroup.params()}

    arguments = declare_launch_arguments(params, prefix="auditory.", reserved=("motor.gain",))

    assert [argument.name for argument in arguments] == ["auditory.motor.trim_db"]
    assert arguments[0].description == "Offset on the motor level. Empty = node default."
