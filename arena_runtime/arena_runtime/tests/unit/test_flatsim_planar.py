"""Drive model message built from the planar block of a mobile cap."""

import math

import pytest


def _jackal():
    from arena_robots.caps import PlanarSpec

    return PlanarSpec.from_dict(
        {
            'velocity': {'linear': [-2.0, 2.0], 'lateral': [0.0, 0.0], 'angular': [-4.0, 4.0]},
            'acceleration': {'linear': 20.0, 'angular': 25.0},
            'gain': {'angular': 1.23},
            'traction': 7.85,
            'pivot': 0.131,
            'timeout': 0.25,
        }
    )


def _axis(axis):
    return (axis.low, axis.high, axis.accel)


def test_spec_sets_every_axis_and_leaves_missing_ones_unbounded():
    pytest.importorskip('flatsim_msgs')
    from arena_runtime.sim.flatsim_simulator.planar import drive_model

    model = drive_model(_jackal(), holonomic=False)
    assert _axis(model.linear) == (-2.0, 2.0, 20.0)
    assert _axis(model.lateral) == (0.0, 0.0, math.inf)
    assert _axis(model.angular) == (-4.0, 4.0, 25.0)
    assert list(model.gain) == [1.0, 1.0, 1.23]
    assert (model.lag, model.traction, model.pivot, model.timeout) == (0.0, 7.85, 0.131, 0.25)


def test_empty_spec_is_unbounded_with_unit_gain():
    pytest.importorskip('flatsim_msgs')
    from arena_robots.caps import PlanarSpec
    from arena_runtime.sim.flatsim_simulator.planar import drive_model

    from flatsim.drive import DEFAULT_TIMEOUT

    model = drive_model(PlanarSpec.from_dict({}), holonomic=False)
    assert _axis(model.linear) == _axis(model.lateral) == _axis(model.angular) == (-math.inf, math.inf, math.inf)
    assert list(model.gain) == [1.0, 1.0, 1.0]
    assert (model.lag, model.traction, model.pivot, model.timeout) == (0.0, math.inf, 0.0, DEFAULT_TIMEOUT)


@pytest.mark.parametrize(('holonomic', 'lateral'), [(False, (0.0, 0.0, math.inf)), (True, (-math.inf, math.inf, math.inf))])
def test_missing_spec_moves_sideways_only_when_holonomic(holonomic, lateral):
    pytest.importorskip('flatsim_msgs')
    from arena_runtime.sim.flatsim_simulator.planar import drive_model

    from flatsim.drive import DEFAULT_TIMEOUT

    model = drive_model(None, holonomic=holonomic)
    assert _axis(model.lateral) == lateral
    assert list(model.gain) == [1.0, 1.0, 1.0]
    assert (model.traction, model.timeout) == (math.inf, DEFAULT_TIMEOUT)


@pytest.mark.parametrize('spec', [None, {}])
def test_turning_radius_reaches_the_drive_model_with_and_without_a_spec(spec):
    pytest.importorskip('flatsim_msgs')
    from arena_robots.caps import PlanarSpec
    from arena_runtime.sim.flatsim_simulator.planar import drive_model

    planar = None if spec is None else PlanarSpec.from_dict(spec)
    assert drive_model(planar, holonomic=False, min_turning_radius=0.74).min_turning_radius == 0.74
    assert drive_model(planar, holonomic=False).min_turning_radius == 0.0


def test_spec_rejects_an_unknown_axis():
    from arena_robots.caps import PlanarSpec

    with pytest.raises(ValueError, match='planar.gain'):
        PlanarSpec.from_dict({'gain': {'yaw': 1.0}})
