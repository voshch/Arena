from __future__ import annotations


def test_sim_simulator_values_stable():
    from arena_bringup.constants import SimSimulator

    assert SimSimulator.DUMMY.value == "dummy"
    assert SimSimulator.FLATSIM.value == "flatsim"
    assert SimSimulator.FLATLAND.value == "flatland"
    assert SimSimulator.GAZEBO.value == "gazebo"
    assert SimSimulator.UNITY.value == "unity"
    assert SimSimulator.ISAAC.value == "isaac"


def test_human_simulator_values_stable():
    from arena_runtime.env.constants import Constants

    HS = Constants.HumanSimulator
    assert HS.DUMMY.value == "dummy"
    assert HS.NONE.value == "none"
    assert HS.HUNAV.value == "hunav"
    assert HS.ARENA.value == "arena"
    assert HS.REPLAY.value == "replay"
    # hunav and arena are both registered — alternative backends chosen by `human:=`,
    # never concurrent. Dropping either changes what `human:=<key>` resolves to.
    assert {m.value for m in HS} == {"dummy", "none", "hunav", "arena", "replay"}
