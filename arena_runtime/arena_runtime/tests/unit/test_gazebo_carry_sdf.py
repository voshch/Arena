"""A carried Gazebo model respawns from its SDF without collisions, its meshes still found from outside its directory."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _generate_box_sdf, without_collisions  # noqa: E402

MODEL = """<sdf version="1.9"><model name="crate"><static>true</static><link name="link">
<visual name="visual"><geometry><mesh><uri>meshes/crate.obj</uri></mesh></geometry></visual>
<collision name="collision"><geometry><mesh><uri>meshes/crate.obj</uri></mesh></geometry></collision>
<visual name="logo"><geometry><mesh><uri>model://logos/meshes/logo.dae</uri></mesh></geometry></visual>
</link></model></sdf>"""


def test_a_model_loses_its_collisions_and_keeps_its_visuals(tmp_path: Path) -> None:
    sdf = without_collisions(MODEL, tmp_path)
    assert "<collision" not in sdf
    assert sdf.count("<visual") == 2
    assert f"<uri>{(tmp_path / 'meshes' / 'crate.obj').as_uri()}</uri>" in sdf
    assert "<uri>model://logos/meshes/logo.dae</uri>" in sdf


def test_a_box_obstacle_loses_its_collision() -> None:
    sdf = without_collisions(_generate_box_sdf("crate", (0.4, 0.3, 0.4), (0.0, 0.0, 0.2)), None)
    assert "<collision" not in sdf
    assert "<size>0.4 0.3 0.4</size>" in sdf
