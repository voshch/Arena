"""Box primitives carry the look of the material they were spawned with."""

from __future__ import annotations

import pytest

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import _generate_box_sdf  # noqa: E402

SIZE = (1.0, 0.1, 2.0)


def test_box_sdf_binds_the_resolved_textures() -> None:
    sdf = _generate_box_sdf("door", SIZE, textures={"albedo": "/m/wood.png", "normal": "/m/wood_n.png"})
    assert "<albedo_map>/m/wood.png</albedo_map>" in sdf
    assert "<normal_map>/m/wood_n.png</normal_map>" in sdf
    assert "<size>1.0 0.1 2.0</size>" in sdf


@pytest.mark.parametrize("textures", [None, {}])
def test_box_sdf_is_flat_gray_without_textures(textures: dict[str, str] | None) -> None:
    sdf = _generate_box_sdf("door", SIZE, textures=textures)
    assert "<albedo_map>" not in sdf
    assert "<diffuse>0.7 0.7 0.7 1</diffuse>" in sdf
