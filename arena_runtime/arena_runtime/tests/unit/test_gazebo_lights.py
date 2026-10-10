"""Gazebo light requests and SDF follow the light state, a rig becomes one point light per cell of fixtures."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("arena_runtime.sim.gazebo_simulator.gazebo_simulator")
light_pb2 = pytest.importorskip("gz.msgs10.light_pb2")

from arena_runtime.sim.gazebo_simulator.gazebo_simulator import (  # noqa: E402
    _default_light_request,
    _generate_light_sdf,
    _glow_request,
    _glow_visuals,
    _light_params,
    _light_request,
    _rig_cells,
    _sdf_file,
)
from arena_simulation_setup.shared import Light  # noqa: E402
from arena_simulation_setup.utils.geometry import Position  # noqa: E402


def _requests(light: Light, level: float, alive: list[bool], position: Position | None = None) -> list[light_pb2.Light]:
    return [_light_request(f"l{index}", params) for index, params in enumerate(_light_params(light, level, alive, position))]


def _rig() -> Light:
    fixtures = [Position(x, y, 2.58) for y in (1.8, 4.2) for x in (1.6, 4.0, 6.4)]
    return Light(name="hall", fixture="panel", lumens=3600.0, rig=True, fixtures=fixtures, fixture_ranks=[0.76, 0.709, 0.174, 0.357, 0.77, 0.696])


def _strip() -> Light:
    return Light(name="exit_strip", fixture="tube", position=Position(7.8, 3.0, 2.2), lumens=300.0, cct_K=6500.0)


def test_rig_cells_group_neighboring_fixtures() -> None:
    assert sorted(sorted(cell) for cell in _rig_cells(_rig())) == [[0, 1, 3, 4], [2, 5]]


def test_single_fixture_rig_is_one_cell() -> None:
    rig = Light(name="closet", fixture="panel", lumens=3600.0, rig=True, fixtures=[Position(1.0, 1.0, 2.4)], fixture_ranks=[0.5])
    assert _rig_cells(rig) == [[0]]


def test_rig_requests_are_point_lights_at_cell_centroids() -> None:
    first, second, _ = _requests(_rig(), 1.0, [True] * 6)
    assert first.type == second.type == light_pb2.Light.POINT
    assert (first.pose.position.x, first.pose.position.y, first.pose.position.z) == pytest.approx((2.8, 3.0, 2.58))
    assert (second.pose.position.x, second.pose.position.y, second.pose.position.z) == pytest.approx((6.4, 3.0, 2.58))
    assert first.intensity == pytest.approx(2.0 * second.intensity)
    assert first.attenuation_quadratic > 0.0
    assert not first.is_light_off


def test_rig_intensity_scales_with_level() -> None:
    full = _requests(_rig(), 1.0, [True] * 6)
    dimmed = _requests(_rig(), 0.3, [True] * 6)
    assert [msg.intensity for msg in dimmed] == pytest.approx([0.3 * msg.intensity for msg in full], rel=1e-5)


def test_dead_fixtures_darken_their_own_cell() -> None:
    full = _requests(_rig(), 1.0, [True] * 6)
    partial = _requests(_rig(), 1.0, [True, True, False, False, True, False])
    assert partial[0].intensity == pytest.approx(0.75 * full[0].intensity, rel=1e-5)
    assert partial[1].intensity == 0.0
    assert partial[1].is_light_off
    assert partial[2].intensity == pytest.approx(0.5 * full[2].intensity, rel=1e-5)


def test_rig_adds_one_unattenuated_bounce_fill_at_the_room_center() -> None:
    *_, bounce = _requests(_rig(), 1.0, [True] * 6)
    assert bounce.type == light_pb2.Light.POINT
    assert (bounce.pose.position.x, bounce.pose.position.y, bounce.pose.position.z) == pytest.approx((4.0, 3.0, 1.29))
    assert bounce.attenuation_quadratic == 0.0
    assert not bounce.cast_shadows
    surface = 2.0 * (7.2 * 4.8 + 7.2 * 2.58 + 4.8 * 2.58)
    assert bounce.intensity == pytest.approx(0.4 * 6 * 3600.0 / (surface * 0.6) / 400.0, rel=1e-4)
    *_, dark = _requests(_rig(), 0.0, [True] * 6)
    assert dark.is_light_off


def test_unlit_request_switches_the_light_off() -> None:
    (msg,) = _requests(_strip(), 0.0, [])
    assert msg.intensity == 0.0
    assert msg.is_light_off


def test_fixture_request_carries_color_temperature_and_range() -> None:
    (msg,) = _requests(_strip(), 1.0, [])
    assert msg.diffuse.b > 0.95
    assert msg.range > 0.0
    assert msg.attenuation_quadratic > 0.0


def test_frame_light_request_keeps_its_tracked_position() -> None:
    headlight = Light(name="headlight", fixture="spot", frame="env_0/turtlebot/base_link", offset=Position(0.15, 0.0, 0.25), lumens=1500.0)
    (placeholder,) = _requests(headlight, 1.0, [])
    (tracked,) = _requests(headlight, 1.0, [], Position(6.6, 8.0, 0.25))
    assert (placeholder.pose.position.x, placeholder.pose.position.y) == pytest.approx((0.15, 0.0))
    assert (tracked.pose.position.x, tracked.pose.position.y, tracked.pose.position.z) == pytest.approx((6.6, 8.0, 0.25))


def test_narrow_cone_concentrates_the_same_lumens() -> None:
    wide = Light(name="wide", fixture="downlight", position=Position(0.0, 0.0, 2.5), lumens=1000.0)
    narrow = Light(name="narrow", fixture="spot", position=Position(0.0, 0.0, 2.5), lumens=1000.0)
    assert _requests(narrow, 1.0, [])[0].intensity > 5.0 * _requests(wide, 1.0, [])[0].intensity


def test_spot_request_carries_its_cone() -> None:
    spot = Light(name="lamp", fixture="spot", position=Position(1.0, 1.0, 2.0), lumens=500.0)
    (msg,) = _requests(spot, 1.0, [])
    assert msg.type == light_pb2.Light.SPOT
    assert msg.spot_outer_angle == pytest.approx(math.radians(40.0), abs=1e-3)
    assert msg.spot_inner_angle == pytest.approx(0.5 * msg.spot_outer_angle)


def test_sun_request_is_directional() -> None:
    sun = Light(name="sun", fixture="sun", lux=400.0, direction=(1.0, 0.0, -1.0))
    (msg,) = _requests(sun, 1.0, [])
    assert msg.type == light_pb2.Light.DIRECTIONAL
    assert (msg.direction.x, msg.direction.z) == pytest.approx((2**-0.5, -(2**-0.5)))
    assert msg.intensity == pytest.approx(1.0)


@pytest.mark.parametrize(("scale", "off"), [(1.0, False), (0.0, True)])
def test_default_light_request_scales_a_directional_light(scale: float, off: bool) -> None:
    msg = _default_light_request("sun", (-1.0, 0.0, -0.35), scale)
    assert msg.type == light_pb2.Light.DIRECTIONAL
    assert msg.intensity == pytest.approx(scale)
    assert msg.is_light_off is off
    assert msg.direction.z == pytest.approx(-0.35)


def test_rig_sdf_is_one_point_light_per_cell_and_a_bounce_fill() -> None:
    documents = [ET.fromstring(_generate_light_sdf(f"env_0/light_{index}", params).strip()) for index, params in enumerate(_light_params(_rig(), 1.0, [True] * 6))]
    lights = [document.find("light") for document in documents]
    assert [light.get("name") for light in lights] == ["env_0/light_0", "env_0/light_1", "env_0/light_2"]
    assert all(light.get("type") == "point" and light.find("spot") is None for light in lights)
    assert all(float(light.findtext("intensity")) > 0.0 for light in lights)


def test_fixture_sdf_is_a_spot_with_its_cone() -> None:
    spot = Light(name="lamp", fixture="spot", position=Position(1.0, 1.0, 2.0), lumens=500.0, direction=(1.0, 0.0, -1.0))
    (params,) = _light_params(spot, 1.0, [])
    light = ET.fromstring(_generate_light_sdf("lamp", params).strip()).find("light")
    assert light.get("type") == "spot"
    assert float(light.findtext("spot/outer_angle")) == pytest.approx(0.6981, abs=1e-3)


def test_bulb_request_is_a_point_light_spreading_its_lumens_over_the_sphere() -> None:
    bulb = Light(name="desk_lamp_bulb", fixture="bulb", position=Position(2.0, 1.0, 1.2), lumens=800.0, intrinsic=True)
    (msg,) = _requests(bulb, 1.0, [])
    (spot,) = _requests(Light(name="lamp", fixture="spot", position=Position(2.0, 1.0, 1.2), lumens=800.0), 1.0, [])
    assert msg.type == light_pb2.Light.POINT
    assert (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) == pytest.approx((2.0, 1.0, 1.2))
    assert msg.attenuation_quadratic > 0.0
    assert 0.0 < msg.intensity < spot.intensity


def test_spot_cone_override_reaches_the_request() -> None:
    wide = Light(name="lamp", fixture="spot", position=Position(0.0, 0.0, 1.0), lumens=300.0, cone_deg=90.0)
    (msg,) = _requests(wide, 1.0, [])
    assert msg.spot_outer_angle == pytest.approx(math.radians(90.0))


_GLOW_SDF = """<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="Desk_Lamp">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry><mesh><uri>Desk_Lamp.dae</uri></mesh></geometry>
      </visual>
      <visual name="glow_Shade">
        <geometry><mesh><uri>Desk_Lamp.glow_0.dae</uri></mesh></geometry>
        <material>
          <ambient>0.85 0.8 0.7 1</ambient>
          <diffuse>0.85 0.8 0.7 1</diffuse>
          <specular>0 0 0 1</specular>
          <emissive>0 0 0 1</emissive>
        </material>
      </visual>
      <visual name="glow_Paper Globe">
        <geometry><mesh><uri>Desk_Lamp.glow_1.dae</uri></mesh></geometry>
      </visual>
    </link>
  </model>
</sdf>"""


def test_glow_visuals_map_each_glow_material_to_its_visual_and_base_color(tmp_path: Path) -> None:
    sdf = tmp_path / "Desk_Lamp.sdf"
    sdf.write_text(_GLOW_SDF)
    assert _glow_visuals(sdf) == {"Shade": ("link::glow_Shade", (0.85, 0.8, 0.7)), "Paper Globe": ("link::glow_Paper Globe", (1.0, 1.0, 1.0))}


def test_model_without_glow_visuals_has_none(tmp_path: Path) -> None:
    sdf = tmp_path / "Desk.sdf"
    sdf.write_text('<sdf version="1.7"><model name="Desk"><link name="link"><visual name="visual"/></link></model></sdf>')
    assert _glow_visuals(sdf) == {}


def test_sdf_file_resolves_a_model_directory_to_its_sdf(tmp_path: Path) -> None:
    named = tmp_path / "Desk_Lamp.sdf"
    named.mkdir()
    (named / "Desk_Lamp.sdf.sdf").write_text("")
    other = tmp_path / "Chair"
    other.mkdir()
    (other / "model.sdf").write_text("")
    assert _sdf_file(named) == named / "Desk_Lamp.sdf.sdf"
    assert _sdf_file(other) == other / "model.sdf"
    assert _sdf_file(other / "model.sdf") == other / "model.sdf"


def test_glow_request_emits_the_light_color_dimmed_with_level_and_keeps_the_base_color() -> None:
    full = _glow_request("env_0/lamp::link::glow_Shade", (0.85, 0.8, 0.7), 2700.0, 1.0)
    dim = _glow_request("env_0/lamp::link::glow_Shade", (0.85, 0.8, 0.7), 2700.0, 0.25)
    off = _glow_request("env_0/lamp::link::glow_Shade", (0.85, 0.8, 0.7), 2700.0, 0.0)
    assert full.entity.name == "env_0/lamp::link::glow_Shade"
    assert (full.diffuse.r, full.diffuse.g, full.diffuse.b, full.diffuse.a) == pytest.approx((0.85, 0.8, 0.7, 1.0))
    assert (full.ambient.r, full.ambient.g, full.ambient.b) == pytest.approx((0.85, 0.8, 0.7))
    assert full.emissive.r > full.emissive.g > full.emissive.b > 0.0
    assert dim.emissive.r / full.emissive.r == pytest.approx(dim.emissive.b / full.emissive.b)
    assert 0.25 < dim.emissive.r / full.emissive.r < 1.0
    assert (off.emissive.r, off.emissive.g, off.emissive.b) == (0.0, 0.0, 0.0)
    assert (off.diffuse.r, off.diffuse.g, off.diffuse.b) == pytest.approx((0.85, 0.8, 0.7))
