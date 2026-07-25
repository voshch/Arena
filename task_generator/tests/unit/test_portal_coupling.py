from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from shapely.geometry import Polygon

from task_generator.auditory.acoustic_room_spec import (
    AcousticBoundarySpec,
    AcousticRoomSpec,
)
from task_generator.auditory.acoustic_world_graph import (
    AcousticPortal,
    AcousticWorldGraph,
)
from task_generator.auditory.portal_coupling import (
    MultiPortalRirCoupler,
    OneDoorRirCoupler,
    PortalCouplingConfig,
)
from task_generator.auditory.pyroomacoustics_adapter import RoomImpulseResponse


def _room(name: str, x_min: float, x_max: float) -> AcousticRoomSpec:
    corners = ((x_min, 0.0), (x_max, 0.0), (x_max, 1.0), (x_min, 1.0))
    boundary = tuple(
        AcousticBoundarySpec(
            start=start,
            end=corners[(index + 1) % len(corners)],
            material_id="wall",
            kind="wall",
        )
        for index, start in enumerate(corners)
    )
    return AcousticRoomSpec(
        zone_name=name,
        boundary=boundary,
        floor_material_id="floor",
        ceiling_material_id="ceiling",
        ceiling_height_m=3.0,
    )


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def compute_rir(self, room, *, source_position_m, listener_position_m):
        self.calls += 1
        return RoomImpulseResponse(
            samples=np.asarray([1.0, 0.5], dtype=np.float64),
            sample_rate_hz=100,
            global_delay_samples=2,
            fallback_material_ids=(),
        )


def test_one_door_rir_is_composed_and_endpoint_rirs_are_cached() -> None:
    first = _room("a", 0.0, 1.0)
    second = _room("b", 1.0, 2.0)
    portal = AcousticPortal(
        portal_id="door:test:a:b",
        door_name="test",
        zone_a="a",
        zone_b="b",
        start=(1.0, 0.4),
        end=(1.0, 0.6),
        height_m=2.0,
        material_id="door",
    )
    graph = AcousticWorldGraph(
        rooms=(first, second),
        portals=(portal,),
        zone_polygons=(
            ("a", Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))),
            ("b", Polygon(((1, 0), (2, 0), (2, 1), (1, 1)))),
        ),
    )
    adapter = _FakeAdapter()
    coupler = OneDoorRirCoupler(
        adapter,  # type: ignore[arg-type]
        graph,
        world_name="test",
        config=PortalCouplingConfig(
            portal_loss_db=6.0,
            source_room_early_window_sec=0.02,
            maximum_rir_duration_sec=1.0,
        ),
    )
    arguments = dict(
        source_zone="a",
        listener_zone="b",
        source_position_m=(0.5, 0.5, 1.6),
        listener_position_m=(1.5, 0.5, 0.35),
    )

    first_result = coupler.compute(**arguments)
    second_result = coupler.compute(**arguments)

    expected = fftconvolve([1.0, 0.5], [1.0, 0.5]) * 10.0 ** (-6.0 / 20.0)
    np.testing.assert_allclose(first_result.rir.samples, expected)
    np.testing.assert_allclose(second_result.rir.samples, expected)
    assert first_result.rir.global_delay_samples == 4
    assert first_result.portal.portal_id == "door:test:a:b"
    assert adapter.calls == 2
    assert coupler.cache_entries == 2
    assert coupler.cache_hits == 0
    assert coupler.cache_misses == 2
    assert coupler.route_cache_entries == 1
    assert coupler.route_cache_hits == 1
    assert coupler.route_cache_misses == 1


def test_multi_portal_rir_composes_transit_rooms_and_reuses_cache() -> None:
    first = _room("a", 0.0, 1.0)
    second = _room("b", 1.0, 2.0)
    third = _room("c", 2.0, 3.0)
    portals = (
        AcousticPortal(
            portal_id="door:ab",
            door_name="ab",
            zone_a="a",
            zone_b="b",
            start=(1.0, 0.4),
            end=(1.0, 0.6),
            height_m=2.0,
            material_id="door",
        ),
        AcousticPortal(
            portal_id="opening:bc",
            door_name="",
            zone_a="b",
            zone_b="c",
            start=(2.0, 0.4),
            end=(2.0, 0.6),
            height_m=2.5,
            material_id="open",
            portal_kind="opening",
        ),
    )
    graph = AcousticWorldGraph(
        rooms=(first, second, third),
        portals=portals,
        zone_polygons=(
            ("a", Polygon(((0, 0), (1, 0), (1, 1), (0, 1)))),
            ("b", Polygon(((1, 0), (2, 0), (2, 1), (1, 1)))),
            ("c", Polygon(((2, 0), (3, 0), (3, 1), (2, 1)))),
        ),
    )
    adapter = _FakeAdapter()
    coupler = MultiPortalRirCoupler(
        adapter,  # type: ignore[arg-type]
        graph,
        world_name="test",
        config=PortalCouplingConfig(
            portal_loss_db=6.0,
            opening_portal_loss_db=1.0,
            source_room_early_window_sec=0.02,
            maximum_rir_duration_sec=1.0,
            max_portal_hops=3,
        ),
    )
    arguments = dict(
        source_zone="a",
        listener_zone="c",
        source_position_m=(0.5, 0.5, 1.6),
        listener_position_m=(2.5, 0.5, 0.35),
    )

    first_result = coupler.compute(**arguments)
    second_result = coupler.compute(**arguments)

    expected = fftconvolve(
        fftconvolve([1.0, 0.5], [1.0, 0.5]),
        [1.0, 0.5],
    ) * 10.0 ** (-7.0 / 20.0)
    np.testing.assert_allclose(first_result.rir.samples, expected)
    np.testing.assert_allclose(second_result.rir.samples, expected)
    assert first_result.route is not None
    assert first_result.route.zones == ("a", "b", "c")
    assert first_result.route.hop_count == 2
    assert first_result.applied_portal_loss_db == 7.0
    assert adapter.calls == 3
    assert coupler.cache_entries == 3
    assert coupler.cache_hits == 0
    assert coupler.cache_misses == 3
    assert coupler.route_cache_entries == 1
    assert coupler.route_cache_hits == 1
    assert coupler.route_cache_misses == 1
