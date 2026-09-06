from __future__ import annotations

from pathlib import Path

import pytest
from task_generator.auditory.acoustic_room_spec import (
    AcousticBoundarySpec,
    AcousticRoomSpec,
)
from task_generator.auditory.material_catalog import AcousticMaterialCatalog
from task_generator.auditory.pyroomacoustics_adapter import (
    PyroomacousticsAdapter,
    PyroomacousticsConfig,
)


def _room() -> AcousticRoomSpec:
    corners = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
    boundary = tuple(
        AcousticBoundarySpec(
            start=start,
            end=end,
            material_id="Acoustic_Default_Wall",
            kind="wall",
        )
        for start, end in zip(corners, (*corners[1:], corners[0]), strict=True)
    )
    return AcousticRoomSpec(
        zone_name="room",
        boundary=boundary,
        floor_material_id="Acoustic_Default_Floor",
        ceiling_material_id="Acoustic_Default_Ceiling",
        ceiling_height_m=3.0,
    )


def _adapter() -> PyroomacousticsAdapter:
    catalog_path = Path(__file__).resolve().parents[2] / "config" / "auditory" / "acoustic_materials.yaml"
    return PyroomacousticsAdapter(
        AcousticMaterialCatalog(catalog_path),
        PyroomacousticsConfig(),
    )


def test_near_boundary_microphone_is_moved_inside_room():
    adjusted = _adapter()._position_inside_room(
        _room(),
        (-0.23, 0.5, 0.22),
        name="listener_position_m",
    )

    assert adjusted[0] == pytest.approx(0.01)
    assert adjusted[1:] == pytest.approx((0.5, 0.22))


def test_position_well_outside_room_is_rejected():
    with pytest.raises(ValueError, match="outside acoustic room"):
        _adapter()._position_inside_room(
            _room(),
            (-0.36, 0.5, 0.22),
            name="listener_position_m",
        )
