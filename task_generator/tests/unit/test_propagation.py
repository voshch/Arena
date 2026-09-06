from __future__ import annotations

import pytest

pytest.importorskip("geometry_msgs.msg")

from geometry_msgs.msg import Point
from task_generator.auditory.acoustic_scene import AcousticScene
from task_generator.auditory.propagation import (
    SPEED_OF_SOUND_MPS,
    Level3Propagation,
)


class _UnusedMaterialCatalog:
    pass


def test_direct_delay_uses_physical_distance_below_attenuation_floor() -> None:
    propagation = Level3Propagation(_UnusedMaterialCatalog())
    scene = AcousticScene(zones=(), walls=())

    result = propagation.calculate(
        scene,
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=0.2, y=0.0, z=0.0),
        60.0,
    )

    assert result.direct_delay_sec == pytest.approx(0.2 / SPEED_OF_SOUND_MPS)
    assert result.paths[0].delay_sec == pytest.approx(0.2 / SPEED_OF_SOUND_MPS)
    assert result.paths[0].gain_db == pytest.approx(0.0)
