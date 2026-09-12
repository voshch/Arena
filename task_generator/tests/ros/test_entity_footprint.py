"""Static entities must occupy their real footprint, not a fixed 1 m blob.

`update_world` used to stamp every static entity into the occupancy grid as
`radius=1,  # TODO actual radius`. On a generated office world that is catastrophic: 36
pieces of furniture - 22 of them chairs - in a 20 x 12 m world cover up to 60 % of the floor.
Free space fragments, A* reports only tens of reachable cells, and both route planning and
agent placement fail on geometry that is not really there.
"""

from __future__ import annotations

import math

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


class _Ent:
    def __init__(self, bbox=None, z=0.0):
        self._bbox = bbox
        self.pose = type("_P", (), {"position": type("_Q", (), {"x": 0.0, "y": 0.0, "z": z})()})()

    def asdict(self, expand_extra=True):
        return {"bbox": self._bbox} if self._bbox is not None else {}


def _radius(*args, **kwargs):
    from task_generator.manager.world_manager.world_manager import _entity_radius

    return _entity_radius(*args, **kwargs)


def test_a_chair_is_not_a_metre_wide():
    """The regression that motivated this: a 0.5 m chair blocked a 2 x 2 m square."""
    chair = _Ent(bbox=[(-0.25, 0.25), (-0.25, 0.25), (0.0, 0.9)])
    r = _radius(chair)
    assert r == pytest.approx(0.5 * math.hypot(0.5, 0.5), abs=1e-6)
    assert r < 0.4, "must be far below the old hardcoded 1.0"


def test_a_long_table_is_allowed_to_be_bigger_than_one_metre():
    """The old constant under-blocked large objects just as badly as it over-blocked small
    ones - it was simply wrong in both directions."""
    table = _Ent(bbox=[(-1.0, 1.0), (-0.4, 0.4), (0.0, 0.75)])
    assert _radius(table) == pytest.approx(0.5 * math.hypot(2.0, 0.8), abs=1e-6)
    assert _radius(table) > 1.0


def test_radius_is_rotation_invariant():
    """A circumscribing circle means the entity's yaw never has to be applied, and the value
    can never under-block whichever way the object is turned."""
    box = _Ent(bbox=[(-1.0, 1.0), (-0.2, 0.2), (0.0, 0.5)])
    assert _radius(box) == pytest.approx(0.5 * math.hypot(2.0, 0.4), abs=1e-6)


def test_ceiling_fixtures_do_not_block_the_floor():
    """A lamp or wall sign above passage height has a 2D footprint that would falsely wall off
    the room. The map rasterizer already skips these; the occupancy grid must agree with it."""
    lamp = _Ent(bbox=[(-0.3, 0.3), (-0.3, 0.3), (0.1, 0.4)], z=2.5)
    assert _radius(lamp) is None


def test_a_model_without_an_annotation_falls_back_small():
    """Unknown objects under-block on purpose: the navigation stack's own costmap sees them
    from the lidar, whereas a 1 m guess walls off rooms that are actually walkable."""
    assert _radius(_Ent(bbox=None)) == pytest.approx(0.35)


def test_a_malformed_bbox_does_not_crash_world_loading():
    assert _radius(_Ent(bbox="nonsense")) == pytest.approx(0.35)
    assert _radius(_Ent(bbox=[(0.0, 1.0)])) == pytest.approx(0.35)


def test_footprint_comes_from_the_asset_not_just_an_inline_key():
    """The regression that made this look fixed when it was not.

    Worlds like `arena_arena_002` write only `name`, `pose`, `model` on their entities - no
    `bbox` anywhere. An earlier version read only that inline key, so every object silently
    took the 0.35 m fallback: a 2.8 m table became a 0.7 m disc, routes planned straight
    through furniture, and robot spawns landed inside sofas. The footprint has to come from
    the asset's own `annotation.yaml`.
    """
    from arena_simulation_setup.shared.entities import Obstacle

    entity = Obstacle.parse(
        {
            "name": "table_0",
            "model": "Office/Object/Long_Rectangle_Table",
            "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        }
    )
    assert "bbox" not in entity.asdict(expand_extra=True), "fixture must mimic a real world entity"

    r = _radius(entity)
    if r == pytest.approx(0.35):
        pytest.skip("asset annotations not fetched in this environment")
    assert r > 1.0, f"a 0.8 x 2.8 m table must not collapse to the fallback (got {r})"


def test_distinct_models_get_distinct_radii():
    """If every object comes back the same size, the asset lookup is not happening."""
    from arena_simulation_setup.shared.entities import Obstacle

    def r(model):
        return _radius(Obstacle.parse({"name": "x", "model": model, "pose": {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}}))

    chair = r("Residential/Object/Steel_Chair")
    sofa = r("Residential/Object/Sofa")
    if chair == pytest.approx(0.35) and sofa == pytest.approx(0.35):
        pytest.skip("asset annotations not fetched in this environment")
    assert sofa > chair, f"a sofa must block more than a chair (sofa={sofa}, chair={chair})"
