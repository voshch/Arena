from __future__ import annotations

import pytest

from arena_simulation_setup.shared.entities import (
    CustomDynamicObstacle,
    DynamicObstacle,
    Entity,
    Named,
    Obstacle,
)
from arena_simulation_setup.tree.assets.Object import ObjectIdentifier
from arena_simulation_setup.utils.geometry import Pose, Scale


def _model_id():
    return ObjectIdentifier("test_model")


def _make_entity(**kwargs) -> Entity:
    defaults = dict(
        name="test_entity",
        pose=Pose(),
        model=_model_id(),
    )
    defaults.update(kwargs)
    return Entity(**defaults)


def _make_obstacle(**kwargs) -> Obstacle:
    defaults = dict(
        name="test_obstacle",
        pose=Pose(),
        model=_model_id(),
        scale=None,
    )
    defaults.update(kwargs)
    return Obstacle(**defaults)


# ---------------------------------------------------------------------------
# Named
# ---------------------------------------------------------------------------


def test_named_sim_path_default_is_name():
    n = Named(name="foo")
    assert n.sim_path == "foo"


def test_named_sim_path_override():
    n = Named(name="foo", extra={"sim_path": "bar"})
    assert n.sim_path == "bar"


def test_named_sim_path_setter():
    n = Named(name="foo")
    n.sim_path = "custom"
    assert n.sim_path == "custom"
    assert n.extra["sim_path"] == "custom"


def test_named_extra_dict_preserved():
    n = Named(name="test", extra={"key": "value"})
    assert n.extra["key"] == "value"


def test_named_serialize_omits_empty_extra():
    n = Named(name="hello")
    d = n.serialize()
    assert "extra" not in d or not d["extra"]


def test_named_serialize_preserves_nonempty_extra():
    n = Named(name="hello", extra={"custom_key": 42})
    d = n.serialize()
    assert d.get("custom_key") == 42 or d.get("extra", {}).get("custom_key") == 42


def test_named_parse_with_pos_alias():
    d = {"name": "x", "pos": [1.0, 2.0]}
    n = Named.parse(d)
    assert n.name == "x"


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


def test_entity_pose_converter():
    e = _make_entity(pose=Pose.parse([1.0, 2.0, 0.5]))
    assert e.pose.position.x == pytest.approx(1.0)


def test_entity_model_is_object_identifier():
    e = _make_entity()
    assert isinstance(e.model, ObjectIdentifier)


def test_entity_asdict_expand_extra_true():
    e = _make_entity()
    e.extra["my_key"] = "my_val"
    d = e.asdict(expand_extra=True)
    assert "my_key" in d


def test_entity_asdict_expand_extra_false():
    e = _make_entity()
    e.extra["my_key"] = "my_val"
    d = e.asdict(expand_extra=False)
    assert "my_key" not in d


# ---------------------------------------------------------------------------
# Obstacle
# ---------------------------------------------------------------------------


def test_obstacle_scale_set():
    obs = _make_obstacle()
    obs.scale = Scale(2.0, 2.0, 2.0)
    assert obs.scale.x == pytest.approx(2.0)


def test_obstacle_asdict_returns_dict():
    obs = _make_obstacle()
    d = obs.asdict()
    assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# DynamicObstacle
# ---------------------------------------------------------------------------


def test_dynamic_obstacle_empty_waypoints():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    do = DynamicObstacle(
        name="dyn",
        pose=Pose(),
        model=HumanIdentifier.converter("some_pedestrian"),
        waypoints=[],
        velocity=1.5,
    )
    assert do.waypoints == []
    assert do.velocity == pytest.approx(1.5)


def test_dynamic_obstacle_velocity_converter_string():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    do = DynamicObstacle(
        name="dyn",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
        velocity="2.5",
    )
    assert do.velocity == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# CustomDynamicObstacle
# ---------------------------------------------------------------------------


def test_custom_dynamic_obstacle_getattr_hit():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    cdo = CustomDynamicObstacle(
        name="cdo",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
    )
    cdo.extra["custom_prop"] = 99
    assert cdo.custom_prop == 99


def test_custom_dynamic_obstacle_getattr_miss():
    from arena_simulation_setup.tree.assets.Human import HumanIdentifier

    cdo = CustomDynamicObstacle(
        name="cdo",
        pose=Pose(),
        model=HumanIdentifier.converter("p"),
    )
    with pytest.raises(AttributeError):
        _ = cdo.nonexistent_field


def test_custom_dynamic_obstacle_parse_emits_future_warning():
    data = {
        "name": "cdo",
        "pose": [0.0, 0.0],
        "model": "p",
        "custom_field": "hello",
    }
    with pytest.warns(FutureWarning):
        cdo = CustomDynamicObstacle.parse(data)
    assert cdo.name == "cdo"
