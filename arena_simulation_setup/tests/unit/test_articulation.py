from __future__ import annotations

import math

import numpy as np
import pytest

from arena_simulation_setup.utils.articulation import Articulation, Joint, JointType, family_members, matrix

CABINET = {
    "links": {"base": [], "door": ["door_mesh"], "drawer": ["drawer_mesh", "handle"]},
    "joints": [
        {"name": "door", "type": "revolute", "parent": "base", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 2], "limit": [0.0, 1.57]},
        {"name": "drawer", "type": "prismatic", "parent": "base", "child": "drawer", "origin": [0.0, 0.0, 0.6], "axis": [0, -1, 0], "limit": [0.0, 0.4]},
    ],
    "states": {"closed": {"door": 0.0, "drawer": 0.0}, "open": {"door": 1.5, "drawer": 0.35}},
}


def cabinet() -> Articulation:
    return Articulation.parse(CABINET)


def point(pose, xyz):
    return list(pose * type(pose.position)(*xyz))


def test_parse_normalizes_the_axis_and_finds_the_root():
    articulation = cabinet()
    assert list(articulation.joint_map["door"].axis) == [0.0, 0.0, 1.0]
    assert articulation.root == "base"
    assert [joint.name for joint in articulation.movable] == ["door", "drawer"]


def test_unstructure_round_trips():
    articulation = cabinet()
    assert Articulation.parse(articulation.unstructure()).unstructure() == articulation.unstructure()


def test_revolute_turns_about_its_origin():
    poses = cabinet().link_poses({"door": math.pi / 2})
    assert point(poses["door"], (0.5, -0.3, 1.0)) == pytest.approx([0.5, -0.3, 1.0])
    assert point(poses["door"], (0.0, -0.3, 0.0)) == pytest.approx([0.5, -0.8, 0.0])
    assert point(poses["base"], (1.0, 2.0, 3.0)) == pytest.approx([1.0, 2.0, 3.0])


def test_prismatic_slides_along_its_axis():
    poses = cabinet().link_poses({"drawer": 0.3})
    assert point(poses["drawer"], (0.1, 0.2, 0.6)) == pytest.approx([0.1, -0.1, 0.6])


def test_chained_joints_compose_parent_first():
    articulation = Articulation.parse(
        {
            "links": {"base": [], "carrier": [], "lid": ["lid"]},
            "joints": [
                {"name": "slide", "type": "prismatic", "parent": "base", "child": "carrier", "axis": [1, 0, 0], "limit": [0, 1]},
                {"name": "hinge", "type": "continuous", "parent": "carrier", "child": "lid", "origin": [0, 0, 1], "axis": [0, 1, 0]},
            ],
        }
    )
    pose = articulation.link_poses({"slide": 0.5, "hinge": math.pi})["lid"]
    assert point(pose, (0.0, 0.0, 1.0)) == pytest.approx([0.5, 0.0, 1.0])
    assert point(pose, (1.0, 0.0, 1.0)) == pytest.approx([-0.5, 0.0, 1.0])


def test_initial_is_the_closed_state_else_zero_clamped():
    assert cabinet().initial() == {"door": 0.0, "drawer": 0.0}
    articulation = Articulation.parse({"links": {"base": [], "lid": []}, "joints": [{"name": "lid", "type": "revolute", "parent": "base", "child": "lid", "limit": [0.2, 1.0]}]})
    assert articulation.initial() == {"lid": 0.2}


def test_resolve_reads_state_names_and_values_and_clamps():
    articulation = cabinet()
    assert articulation.resolve("open") == {"door": 1.5, "drawer": 0.35}
    assert articulation.resolve({"drawer": 9.0}) == {"door": 0.0, "drawer": 0.4}
    assert articulation.resolve({"door": "open"}) == {"door": 1.5, "drawer": 0.0}
    with pytest.raises(ValueError, match="unknown state 'ajar', states: closed, open"):
        articulation.resolve("ajar")
    with pytest.raises(ValueError, match="unknown movable joint 'lid'"):
        articulation.resolve({"lid": 1.0})


def test_state_of_matches_within_tolerance_and_is_empty_between():
    articulation = cabinet()
    assert articulation.state_of({"door": 1.5, "drawer": 0.3500001}) == "open"
    assert articulation.state_of({"door": 0.7, "drawer": 0.0}) == ""


def test_continuous_joint_wraps_and_fixed_stays():
    continuous = Joint(name="wheel", type=JointType.CONTINUOUS, parent="a", child="b")
    assert continuous.clamp(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert continuous.bounds == (-math.pi, math.pi)
    fixed = Joint(name="glass", type=JointType.FIXED, parent="a", child="c")
    assert fixed.clamp(1.0) == 0.0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value["joints"][0].pop("limit"), "revolute joint door needs a limit"),
        (lambda value: value["joints"][0].update(child="base"), "connects link base to itself"),
        (lambda value: value["joints"][1].update(child="door"), "a link has one parent joint, door has more"),
        (lambda value: value["joints"][0].update(parent="frame"), "names link 'frame', which links does not list"),
        (lambda value: value["links"].update(spare=[]), "exactly one root link"),
        (lambda value: value["links"].update(drawer=["door_mesh"]), "door_mesh is listed twice"),
        (lambda value: value["states"].update(moving={"door": 0.0}), "state name 'moving' is taken"),
        (lambda value: value["states"].update(wide={"door": 3.0}), "outside its limit"),
        (lambda value: value["joints"][0].update(axis=[0, 0, 0]), "joint axis must not be zero"),
        (lambda value: value["joints"][0].update(hinge="left"), "unknown keys \\['hinge'\\]"),
        (lambda value: value.update(extra=1), "unknown keys \\['extra'\\]"),
        (lambda value: value["joints"][0].update(dynamics={"friction": -1.0}), "friction takes a number >= 0"),
        (lambda value: value["joints"][0].update(dynamics={"friction": 1.0, "friction_coefficient": 0.2}), "friction or friction_coefficient, not both"),
        (lambda value: value["joints"][0].update(dynamics={"spring": {"stiffness": 1.0, "reference": 0.0}}), "spring has unknown keys \\['reference'\\]"),
        (lambda value: value["joints"][0].update(role="trapdoor"), "joint role must be one of"),
        (lambda value: value.update(materials={"lid": {"static_friction": 0.5}}), "materials name links lid"),
        (lambda value: value.update(materials={"door": {"restitution": 1.5}}), "restitution takes a number in \\[0, 1\\]"),
    ],
)
def test_parse_names_the_problem(change, message):
    value = {"links": {k: list(v) for k, v in CABINET["links"].items()}, "joints": [dict(joint) for joint in CABINET["joints"]], "states": {k: dict(v) for k, v in CABINET["states"].items()}}
    change(value)
    with pytest.raises(ValueError, match=message):
        Articulation.parse(value)


def test_a_cycle_is_refused():
    with pytest.raises(ValueError, match="exactly one root link|cycle"):
        Articulation.parse(
            {
                "links": {"a": [], "b": []},
                "joints": [
                    {"name": "ab", "type": "continuous", "parent": "a", "child": "b"},
                    {"name": "ba", "type": "continuous", "parent": "b", "child": "a"},
                ],
            }
        )


def test_only_fixed_joints_are_refused():
    with pytest.raises(ValueError, match="at least one joint that is not fixed"):
        Articulation.parse({"links": {"a": [], "b": []}, "joints": [{"name": "ab", "type": "fixed", "parent": "a", "child": "b"}]})


def test_transformed_follows_the_floor_transform_and_scales_prismatic_values():
    transform = np.eye(4)
    transform[:3, :3] = 0.01 * np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform[:3, 3] = [0.0, 0.0, 0.2]
    moved = cabinet().transformed(transform)
    door = moved.joint_map["door"]
    drawer = moved.joint_map["drawer"]
    assert list(door.origin) == pytest.approx([0.003, 0.005, 0.2])
    assert door.limit == (0.0, 1.57)
    assert list(drawer.axis) == pytest.approx([1.0, 0.0, 0.0])
    assert drawer.limit == pytest.approx((0.0, 0.004))
    assert moved.states["open"] == pytest.approx({"door": 1.5, "drawer": 0.0035})
    p = np.array([0.4, -0.2, 0.1, 1.0])
    original = matrix(cabinet().link_poses({"door": 1.0})["door"]) @ p
    carried = matrix(moved.link_poses({"door": 1.0})["door"]) @ (transform @ p)
    assert carried[:3] == pytest.approx((transform @ original)[:3])


def test_transformed_refuses_shear():
    shear = np.eye(4)
    shear[0, 1] = 0.5
    with pytest.raises(ValueError, match="uniform scale only"):
        cabinet().transformed(shear)


def test_sweep_covers_the_door_swing():
    swept = cabinet().sweep({"base": ((-0.5, 0.5), (-0.3, 0.3), (0.0, 1.0)), "door": ((-0.5, 0.5), (-0.32, -0.3), (0.0, 1.0))})
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = swept
    assert y_min == pytest.approx(-1.30, abs=0.005)
    assert x_max == pytest.approx(0.52, abs=0.005)
    assert (x_min, y_max, z_min, z_max) == pytest.approx((-0.5, 0.3, 0.0, 1.0))


def test_configurations_fall_back_to_draws_for_many_joints():
    articulation = Articulation.parse(
        {
            "links": {"base": [], **{f"wheel_{i}": [] for i in range(10)}},
            "joints": [{"name": f"wheel_{i}", "type": "continuous", "parent": "base", "child": f"wheel_{i}"} for i in range(10)],
        }
    )
    configurations = list(articulation.configurations(samples=64))
    assert len(configurations) == 10 * 9 + 64
    assert configurations[0] == {"wheel_0": -math.pi}


KITCHEN = {
    "links": {"base": [], "door": [], "drawer": [], "lid": [], "flap": [], "knob": [], "wheel": [], "dial": []},
    "joints": [
        {"name": "door", "type": "revolute", "parent": "base", "child": "door", "limit": [0.0, 1.57]},
        {"name": "drawer", "type": "prismatic", "parent": "base", "child": "drawer", "axis": [0, -1, 0], "limit": [0.0, 0.4]},
        {"name": "lid", "type": "revolute", "parent": "base", "child": "lid", "axis": [1, 0, 0], "limit": [-1.5, 1.5]},
        {"name": "flap", "type": "revolute", "parent": "base", "child": "flap", "limit": [-1.0, 0.0]},
        {"name": "knob", "type": "revolute", "parent": "base", "child": "knob", "limit": [0.5, 2.5]},
        {"name": "wheel", "type": "continuous", "parent": "base", "child": "wheel"},
        {"name": "dial", "type": "prismatic", "parent": "base", "child": "dial", "limit": [0.0, 0.02]},
    ],
    "states": {"closed": {"door": 0.0, "drawer": 0.0, "lid": 0.0, "flap": 0.0, "knob": 1.0}, "open": {"door": 1.5, "drawer": 0.35, "lid": 1.2, "flap": -0.8, "knob": 1.0}},
    "detents": {"dial": {"off": 0.0, "low": 0.01, "high": 0.02}, "wheel": {"up": 0.0, "down": 3.0}},
}


def kitchen() -> Articulation:
    return Articulation.parse(KITCHEN)


def test_families_sort_every_movable_joint_into_one_value_family_plus_q_and_setting():
    assert kitchen().families() == {
        "openness": ("door", "drawer", "flap", "lid"),
        "pos": ("dial", "knob"),
        "angle": ("wheel",),
        "setting": ("dial", "wheel"),
        "q": ("dial", "door", "drawer", "flap", "knob", "lid", "wheel"),
    }
    assert set(cabinet().families()) == {"openness", "q"}


def test_field_names_cover_state_target_predicates_and_every_family_field():
    assert kitchen().field_names() == {
        "state",
        "target",
        "moving",
        "blocked",
        "closed",
        "open",
        *(f"openness.{name}" for name in ("door", "drawer", "flap", "lid")),
        "pos.dial",
        "pos.knob",
        "angle.wheel",
        "setting.dial",
        "setting.wheel",
        *(f"q.{name}" for name in ("dial", "door", "drawer", "flap", "knob", "lid", "wheel")),
    }


@pytest.mark.parametrize(
    ("values", "field", "expected"),
    [
        ({"door": 0.75}, "openness.door", 0.5),
        ({"door": 1.57}, "openness.door", 1.0),
        ({"drawer": 0.07}, "openness.drawer", 0.2),
        ({"lid": 0.6}, "openness.lid", 0.5),
        ({"lid": -1.2}, "openness.lid", 1.0),
        ({"lid": -1.5}, "openness.lid", 1.0),
        ({"flap": -0.4}, "openness.flap", 0.5),
        ({"flap": -1.0}, "openness.flap", 1.0),
        ({"knob": 0.5}, "pos.knob", 0.0),
        ({"knob": 2.0}, "pos.knob", 0.75),
        ({"dial": 0.01}, "pos.dial", 0.5),
        ({"wheel": -math.pi / 2}, "angle.wheel", 3 * math.pi / 2),
        ({"wheel": 5 * math.pi / 2}, "angle.wheel", math.pi / 2),
        ({"wheel": -1e-20}, "angle.wheel", 0.0),
        ({"knob": 2.0}, "q.knob", 2.0),
        ({"wheel": -0.3}, "q.wheel", -0.3),
    ],
)
def test_continuous_fields_read_the_joint_values(values, field, expected):
    continuous, _ = kitchen().fields(values)
    assert continuous[field] == pytest.approx(expected)
    assert 0.0 <= continuous["angle.wheel"] < 2 * math.pi


def test_setting_names_the_detent_within_tolerance_and_is_empty_between():
    articulation = kitchen()
    assert articulation.fields({"dial": 0.0105, "wheel": 3.0 - 2 * math.pi})[1] == {"setting.dial": "low", "setting.wheel": "down"}
    assert articulation.fields({"dial": 0.015, "wheel": 1.0})[1] == {"setting.dial": "", "setting.wheel": ""}
    assert articulation.fields({"dial": 0.0, "wheel": 0.0005})[1] == {"setting.dial": "off", "setting.wheel": "up"}


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("target", "open", {"door": 1.5, "drawer": 0.35, "lid": 1.2, "flap": -0.8}),
        ("openness", "0.5", {"door": 0.75, "drawer": 0.175, "lid": 0.6, "flap": -0.4}),
        ("openness.lid", "1", {"lid": 1.2}),
        ("pos.knob", "0.75", {"knob": 2.0}),
        ("setting.dial", " high ", {"dial": 0.02}),
        ("setting.wheel", "down", {"wheel": 3.0}),
        ("q.drawer", "9", {"drawer": 0.4}),
        ("q.wheel", str(3 * math.pi / 2), {"wheel": -math.pi / 2}),
    ],
)
def test_target_for_moves_only_the_joints_the_field_names(field, value, expected):
    articulation = kitchen()
    base = articulation.initial()
    target = articulation.target_for(field, value, base)
    assert target == pytest.approx({**base, **expected})
    assert base == articulation.initial()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target", "ajar", "unknown state 'ajar', states: closed, open"),
        ("openness", "1.5", "openness takes a number in \\[0, 1\\], got '1.5'"),
        ("openness", "nan", "openness takes a number in \\[0, 1\\]"),
        ("openness.door", "wide", "openness.door takes a number in \\[0, 1\\], got 'wide'"),
        ("pos.knob", "-0.1", "pos.knob takes a number in \\[0, 1\\]"),
        ("pos.door", "0.5", "field 'pos.door' names no pos joint, pos joints: dial, knob"),
        ("setting.dial", "max", "setting.dial takes a detent name of dial: off, low, high, got 'max'"),
        ("setting.knob", "off", "field 'setting.knob' names no setting joint, setting joints: dial, wheel"),
        ("q.drawer", "inf", "q.drawer takes a number"),
        ("q.lever", "1", "field 'q.lever' names no q joint"),
        ("angle.wheel", "1", "field 'angle.wheel' is not writable, writable: openness, openness.door"),
        ("moving", "true", "field 'moving' is not writable"),
        ("door", "1", "field 'door' is not writable"),
    ],
)
def test_target_for_names_the_field_and_what_it_accepts(field, value, message):
    with pytest.raises(ValueError, match=message):
        kitchen().target_for(field, value, kitchen().initial())


def test_openness_needs_a_joint_its_closed_and_open_states_set_apart():
    articulation = Articulation.parse({"links": {"base": [], "lid": []}, "joints": [{"name": "lid", "type": "continuous", "parent": "base", "child": "lid"}]})
    assert "openness" not in articulation.writable_fields()
    with pytest.raises(ValueError, match="openness needs a joint the closed and open states set apart"):
        articulation.target_for("openness", "0.5", articulation.initial())
    with pytest.raises(ValueError, match="openness needs a joint"):
        articulation.resolve(0.5)


def test_resolve_takes_an_openness_and_family_mappings():
    articulation = kitchen()
    initial = articulation.initial()
    assert articulation.resolve(1) == pytest.approx({**initial, "door": 1.5, "drawer": 0.35, "lid": 1.2, "flap": -0.8})
    assert articulation.resolve(0.5)["door"] == pytest.approx(0.75)
    assert articulation.resolve({"openness": {"door": 0.5}, "setting": {"dial": "low"}, "pos": {"knob": 1.0}, "q": {"wheel": 1.0}}) == pytest.approx({**initial, "door": 0.75, "dial": 0.01, "knob": 2.5, "wheel": 1.0})
    assert articulation.resolve({"door": 1.0, "lid": "open"}) == pytest.approx({**initial, "door": 1.0, "lid": 1.2})


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (1.5, "openness takes a number in \\[0, 1\\], got 1.5"),
        ({"openness": {"door": 0.5}, "lid": 1.0}, "joints mixes family keys \\['openness'\\] with joint names \\['lid'\\]"),
        ({"angle": {"wheel": 1.0}}, "joints family 'angle' is read-only, set openness, pos, setting, q"),
        ({"openness": 0.5}, "joints family openness maps its member joints to values, got 0.5"),
        ({"openness": {"knob": 0.5}}, "field 'openness.knob' names no openness joint"),
        ({"setting": {"dial": 0.01}}, "setting.dial takes a detent name of dial"),
        ({"door": True}, "joint door takes a number or a state name, got True"),
        ({"lever": 1.0}, "unknown movable joint 'lever'"),
    ],
)
def test_resolve_names_the_problem_of_a_joints_spec(spec, message):
    with pytest.raises(ValueError, match=message):
        kitchen().resolve(spec)


def _kitchen_with(change) -> dict:
    value = {
        "links": dict(KITCHEN["links"]),
        "joints": [dict(joint) for joint in KITCHEN["joints"]],
        "states": {k: dict(v) for k, v in KITCHEN["states"].items()},
        "detents": {k: dict(v) for k, v in KITCHEN["detents"].items()},
    }
    change(value)
    return value


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value["states"].update(openness={"door": 0.0}), "state name 'openness' is taken, avoid angle, blocked, moving, openness, pos, q, setting, state, target and the joint names"),
        (lambda value: value["states"].update(q={"door": 0.0}), "state name 'q' is taken"),
        (lambda value: value["joints"][4].update(name="pos"), "joint name 'pos' is reserved, avoid angle, blocked, moving, openness, pos, q, setting, state, target"),
        (lambda value: value["joints"][4].update(name="target"), "joint name 'target' is reserved"),
        (lambda value: value["detents"]["dial"].update(setting=0.0), "detent name 'setting' is reserved, avoid angle, blocked, moving, openness, pos, q, setting, state, target"),
        (lambda value: value["detents"]["dial"].update(Max=0.0), "detent name must match"),
        (lambda value: value["detents"]["dial"].update(far=0.05), "detent far of dial at 0.05 is outside its limit \\[0.0, 0.02\\]"),
        (lambda value: value["detents"].update(lever={"up": 0.0}), "detents name 'lever', which is no movable joint"),
        (lambda value: value["detents"].update(dial={}), "detents of joint 'dial' must map detent names to values"),
    ],
)
def test_reserved_names_and_bad_detents_are_refused(change, message):
    with pytest.raises(ValueError, match=message):
        Articulation.parse(_kitchen_with(change))


def test_detents_round_trip_and_are_left_out_when_empty():
    articulation = kitchen()
    assert articulation.unstructure()["detents"] == {"dial": {"off": 0.0, "low": 0.01, "high": 0.02}, "wheel": {"up": 0.0, "down": 3.0}}
    assert Articulation.parse(articulation.unstructure()).unstructure() == articulation.unstructure()
    assert "detents" not in cabinet().unstructure()


def test_transformed_scales_prismatic_detents_and_keeps_angular_ones():
    transform = np.eye(4)
    transform[:3, :3] *= 0.5
    moved = kitchen().transformed(transform)
    assert moved.detents["dial"] == pytest.approx({"off": 0.0, "low": 0.005, "high": 0.01})
    assert moved.detents["wheel"] == {"up": 0.0, "down": 3.0}
    assert moved.fields({"dial": 0.005})[1]["setting.dial"] == "low"


def test_family_members_lists_one_family_of_a_snapshot_without_naming_its_joints():
    continuous, _ = cabinet().fields({"door": 0.75, "drawer": 0.4})
    assert family_members(continuous, "openness") == {"door": pytest.approx(0.5), "drawer": pytest.approx(1.0)}
    assert family_members(continuous, "angle") == {}


def test_role_dynamics_and_materials_round_trip():
    value = {
        **CABINET,
        "joints": [
            {**CABINET["joints"][0], "role": "door", "dynamics": {"friction": 0.4, "damping": 1.5, "armature": 0.02, "spring": {"stiffness": 3.0, "rest": 0.1}}},
            {**CABINET["joints"][1], "role": "drawer", "dynamics": {"friction_coefficient": 0.2}},
        ],
        "materials": {"door": {"static_friction": 0.6, "dynamic_friction": 0.4, "restitution": 0.1}, "drawer": {"dynamic_friction": 0.3}},
    }
    articulation = Articulation.parse(value)
    door, drawer = articulation.joint_map["door"], articulation.joint_map["drawer"]
    assert (door.role, door.dynamics.friction, door.dynamics.spring.stiffness, door.dynamics.spring.rest) == ("door", 0.4, 3.0, 0.1)
    assert (drawer.role, drawer.dynamics.friction, drawer.dynamics.friction_coefficient) == ("drawer", None, 0.2)
    assert articulation.materials["drawer"].static_friction is None
    raw = articulation.unstructure()
    assert raw["joints"][0]["dynamics"] == {"friction": 0.4, "damping": 1.5, "armature": 0.02, "spring": {"stiffness": 3.0, "rest": 0.1}}
    assert raw["joints"][1]["dynamics"] == {"friction_coefficient": 0.2}
    assert raw["materials"] == {"door": {"static_friction": 0.6, "dynamic_friction": 0.4, "restitution": 0.1}, "drawer": {"dynamic_friction": 0.3}}
    assert Articulation.parse(raw).unstructure() == raw


def test_a_fixed_joint_takes_no_dynamics():
    with pytest.raises(ValueError, match="fixed joint glue takes no dynamics"):
        Joint(name="glue", type="fixed", parent="a", child="b", dynamics={"damping": 1.0})


def test_transformed_scales_a_prismatic_spring_rest_and_keeps_materials():
    value = {
        **CABINET,
        "joints": [CABINET["joints"][0], {**CABINET["joints"][1], "dynamics": {"spring": {"stiffness": 20.0, "rest": 30.0}}}],
        "materials": {"drawer": {"dynamic_friction": 0.3}},
    }
    moved = Articulation.parse(value).transformed(np.diag([0.01, 0.01, 0.01, 1.0]))
    spring = moved.joint_map["drawer"].dynamics.spring
    assert (spring.stiffness, spring.rest) == (20.0, pytest.approx(0.3))
    assert moved.materials["drawer"].dynamic_friction == 0.3


LATCHED_DOOR = {
    "links": {"frame": [], "leaf": ["leaf"], "lever": ["lever"]},
    "joints": [
        {"name": "hinge", "type": "revolute", "parent": "frame", "child": "leaf", "origin": [0.45, 0.0, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.6], "role": "door", "latch": {"by": "handle", "release": 0.6}},
        {"name": "handle", "type": "revolute", "parent": "leaf", "child": "lever", "origin": [-0.35, 0.03, 1.0], "axis": [0, 1, 0], "limit": [0.0, 0.8], "role": "handle"},
    ],
    "states": {"closed": {"hinge": 0.0}, "open": {"hinge": 1.5}},
}


def test_a_latch_round_trips_and_survives_the_floor_transform():
    articulation = Articulation.parse(LATCHED_DOOR)
    raw = articulation.unstructure()
    assert raw["joints"][0]["latch"] == {"by": "handle", "release": 0.6}
    assert "latch" not in raw["joints"][1]
    assert Articulation.parse(raw).unstructure() == raw
    assert articulation.transformed(np.diag([0.01, 0.01, 0.01, 1.0])).joint_map["hinge"].latch == articulation.joint_map["hinge"].latch


def test_a_latch_holds_until_its_handle_passes_the_release_fraction():
    articulation = Articulation.parse(LATCHED_DOOR)
    hinge, handle = articulation.joint_map["hinge"], articulation.joint_map["handle"]
    assert articulation.latch_holds(hinge, {"hinge": 0.0, "handle": 0.0})
    assert articulation.latch_holds(hinge, {"hinge": 0.0, "handle": 0.47})
    assert not articulation.latch_holds(hinge, {"hinge": 0.0, "handle": 0.48})
    assert not articulation.latch_holds(handle, {"hinge": 0.0, "handle": 0.0})


@pytest.mark.parametrize(
    ("latch", "message"),
    [
        ({"by": "knob"}, "hinge latches on 'knob', which is no limited joint"),
        ({"by": "hinge"}, "hinge cannot latch on itself"),
        ({"by": "handle", "release": 1.5}, r"latch release takes a number in \[0, 1\]"),
        ({"by": "handle", "bolt": 0.2}, r"latch has unknown keys \['bolt'\]"),
    ],
)
def test_a_bad_latch_names_the_problem(latch, message):
    value = {**LATCHED_DOOR, "joints": [{**LATCHED_DOOR["joints"][0], "latch": latch}, LATCHED_DOOR["joints"][1]]}
    with pytest.raises(ValueError, match=message):
        Articulation.parse(value)


def test_a_fixed_joint_takes_no_latch():
    with pytest.raises(ValueError, match="fixed joint glue takes no latch"):
        Joint(name="glue", type="fixed", parent="a", child="b", latch={"by": "handle"})


def test_a_continuous_joint_takes_no_latch():
    with pytest.raises(ValueError, match="continuous joint wheel takes a latch only with a limit"):
        Joint(name="wheel", type="continuous", parent="a", child="b", latch={"by": "handle"})
