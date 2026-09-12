from arena_viz import DisplayKind, StyleSpec


def test_empty_json_yields_defaults():
    s = StyleSpec.from_json("")
    assert s == StyleSpec()


def test_roundtrip_defaults():
    s = StyleSpec()
    assert StyleSpec.from_json(s.to_json()) == s


def test_roundtrip_full():
    s = StyleSpec(
        color=(255, 0, 0),
        alpha=0.7,
        line_width=0.1,
        enabled=False,
        decay=0.3,
        extra={"rviz": {"Color Scheme": "costmap"}},
    )
    assert StyleSpec.from_json(s.to_json()) == s


def test_color_tuple_coerced():
    s = StyleSpec.from_json('{"color": [10, 20, 30]}')
    assert s.color == (10, 20, 30)


def test_all_kinds_enumerated():
    expected = {
        "map",
        "tf",
        "pedestrians",
        "marker_array",
        "robot_model",
        "odom",
        "laser_scan",
        "points3d",
        "image",
        "imu",
        "foot_contact",
        "path",
        "pose",
        "polygon",
        "trajectory",
        "planning_scene",
    }
    assert {k.value for k in DisplayKind} == expected
