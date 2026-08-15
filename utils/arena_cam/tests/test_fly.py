from __future__ import annotations

import math

from arena_cam import curves
from arena_cam.fly import FOV_RANGE, MIN_RADIUS, PITCH_LIMIT, SPEED_MPS, Fly, Intent, Mode, View

DT = 1.0 / 60.0


def run(fly: Fly, intent: Intent, seconds: float) -> None:
    for _ in range(int(seconds / DT)):
        fly.tick(DT, intent)


def test_damping_converges_to_the_commanded_speed() -> None:
    fly = Fly()
    run(fly, Intent(forward=1.0), 2.0)
    before = fly.pos
    fly.tick(DT, Intent(forward=1.0))
    step = curves.vlen(curves.vsub(fly.pos, before)) / DT
    assert abs(step - SPEED_MPS) < 0.01 * SPEED_MPS


def test_released_keys_coast_to_a_stop() -> None:
    fly = Fly()
    run(fly, Intent(forward=1.0), 1.0)
    run(fly, Intent(), 2.0)
    before = fly.pos
    fly.tick(DT, Intent())
    assert curves.vlen(curves.vsub(fly.pos, before)) < 1e-3 * DT


def test_fly_axes_are_view_frame_with_world_up() -> None:
    fly = Fly()
    fly.pos = (0.0, 0.0, 0.0)
    fly.pitch = 0.5
    run(fly, Intent(up=1.0), 1.0)
    assert abs(fly.pos[0]) < 1e-9
    assert abs(fly.pos[1]) < 1e-9
    assert fly.pos[2] > 0.0

    fly = Fly()
    fly.pos = (0.0, 0.0, 0.0)
    run(fly, Intent(right=1.0), 1.0)
    assert fly.pos[1] < 0.0
    assert abs(fly.pos[0]) < 1e-9
    assert abs(fly.pos[2]) < 1e-9


def test_dolly_follows_the_view_direction() -> None:
    fly = Fly()
    fly.pos = (0.0, 0.0, 0.0)
    fly.yaw = math.pi / 2.0
    fly.pitch = 0.3
    run(fly, Intent(forward=1.0), 1.0)
    travelled = curves.vnorm(fly.pos)
    for axis, expected in zip(travelled, fly.forward_vec(), strict=True):
        assert abs(axis - expected) < 1e-6


def test_pitch_clamps_short_of_vertical() -> None:
    fly = Fly()
    run(fly, Intent(pitch=1.0), 10.0)
    assert fly.pitch == PITCH_LIMIT
    run(fly, Intent(pitch=-1.0), 20.0)
    assert fly.pitch == -PITCH_LIMIT


def test_boost_and_crawl_scale_the_step() -> None:
    plain, boosted, crawling = Fly(), Fly(), Fly()
    for fly, intent in ((plain, Intent(forward=1.0)), (boosted, Intent(forward=1.0, boost=True)), (crawling, Intent(forward=1.0, crawl=True))):
        fly.pos = (0.0, 0.0, 0.0)
        run(fly, intent, 1.0)
    assert curves.vlen(boosted.pos) > curves.vlen(plain.pos) > curves.vlen(crawling.pos)


def test_orbit_keeps_the_subject_distance() -> None:
    fly = Fly()
    fly.frame((1.0, 2.0, 0.5), 5.0)
    run(fly, Intent(right=1.0, forward=0.4), 3.0)
    assert abs(curves.vlen(curves.vsub(fly.pos, fly.focus)) - 5.0) < 1e-9
    assert fly.mode is Mode.ORBIT


def test_orbit_azimuth_is_turntable_handed() -> None:
    """D swings the subject right, so the camera itself goes left."""
    fly = Fly()
    fly.frame((0.0, 0.0, 0.0), 5.0)
    fly.azimuth, fly.elevation = 0.0, 0.0
    # camera starts on +X looking back at the focus, so its own right is +Y
    run(fly, Intent(right=1.0), 0.5)
    assert fly.pos[1] < 0.0


def test_orbit_elevation_clamps_and_radius_stays_positive() -> None:
    fly = Fly()
    fly.frame((0.0, 0.0, 0.0), 5.0)
    run(fly, Intent(forward=1.0), 10.0)
    assert fly.elevation == PITCH_LIMIT
    run(fly, Intent(up=-1.0), 30.0)
    assert fly.radius >= MIN_RADIUS


def test_orbit_aims_at_the_focus() -> None:
    fly = Fly()
    fly.frame((2.0, -1.0, 0.0), 4.0)
    run(fly, Intent(right=1.0), 1.0)
    pos, quat, _fov = fly.pose()
    aim = curves.vnorm(curves.vsub(fly.focus, pos))
    for axis, expected in zip(curves.quat_forward(quat), aim, strict=True):
        assert abs(axis - expected) < 1e-6


def test_mode_round_trip_preserves_the_pose() -> None:
    fly = Fly()
    fly.seed((3.0, -2.0, 1.5), curves.look_at_quat((3.0, -2.0, 1.5), (0.0, 0.0, 0.5)))
    pos, quat, _fov = fly.pose()
    fly.toggle_mode()
    fly.toggle_mode()
    again_pos, again_quat, _again_fov = fly.pose()
    assert fly.mode is Mode.FLY
    for axis, expected in zip(again_pos, pos, strict=True):
        assert abs(axis - expected) < 1e-9
    for component, expected in zip(again_quat, quat, strict=True):
        assert abs(component - expected) < 1e-9


def test_seed_recovers_yaw_and_pitch() -> None:
    fly = Fly()
    eye, target = (4.0, 4.0, 3.0), (0.0, 1.0, 0.5)
    fly.seed(eye, curves.look_at_quat(eye, target))
    aim = curves.vnorm(curves.vsub(target, eye))
    for axis, expected in zip(fly.forward_vec(), aim, strict=True):
        assert abs(axis - expected) < 1e-9


def test_canonical_views_frame_the_focus() -> None:
    fly = Fly()
    fly.frame((1.0, 1.0, 0.0), 3.0)
    fly.canonical(View.TOP)
    assert fly.pos[2] > fly.focus[2]
    fly.canonical(View.FRONT)
    assert fly.pos[0] > fly.focus[0]
    assert abs(fly.pos[1] - fly.focus[1]) < 1e-9
    fly.canonical(View.RIGHT)
    assert fly.pos[1] > fly.focus[1]
    assert abs(fly.pos[0] - fly.focus[0]) < 1e-9


def test_fov_is_untouched_until_a_fov_key_is_pressed() -> None:
    fly = Fly()
    run(fly, Intent(forward=1.0), 1.0)
    assert fly.fov == 0.0
    run(fly, Intent(fov=1.0), 0.5)
    assert fly.fov > fly.default_fov
    run(fly, Intent(fov=1.0), 60.0)
    assert fly.fov == FOV_RANGE[1]


def test_stop_drops_the_carried_velocity() -> None:
    fly = Fly()
    run(fly, Intent(forward=1.0), 1.0)
    fly.stop()
    before = fly.pos
    fly.tick(DT, Intent())
    assert curves.vlen(curves.vsub(fly.pos, before)) < 1e-12
