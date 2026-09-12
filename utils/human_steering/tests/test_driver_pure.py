from __future__ import annotations

from human_steering import integrate
from human_steering.driver import (
    RUN_THRESHOLD_MPS,
    ClipState,
    Driver,
    Intent,
    IntentStore,
    _empty_manifest_due,
    _roster_state_label,
    _roster_waypoint_progress,
    _viz_matches,
    auto_state,
)


def test_driver_module_imports_without_ros() -> None:
    # the module-level guards (GaitGenerator/LIMITS, arena_people_msgs, ...) must
    # keep driver.py importable even when no ROS install is sourced.
    import human_steering.driver as driver_module

    assert driver_module is not None


def test_auto_state_idle_below_threshold() -> None:
    assert auto_state(0.0) == 0


def test_auto_state_walking_between_thresholds() -> None:
    assert auto_state(0.5) == 1


def test_auto_state_running_at_and_above_threshold() -> None:
    assert auto_state(RUN_THRESHOLD_MPS) == 2
    assert auto_state(RUN_THRESHOLD_MPS + 1.0) == 2


def test_intent_default_is_not_held() -> None:
    assert Intent().held is False


def test_driver_authoring_actions_claim_the_ped() -> None:
    driver = _stub_driver()
    driver.teleop_input("a", 1.0, 0.0, 0.0, now_wall=0.0)
    driver.engage_joint("b", "waist", 0.1)
    driver.set_gaze("c", (1.0, 2.0, 0.0))
    driver.set_waypoints("d", [(1.0, 0.0)], loop=False, speed=1.0)
    driver.set_state_override("e", 2)
    assert driver.held_names() == {"a", "b", "c", "d", "e"}


def test_driver_clearing_actions_do_not_claim() -> None:
    driver = _stub_driver()
    driver.set_state_override("a", None)
    driver.set_gaze("b", None)
    driver.disengage_joint("c", "waist")
    driver.stop("d")
    assert driver.held_names() == set()


def test_intent_store_resync_seeds_idle_ped() -> None:
    store = IntentStore()
    store.resync("alice", 1.0, 2.0, 0.3)
    assert store.pose("alice") == (1.0, 2.0, 0.3)


def test_intent_store_resync_skips_held_ped() -> None:
    store = IntentStore()
    store.get("alice").held = True
    store.set_pose("alice", 5.0, 5.0, 0.0)
    store.resync("alice", 1.0, 2.0, 0.3)
    # the bus position must not clobber the ped's own integrated pose while held.
    assert store.pose("alice") == (5.0, 5.0, 0.0)


def test_intent_store_teleport_overrides_regardless_of_mode() -> None:
    store = IntentStore()
    store.get("alice").mode = "waypoints"
    store.set_pose("alice", 5.0, 5.0, 0.0)
    store.teleport("alice", 9.0, 9.0, 1.0)
    # GUI-initiated teleport reseeds the integrator immediately, unlike resync.
    assert store.pose("alice") == (9.0, 9.0, 1.0)


def test_intent_store_teleport_keeps_old_yaw_if_not_given() -> None:
    store = IntentStore()
    store.set_pose("alice", 0.0, 0.0, 0.75)
    store.teleport("alice", 1.0, 1.0)
    assert store.pose("alice") == (1.0, 1.0, 0.75)


def test_intent_store_episode_change_clears_intents() -> None:
    store = IntentStore()
    store.get("alice").mode = "waypoints"
    store.set_pose("alice", 1.0, 1.0, 0.0)

    # first observation only seeds the episode id, nothing to clear yet.
    assert store.on_episode(1) is False
    assert store.get("alice").mode == "waypoints"

    # same episode id again: still no clear.
    assert store.on_episode(1) is False
    assert store.get("alice").mode == "waypoints"

    # a genuine episode change clears every intent and pose seed.
    assert store.on_episode(2) is True
    assert store.get("alice").mode == "idle"
    assert store.pose("alice") is None


def test_intent_store_forget_drops_intent_and_pose() -> None:
    store = IntentStore()
    store.get("alice").mode = "teleop"
    store.set_pose("alice", 1.0, 1.0, 0.0)
    store.forget("alice")
    assert "alice" not in store
    assert store.pose("alice") is None


def test_intent_store_held_lists_only_claimed_peds() -> None:
    store = IntentStore()
    store.get("untouched_ped")
    store.get("held_ped").held = True
    names = {name for name, _intent in store.held()}
    assert names == {"held_ped"}


def test_intent_store_expire_teleop_stops_motion_but_keeps_claim() -> None:
    store = IntentStore()
    intent = store.get("alice")
    intent.held = True
    intent.mode = "teleop"
    intent.teleop_twist = (1.0, 0.0, 0.0)
    intent.teleop_last_cmd_wall = 0.0
    dropped = store.expire_teleop("alice", now_wall=1.0, deadman_s=0.5)
    assert dropped is True
    assert intent.mode == "idle"
    assert intent.teleop_twist == (0.0, 0.0, 0.0)
    assert intent.held is True


def test_intent_store_expire_teleop_keeps_fresh_command() -> None:
    store = IntentStore()
    intent = store.get("alice")
    intent.mode = "teleop"
    intent.teleop_last_cmd_wall = 0.0
    dropped = store.expire_teleop("alice", now_wall=0.1, deadman_s=0.5)
    assert dropped is False
    assert intent.mode == "teleop"


def test_roster_state_label_from_animation_state() -> None:
    assert _roster_state_label(None, 0) == "IDLE"
    assert _roster_state_label(None, 1) == "WALKING"
    assert _roster_state_label(None, 2) == "RUNNING"


def test_roster_state_label_teleop_overrides_animation_state() -> None:
    assert _roster_state_label(Intent(mode="teleop"), 1) == "TELEOP"


def test_roster_state_label_non_teleop_intent_uses_animation_state() -> None:
    assert _roster_state_label(Intent(mode="waypoints"), 2) == "RUNNING"


def test_roster_waypoint_progress_none_when_not_driving_waypoints() -> None:
    assert _roster_waypoint_progress(None) is None
    assert _roster_waypoint_progress(Intent(mode="teleop")) is None


def test_roster_waypoint_progress_none_when_total_unset() -> None:
    assert _roster_waypoint_progress(Intent(mode="waypoints")) is None


def test_roster_waypoint_progress_non_looping_climbs_and_clamps() -> None:
    intent = Intent(mode="waypoints", loop=False)
    intent.waypoint_total = 3
    intent.waypoints = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    assert _roster_waypoint_progress(intent) == "wp 1/3"
    intent.waypoint_cursor = 1
    assert _roster_waypoint_progress(intent) == "wp 2/3"
    intent.waypoint_cursor = 3
    assert _roster_waypoint_progress(intent) == "wp 3/3"


def test_roster_waypoint_progress_looping_reports_total() -> None:
    intent = Intent(mode="waypoints", loop=True)
    intent.waypoint_total = 2
    intent.waypoints = [(0.0, 0.0), (1.0, 0.0)]
    assert _roster_waypoint_progress(intent) == "loop (2 wp)"


def _stub_driver() -> Driver:
    """Bare Driver with only the ROS-free intent store wired up, for exercising the routing methods without a live node."""
    driver = object.__new__(Driver)
    driver._intents = IntentStore()
    driver._last_composed = {}
    return driver


def _arrive(intent: Intent, x: float, y: float) -> None:
    """Simulate one waypoint-following tick from a pose already at the target."""
    _nx, _ny, _nyaw, cursor = integrate.advance_waypoints(
        x,
        y,
        0.0,
        intent.waypoints,
        intent.waypoint_cursor,
        intent.speed,
        dt=1.0,
        loop=intent.loop,
    )
    intent.waypoint_cursor = cursor


def test_reported_bug_append_after_arrival_does_not_resend_through_visited_point() -> None:
    """route [A, B], arrive at A, append C: order continues B, C, A, never back through A first."""
    driver = _stub_driver()
    point_a, point_b, point_c = (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)

    driver.set_waypoints("alice", [point_a, point_b], loop=True, speed=1.0)
    intent = driver._intents.get("alice")

    _arrive(intent, *point_a)
    assert intent.waypoints[intent.waypoint_cursor] == point_b

    driver.append_waypoint("alice", point_c, speed=1.0)
    assert driver.waypoints("alice") == [point_a, point_b, point_c]
    assert intent.waypoints[intent.waypoint_cursor] == point_b  # append keeps the current target

    _arrive(intent, *point_b)
    assert intent.waypoints[intent.waypoint_cursor] == point_c

    _arrive(intent, *point_c)
    assert intent.waypoints[intent.waypoint_cursor] == point_a


def test_driver_stop_clears_route_and_cursor_but_keeps_claim() -> None:
    driver = _stub_driver()
    driver.set_waypoints("alice", [(0.0, 0.0), (1.0, 0.0)], loop=True, speed=1.0)
    driver.stop("alice")
    assert driver.waypoints("alice") == []
    intent = driver._intents.get("alice")
    assert intent.mode == "idle"
    assert intent.waypoint_cursor == 0
    assert "alice" in driver.held_names()


def test_driver_append_waypoint_starts_fresh_route_when_idle() -> None:
    driver = _stub_driver()
    driver.append_waypoint("alice", (1.0, 2.0), speed=1.0)
    assert driver.waypoints("alice") == [(1.0, 2.0)]
    intent = driver._intents.get("alice")
    assert intent.mode == "waypoints"
    assert intent.loop is True
    assert intent.waypoint_cursor == 0
    assert intent.waypoint_total == 1


def test_driver_release_drops_claim_and_whole_intent() -> None:
    driver = _stub_driver()
    driver.set_waypoints("alice", [(1.0, 0.0)], loop=True, speed=1.0)
    intent = driver._intents.get("alice")
    intent.posed["waist"] = 0.2
    intent.gaze = (1.0, 2.0, 0.0)
    intent.clip = ClipState(name="wave", t0_engine=0.0)
    intent.clip_release = ({"waist": 0.1}, 0.0)
    intent.teleop_twist = (1.0, 0.5, 0.2)
    driver._last_composed["alice"] = {"waist": 0.2}

    driver.release("alice")

    assert "alice" not in driver.held_names()
    assert "alice" not in driver._intents
    assert driver.waypoints("alice") == []
    assert "alice" not in driver._last_composed


def test_reported_bug_teleop_deadman_does_not_release_claim() -> None:
    """Stopping key input stops motion, the engine must not reclaim the ped before an explicit release."""
    driver = _stub_driver()
    driver.teleop_input("alice", 1.0, 0.0, 0.0, now_wall=0.0)
    driver._intents.expire_teleop("alice", now_wall=10.0)
    assert "alice" in driver.held_names()


def test_driver_held_names_reflects_local_claim_set() -> None:
    driver = _stub_driver()
    driver._intents.get("untouched_ped")
    driver.set_waypoints("held_ped", [(1.0, 0.0)], loop=False, speed=1.0)
    assert driver.held_names() == {"held_ped"}


def test_empty_manifest_due_only_on_nonempty_to_empty_transition() -> None:
    assert _empty_manifest_due(0, 0) is False
    assert _empty_manifest_due(0, 3) is False
    assert _empty_manifest_due(2, 1) is False
    assert _empty_manifest_due(1, 0) is True


def test_viz_matches_node_ns_or_env_ns_or_basename() -> None:
    node_ns = "/arena/env_0/task_generator_node"
    assert _viz_matches(node_ns, node_ns)
    assert _viz_matches(node_ns, "/arena/env_0")
    assert _viz_matches(node_ns, "arena/env_0")
    assert _viz_matches(node_ns, "env_0")
    assert not _viz_matches(node_ns, "env_1")
