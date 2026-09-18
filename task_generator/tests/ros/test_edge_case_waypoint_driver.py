from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


HOMES = {"a": (0.0, 0.0), "b": (10.0, 0.0)}


class _Harness:
    """A driver whose clock, reroutes and robot pose are all under the test's control."""

    def __init__(self, *, reroute_result=True, reroute_raises=None, submit_raises=None, pose=(5.0, 5.0), depart_after_m=None):
        from task_generator.tasks.obstacles.edge_case.waypoint_driver import WaypointDriver

        self.sent: list[dict] = []
        self.submitted: list[object] = []
        self._result = reroute_result
        self._raises = reroute_raises
        self._submit_raises = submit_raises
        self.pose = pose
        self.now = 0.0

        async def reroute(routes):
            if self._raises is not None:
                raise self._raises
            self.sent.append({k: list(v) for k, v in routes.items()})
            return self._result

        def submit(coroutine):
            if self._submit_raises is not None:
                raise self._submit_raises
            self.submitted.append(coroutine)

        self.retuned: list[dict] = []

        async def retune(changes):
            self.retuned.append({k: dict(v) for k, v in changes.items()})
            return len(changes)

        self.driver = WaypointDriver("/pytest_waypoints", reroute=reroute, robot_pose=lambda: self.pose, submit=submit, retune=retune, depart_after_m=depart_after_m)
        self.driver._now = lambda: self.now

    def arm(self, *entries, origin=None):
        # t0 is taken on the first tick, not on arm; prime at an epsilon (an exact zero reads
        # as "the sim clock has not started").
        self.now = 1e-6
        self.driver.arm([e if isinstance(e, tuple) else (e, None, None) for e in entries], origin=origin)
        self.driver._tick()

    def run(self, at):
        self.now = at
        self.driver._tick()
        for coroutine in self.submitted:
            asyncio.run(coroutine)
        self.submitted.clear()

    def close(self):
        for coroutine in self.submitted:
            coroutine.close()
        self.driver.shutdown()


@pytest.fixture()
def harness():
    h = _Harness()
    yield h
    h.close()


def _plan(mode, homes=None, **kw):
    from task_generator.tasks.obstacles.edge_case.waypoints import Mode, Plan

    return Plan(mode=Mode(mode), homes=dict(HOMES if homes is None else homes), **kw)


def _rally(target=(1.0, 1.0), **kw):
    return _plan("rally", targets={n: target for n in HOMES}, **kw)


def test_the_clock_starts_on_the_first_tick_not_on_arm(harness) -> None:
    harness.driver.arm([(_rally(at=5.0), None, None)])
    harness.run(at=1000.0)
    assert harness.sent == []
    harness.run(at=1006.0)
    assert harness.sent == [{"a": [(1.0, 1.0)], "b": [(1.0, 1.0)]}]


def test_a_rally_fires_once_and_stays_fired(harness) -> None:
    harness.arm(_rally(target=(3.0, 4.0)))
    for t in (1.0, 2.0, 20.0, 200.0):
        harness.run(at=t)
    assert len(harness.sent) == 1
    (outcome,) = harness.driver.outcomes()
    assert outcome.issued == 2 and outcome.t_first == pytest.approx(1.0 - 1e-6)


def test_track_robot_re_issues_the_current_pose(harness) -> None:
    harness.arm(_plan("track_robot", period=2.0))
    harness.run(at=1.0)
    harness.pose = (7.0, 7.0)
    harness.run(at=1.5)  # inside the period: nothing
    harness.run(at=3.5)
    assert harness.sent == [{"a": [(5.0, 5.0)], "b": [(5.0, 5.0)]}, {"a": [(7.0, 7.0)], "b": [(7.0, 7.0)]}]


def test_track_robot_waits_out_a_respawn_window(harness) -> None:
    harness.pose = None
    harness.arm(_plan("track_robot", period=1.0))
    harness.run(at=2.0)
    assert harness.sent == []
    harness.pose = (1.0, 1.0)
    harness.run(at=3.0)
    assert len(harness.sent) == 1


def test_track_robot_releases_after_its_duration(harness) -> None:
    harness.arm(_plan("track_robot", period=1.0, duration=3.0, resume={"a": (9.0, 9.0)}))
    harness.run(at=1.0)
    harness.run(at=2.5)
    harness.run(at=4.5)
    assert harness.sent[-1] == {"a": [(9.0, 9.0)]}
    (outcome,) = harness.driver.outcomes()
    assert outcome.t_release == pytest.approx(4.5 - 1e-6)
    harness.run(at=8.0)
    assert len(harness.sent) == 3


def test_shuffle_re_issues_a_fresh_point_each_period(harness) -> None:
    harness.arm(_plan("shuffle", period=2.0, radius=1.0))
    harness.run(at=1.0)
    harness.run(at=3.5)
    assert len(harness.sent) == 2 and harness.sent[0]["a"] != harness.sent[1]["a"]


def test_scatter_fires_once(harness) -> None:
    harness.arm(_plan("shuffle", period=1.0, radius=1.0, once=True))
    for t in (1.0, 3.0, 9.0):
        harness.run(at=t)
    assert len(harness.sent) == 1


def test_hold_pins_then_releases(harness) -> None:
    harness.arm(_plan("hold", at=2.0, duration=3.0, points={"a": (4.0, 4.0)}, resume={"a": (0.0, 9.0), "b": (10.0, 9.0)}))
    harness.run(at=1.0)
    assert harness.sent == []
    harness.run(at=2.5)
    assert harness.sent == [{"a": [(4.0, 4.0)], "b": [(10.0, 0.0)]}]
    harness.run(at=4.0)
    assert len(harness.sent) == 1
    harness.run(at=5.5)
    assert harness.sent[-1] == {"a": [(0.0, 9.0)], "b": [(10.0, 9.0)]}


def test_hold_without_duration_pins_for_good(harness) -> None:
    harness.arm(_plan("hold", duration=0.0))
    for t in (1.0, 50.0, 500.0):
        harness.run(at=t)
    assert len(harness.sent) == 1


def test_a_gated_plan_waits_for_the_robot(harness) -> None:
    from task_generator.tasks.obstacles.edge_case.triggers import Trigger

    harness.pose = (50.0, 50.0)
    harness.arm((_rally(at=1.0), Trigger(robot_within=3.0), (0.0, 0.0)))
    harness.run(at=10.0)
    harness.run(at=20.0)
    assert harness.sent == []
    harness.pose = (1.0, 1.0)
    harness.run(at=30.0)  # trigger latches; the plan's clock starts here
    harness.run(at=30.5)
    assert harness.sent == []
    harness.run(at=31.5)
    assert len(harness.sent) == 1
    (outcome,) = harness.driver.outcomes()
    assert outcome.t_trigger == pytest.approx(30.0 - 1e-6) and "within 3" in outcome.trigger


def test_an_unfired_gate_is_reported_as_not_run(harness) -> None:
    from task_generator.tasks.obstacles.edge_case.triggers import Trigger

    harness.pose = (50.0, 50.0)
    harness.arm((_rally(), Trigger(robot_within=3.0), (0.0, 0.0)))
    harness.run(at=100.0)
    (outcome,) = harness.driver.outcomes()
    assert outcome.t_trigger is None and outcome.issued == 0


def test_plans_run_independently(harness) -> None:
    from task_generator.tasks.obstacles.edge_case.triggers import Trigger

    harness.pose = (50.0, 50.0)
    harness.arm(_rally(at=1.0), (_plan("hold", homes={"c": (2.0, 2.0)}, duration=0.0), Trigger(robot_within=1.0), (2.0, 2.0)))
    harness.run(at=2.0)
    assert harness.sent == [{"a": [(1.0, 1.0)], "b": [(1.0, 1.0)]}]
    harness.pose = (2.5, 2.0)
    harness.run(at=3.0)
    harness.run(at=3.5)
    assert harness.sent[-1] == {"c": [(2.0, 2.0)]}
    rally, hold = harness.driver.outcomes()
    assert rally.issued == 2 and hold.issued == 1 and hold.t_trigger is not None


def test_a_failed_reroute_is_counted_not_fatal() -> None:
    h = _Harness(reroute_raises=RuntimeError("service down"))
    try:
        h.arm(_plan("shuffle", period=1.0))
        h.run(at=1.0)
        h.run(at=2.5)
        (outcome,) = h.driver.outcomes()
        assert outcome.failures == 2 and outcome.issued == 0
    finally:
        h.close()


def test_a_reroute_reaching_nobody_is_counted() -> None:
    h = _Harness(reroute_result=False)
    try:
        h.arm(_rally())
        h.run(at=1.0)
        (outcome,) = h.driver.outcomes()
        assert outcome.failures == 1 and outcome.issued == 0
    finally:
        h.close()


def test_a_scheduling_failure_disables_the_driver_and_says_why() -> None:
    h = _Harness(submit_raises=RuntimeError("loop closed"))
    try:
        h.arm(_plan("shuffle", period=1.0))
        h.run(at=1.0)
        h.run(at=5.0)
        (outcome,) = h.driver.outcomes()
        assert "loop closed" in outcome.reason and h.sent == []
    finally:
        h.close()


def test_arming_nothing_disarms(harness) -> None:
    harness.arm(_rally())
    harness.driver.arm([])
    harness.run(at=5.0)
    assert harness.sent == [] and harness.driver.outcomes() == []


def test_a_retune_fires_once_through_the_hook(harness) -> None:
    harness.arm(_plan("retune", at=3.0, retune={"a": {"agent_type": "./types/x.yaml", "desired_velocity": 0.6}, "b": {"agent_type": "elder"}}))
    harness.run(at=2.0)
    assert harness.retuned == []
    harness.run(at=3.5)
    harness.run(at=9.0)
    assert len(harness.retuned) == 1 and harness.retuned[0]["a"]["desired_velocity"] == 0.6
    (outcome,) = harness.driver.outcomes()
    assert outcome.issued == 2 and outcome.t_first == pytest.approx(3.5 - 1e-6) and harness.sent == []


def test_the_case_clock_waits_for_the_robot_to_set_off() -> None:
    h = _Harness(depart_after_m=0.3)
    try:
        h.arm(_rally(at=1.0))
        for t in (5.0, 10.0, 40.0):
            h.run(at=t)  # the robot stands where it was armed: nothing runs
        assert h.sent == [] and h.driver.clock_started_at is None
        h.pose = (5.2, 5.0)
        h.run(at=41.0)  # 0.2 m is not under way yet
        assert h.sent == []
        h.pose = (5.5, 5.0)
        h.run(at=42.0)  # under way: the clock starts here, the rally is due 1 s later
        assert h.sent == [] and h.driver.clock_started_at == pytest.approx(42.0)
        h.run(at=43.0)
        assert len(h.sent) == 1
        (outcome,) = h.driver.outcomes()
        assert outcome.t_first == pytest.approx(1.0)
    finally:
        h.close()


def test_a_respawning_robot_does_not_start_the_clock() -> None:
    h = _Harness(depart_after_m=0.3, pose=None)
    try:
        h.arm(_rally(at=0.0))
        h.run(at=5.0)
        assert h.sent == [] and h.driver.clock_started_at is None
        h.pose = (0.0, 0.0)
        h.run(at=6.0)  # first sighting fixes the origin
        h.pose = (1.0, 0.0)
        h.run(at=7.0)  # under way: the clock starts, the plan's own clock on the next tick
        h.run(at=7.5)
        assert len(h.sent) == 1 and h.driver.clock_started_at == pytest.approx(7.0)
    finally:
        h.close()


def test_the_reset_teleport_moves_the_origin_instead_of_starting_the_clock() -> None:
    h = _Harness(depart_after_m=0.3, pose=(20.0, 20.0))  # where the robot ended the last episode
    try:
        h.arm(_rally(at=0.0))
        h.run(at=1.0)
        h.pose = (5.0, 5.0)  # the reset puts it on its start: a 21 m jump between ticks
        h.run(at=1.5)
        h.run(at=2.0)
        assert h.sent == [] and h.driver.clock_started_at is None
        h.pose = (5.4, 5.0)  # now it drives
        h.run(at=2.5)
        h.run(at=3.0)
        assert len(h.sent) == 1 and h.driver.clock_started_at == pytest.approx(2.5)
    finally:
        h.close()


def test_a_known_start_ignores_the_reset_teleport_however_short() -> None:
    h = _Harness(depart_after_m=0.3, pose=(5.6, 5.0))  # 0.6 m from the start: where episode 1 left it
    try:
        h.arm(_rally(at=0.0), origin=(5.0, 5.0))
        h.run(at=1.0)
        h.pose = (5.05, 5.0)  # the reset puts it on its start (a 0.55 m hop, below the teleport threshold)
        h.run(at=1.5)
        h.run(at=2.0)
        assert h.sent == [] and h.driver.clock_started_at is None
        h.pose = (5.4, 5.0)
        h.run(at=2.5)
        h.run(at=3.0)
        assert len(h.sent) == 1 and h.driver.clock_started_at == pytest.approx(2.5)
    finally:
        h.close()


def test_a_robot_never_placed_on_its_start_departs_from_where_it_is() -> None:
    h = _Harness(depart_after_m=0.3, pose=(9.0, 9.0))
    try:
        h.arm(_rally(at=0.0), origin=(5.0, 5.0))
        for t in (2.0, 6.0, 9.0):
            h.run(at=t)
        assert h.sent == []
        h.run(at=11.0)  # settle timeout: origin becomes (9, 9)
        h.pose = (9.5, 9.0)
        h.run(at=12.0)
        h.run(at=13.0)
        assert len(h.sent) == 1
    finally:
        h.close()
