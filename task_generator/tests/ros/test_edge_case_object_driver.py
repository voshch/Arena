"""The part of an object timeline that has to touch ROS: deciding when each event fires.

:mod:`objects` turns a timeline into `(time, pose)` pairs and is tested purely. This covers
what the driver does with them — fire-once semantics, what happens across a pause, and the
asymmetry between a spawn that fails (the case did not run) and a despawn that cannot find
its object (nothing to remove, which is what the caller wanted).

The clock is driven by hand rather than by a real timer: a test that waits for sim time to
advance is a test that is slow and flaky for no extra coverage.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _ros_gate():
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


def _events(*specs):
    """Resolve a tiny timeline. `specs` are (action, entity, t) triples."""
    from task_generator.tasks.obstacles.edge_case.objects import ObjectEvent, ResolvedEvent

    out = []
    for action, entity, t in specs:
        event = ObjectEvent(action=action, entity=entity, model="m" if action == "spawn" else "", at_t=t)
        out.append(ResolvedEvent(event=event, t=t, pose=(1.0, 2.0) if action == "spawn" else None))
    return out


class _Harness:
    """A driver whose clock, spawns and despawns are all under the test's control."""

    def __init__(self, *, spawn_result="env_0/cart", spawn_raises=None, submit_raises=None):
        from task_generator.tasks.obstacles.edge_case.object_driver import ObjectDriver

        self.spawned: list[str] = []
        self.despawned: list[str] = []
        self.submitted: list[object] = []
        self._spawn_result = spawn_result
        self._spawn_raises = spawn_raises
        self._submit_raises = submit_raises
        self.now = 0.0

        async def spawn(event):
            if self._spawn_raises is not None:
                raise self._spawn_raises
            self.spawned.append(event.entity)
            return self._spawn_result

        async def despawn(event, entity_id):
            self.despawned.append(entity_id)
            return True

        def submit(coroutine):
            if self._submit_raises is not None:
                # Deliberately NOT closing it here: cleaning up after a failed schedule is
                # the driver's job, and a harness that did it would hide a leak.
                raise self._submit_raises
            self.submitted.append(coroutine)

        self.driver = ObjectDriver("/pytest_objects", spawn=spawn, despawn=despawn, submit=submit)
        self.driver._now = lambda: self.now

    def tick(self, at=None):
        if at is not None:
            self.now = at
        self.driver._tick()

    def drain(self):
        """Run whatever the ticks scheduled, in order, as the event loop would."""
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


# Timing
# ------


def test_the_clock_starts_on_the_first_tick_not_on_arm(harness):
    """`arm` runs at the end of a reset, while the simulator may still be paused. Taking
    the time there would fire every early event the moment it resumes."""
    harness.driver.arm(_events(("spawn", "cart", 1.0)))
    harness.tick(at=1000.0)  # establishes t0; nothing is due yet

    assert harness.submitted == []

    harness.tick(at=1001.5)
    harness.drain()
    assert harness.spawned == ["cart"]


def test_a_clock_reading_of_zero_is_not_a_start(harness):
    """`use_sim_time` reports 0 until the clock topic produces something. Anchoring t0 there
    would make every event overdue the instant real sim time arrived."""
    harness.driver.arm(_events(("spawn", "cart", 1.0)))
    harness.tick(at=0.0)
    harness.tick(at=0.0)

    assert harness.submitted == []
    assert harness.driver.pending() == 1


def test_an_event_fires_once(harness):
    harness.driver.arm(_events(("spawn", "cart", 1.0)))
    harness.tick(at=100.0)
    for t in (101.0, 102.0, 103.0):
        harness.tick(at=t)
    harness.drain()

    assert harness.spawned == ["cart"]


def test_events_before_their_time_do_not_fire(harness):
    harness.driver.arm(_events(("spawn", "late", 30.0)))
    harness.tick(at=10.0)
    harness.tick(at=25.0)

    assert harness.submitted == []
    assert harness.driver.pending() == 1


def test_a_pause_that_outlasts_several_events_fires_all_of_them(harness):
    """A `while`, not an `if`. Dropping the skipped events would silently shorten a
    timeline, and the case record would still claim they were designed."""
    harness.driver.arm(_events(("spawn", "a", 1.0), ("spawn", "b", 2.0), ("spawn", "c", 3.0)))
    harness.tick(at=100.0)
    harness.tick(at=110.0)  # ten seconds later: all three are due
    harness.drain()

    assert harness.spawned == ["a", "b", "c"]
    assert harness.driver.pending() == 0


def test_pending_counts_what_never_fired(harness):
    harness.driver.arm(_events(("spawn", "a", 1.0), ("spawn", "b", 60.0)))
    harness.tick(at=100.0)
    harness.tick(at=102.0)
    harness.drain()

    assert harness.driver.pending() == 1, "an episode that ended before its timeline did"


def test_arming_a_new_timeline_forgets_the_old_one(harness):
    harness.driver.arm(_events(("spawn", "old", 1.0)))
    harness.tick(at=100.0)
    harness.driver.arm(_events(("spawn", "new", 1.0)))
    harness.tick(at=100.0)
    harness.tick(at=102.0)
    harness.drain()

    assert harness.spawned == ["new"]


# Spawn and despawn
# -----------------


def test_a_despawn_uses_the_id_its_spawn_returned(harness):
    """The author writes an entity name; the spawn returns a `sim_path`. Joining the two is
    the driver's job, so a timeline never has to know what the world will call the object."""
    harness.driver.arm(_events(("spawn", "cart", 1.0), ("despawn", "cart", 5.0)))
    harness.tick(at=100.0)
    harness.tick(at=102.0)
    harness.drain()
    harness.tick(at=106.0)
    harness.drain()

    assert harness.despawned == ["env_0/cart"]


def test_a_despawn_whose_spawn_failed_is_skipped_with_a_reason():
    """Not fatal: the object is not there, which is what the despawn wanted. Reporting a
    removal that never happened would be worse than saying nothing."""
    harness = _Harness(spawn_raises=RuntimeError("no such model"))
    try:
        harness.driver.arm(_events(("spawn", "cart", 1.0), ("despawn", "cart", 5.0)))
        harness.tick(at=100.0)
        harness.tick(at=102.0)
        harness.drain()
        harness.tick(at=106.0)
        harness.drain()

        spawn, despawn = harness.driver.outcomes()
        assert spawn.status == "failed"
        assert "no such model" in spawn.reason
        assert despawn.status == "skipped"
        assert "nothing was spawned" in despawn.reason
        assert harness.despawned == []
    finally:
        harness.close()


def test_a_spawn_that_returns_no_id_is_a_failure():
    """`extend` returning an empty string means nothing was placed."""
    harness = _Harness(spawn_result="")
    try:
        harness.driver.arm(_events(("spawn", "cart", 1.0)))
        harness.tick(at=100.0)
        harness.tick(at=102.0)
        harness.drain()

        (outcome,) = harness.driver.outcomes()
        assert outcome.status == "failed"
    finally:
        harness.close()


# Recording
# ---------


def test_outcomes_record_designed_and_fired_times(harness):
    """Same discipline as `approach_angle_requested` / `_achieved`: a timeline that fired
    late - because the episode was paused - must not read as one that fired on time."""
    harness.driver.arm(_events(("spawn", "cart", 3.0)))
    harness.tick(at=100.0)
    harness.tick(at=107.5)
    harness.drain()

    (outcome,) = harness.driver.outcomes()
    assert outcome.t_designed == 3.0
    assert outcome.t_fired == pytest.approx(7.5)
    assert outcome.status == "ok"
    assert outcome.entity_id == "env_0/cart"


def test_outcomes_are_serialisable():
    """They go into the score row, which is JSON."""
    import json

    harness = _Harness()
    try:
        harness.driver.arm(_events(("spawn", "cart", 1.0)))
        harness.tick(at=100.0)
        harness.tick(at=102.0)
        harness.drain()

        json.dumps([o.as_dict() for o in harness.driver.outcomes()])
    finally:
        harness.close()


# Failure boundary
# ----------------


def test_a_scheduling_failure_disables_the_timeline_rather_than_raising(recwarn):
    """The same discipline as the scorer: instrumentation degrades, it does not take the
    episode down. A raise here would surface inside a ROS timer callback, where nothing is
    positioned to handle it.

    `recwarn` guards the cleanup: an un-awaited coroutine left behind emits a RuntimeWarning
    from wherever the collector reaches it, which is a confusing place to learn that
    scheduling failed.
    """
    harness = _Harness(submit_raises=RuntimeError("loop is closed"))
    try:
        harness.driver.arm(_events(("spawn", "a", 1.0), ("spawn", "b", 2.0)))
        harness.tick(at=100.0)
        harness.tick(at=105.0)

        first, second = harness.driver.outcomes()
        assert first.status == "failed"
        assert "loop is closed" in first.reason
        assert second.status == "pending", "the timeline stops rather than retrying every tick"
    finally:
        harness.close()


def test_an_empty_timeline_is_inert(harness):
    harness.driver.arm([])
    harness.tick(at=100.0)
    harness.tick(at=1000.0)

    assert harness.driver.outcomes() == []
    assert harness.driver.pending() == 0
