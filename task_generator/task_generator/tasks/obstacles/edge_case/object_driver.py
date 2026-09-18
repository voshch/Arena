"""Fires a resolved object timeline during an episode.

Separated from :mod:`objects` for the same reason :mod:`scoring` is separated from
:mod:`criticality`: the arithmetic stays testable without a simulator, and this file holds
the part that has to touch ROS.

Modelled on :class:`~scoring.EpisodeScorer`, and for the same reasons:

* **Its own node, on sim time.** The tick is gated while the simulator is paused, so a
  paused episode does not race through a timeline. Wall time would fire every event during
  the first pause.
* **Its own failure boundary.** A timeline that cannot fire disables itself and says so; it
  never takes the episode down. The asymmetry that matters is between the two actions: a
  **spawn** that fails means the case did not run and is reported as a failure the mode can
  abort on, while a **despawn** that fails leaves an object behind on a case that otherwise
  did run, and is recorded as an anomaly instead.

The timer only *decides*; the work is handed to the task generator's event loop, because
spawning goes through the async environment manager and the timer callback is not a
coroutine.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

#: What a spawn or despawn hook hands back to be awaited on the task generator's loop.
Work = Coroutine[Any, Any, object]

import attrs
import rclpy.node
from rclpy.parameter import Parameter

from .objects import DESPAWN, SPAWN, ResolvedEvent

#: How often the timeline is checked, Hz. Sim seconds, so a 5 Hz tick resolves a 1.1 m/s
#: robot to ~22 cm - far finer than the metre-scale placements a timeline expresses, and
#: cheap enough that it costs nothing next to the 10 Hz scorer.
DEFAULT_RATE_HZ = 5.0


@attrs.define
class EventOutcome:
    """What actually happened to one event, for the case record."""

    entity: str
    action: str
    #: Sim seconds after the episode started, as fired. Differs from the designed `t` by up
    #: to one tick, and by more if the episode was paused - which is why it is recorded.
    t_fired: float | None = None
    t_designed: float = 0.0
    pose: tuple[float, float] | None = None
    reveal_distance_m: float | None = None
    #: The `sim_path` a spawn returned, i.e. what a despawn of it will name.
    entity_id: str = ""
    status: str = "pending"
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return attrs.asdict(self)


class ObjectDriver(rclpy.node.Node):
    """Ticks a resolved timeline and fires each event once."""

    def __init__(
        self,
        namespace: str,
        *,
        rate_hz: float = DEFAULT_RATE_HZ,
        spawn: Callable[[ResolvedEvent], Work],
        despawn: Callable[[ResolvedEvent, str], Work],
        submit: Callable[[Work], None],
    ) -> None:
        super().__init__(
            "edge_case_object_driver",
            namespace=namespace,
            # Same isolation the scorer uses: sim time so the tick is gated while paused,
            # and no global arguments so the task generator's own CLI remaps do not land here.
            use_global_arguments=False,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self._spawn = spawn
        self._despawn = despawn
        self._submit = submit

        self._events: list[ResolvedEvent] = []
        self._outcomes: list[EventOutcome] = []
        self._ids: dict[str, str] = {}
        self._next = 0
        self._t0: float | None = None
        self._broken = False

        self._timer = self.create_timer(1.0 / max(rate_hz, 1e-3), self._tick)

    # Lifecycle
    # ---------

    def arm(self, events: Sequence[ResolvedEvent]) -> None:
        """Load a timeline and start its clock from the next tick.

        The clock starts on the first *tick* rather than here, because `arm` is called at the
        end of a reset while the simulator may still be paused; taking the time now would
        make every early event fire the moment it resumes.
        """
        self._events = list(events)
        self._outcomes = [EventOutcome(entity=e.entity, action=e.action, t_designed=e.t) for e in self._events]
        self._ids = {}
        self._next = 0
        self._t0 = None
        self._broken = False

    def outcomes(self) -> list[EventOutcome]:
        """What happened to each event, in fire order."""
        return list(self._outcomes)

    def pending(self) -> int:
        """Events that never fired. Non-zero means the episode ended before the timeline did."""
        return sum(1 for o in self._outcomes if o.status == "pending")

    def note_result(self, index: int, *, entity_id: str = "", ok: bool = True, reason: str = "") -> None:
        """Record what the async half of an event did. Called back from the event loop."""
        if not 0 <= index < len(self._outcomes):
            return
        outcome = self._outcomes[index]
        outcome.status = "ok" if ok else "failed"
        outcome.reason = reason
        if entity_id:
            outcome.entity_id = entity_id
            self._ids[outcome.entity] = entity_id

    def shutdown(self) -> None:
        self._timer.cancel()
        self.destroy_node()

    # Ticking
    # -------

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _tick(self) -> None:
        if self._broken or self._next >= len(self._events):
            return

        now = self._now()
        if self._t0 is None:
            if now <= 0.0:
                return  # sim clock has not started publishing yet
            self._t0 = now
            return

        elapsed = now - self._t0
        # A loop, not an `if`: a pause that outlasts several events must fire all of them
        # rather than dropping the ones it skipped over.
        while self._next < len(self._events) and self._events[self._next].t <= elapsed:
            index = self._next
            self._next += 1
            self._fire(index, self._events[index], elapsed)
            if self._broken:
                # Whatever stopped the first event (a closed loop, a dead executor) will stop
                # the rest. Draining the backlog into the same failure just fills the log and
                # marks events failed that were never really attempted.
                return

    def _fire(self, index: int, event: ResolvedEvent, elapsed: float) -> None:
        outcome = self._outcomes[index]
        outcome.t_fired = elapsed
        outcome.pose = event.pose
        outcome.reveal_distance_m = event.reveal_distance_m

        coroutine: Work
        if event.action == SPAWN:
            coroutine = self._spawn(event)
        elif event.action == DESPAWN:
            entity_id = self._ids.get(event.entity, "")
            if not entity_id:
                # Its spawn failed or never fired. Not fatal: the object is not there, which
                # is what the despawn wanted, and saying so beats reporting a phantom removal.
                outcome.status = "skipped"
                outcome.reason = "nothing was spawned under this entity"
                return
            coroutine = self._despawn(event, entity_id)
        else:  # pragma: no cover - parse_event rejects anything else
            outcome.status = "failed"
            outcome.reason = f"unknown action {event.action!r}"
            return

        work = self._run(index, coroutine)
        try:
            self._submit(work)
        except Exception as exc:
            # Close both coroutines. Neither will ever run - the loop that would have run
            # them is what just failed - and an un-awaited coroutine left behind emits a
            # RuntimeWarning from wherever the garbage collector happens to reach it, which
            # is a confusing place to read about a scheduling failure.
            work.close()
            coroutine.close()
            self._broken = True
            outcome.status = "failed"
            outcome.reason = f"could not schedule: {exc!r}"
            self.get_logger().warn(f"edge_case objects: timeline disabled for this episode ({exc!r})")

    async def _run(self, index: int, coroutine: Work) -> None:
        try:
            result = await coroutine
        except Exception as exc:
            self.note_result(index, ok=False, reason=repr(exc))
            self.get_logger().warn(f"edge_case objects: event {index} failed ({exc!r})")
            return
        if isinstance(result, str):
            self.note_result(index, entity_id=result, ok=bool(result), reason="" if result else "spawn returned no id")
        else:
            self.note_result(index, ok=bool(result), reason="" if result else "the entity was not there")


def submit_to(loop: asyncio.AbstractEventLoop) -> Callable[[Work], None]:
    """Scheduler that hands a coroutine to `loop` from the timer's thread.

    `call_soon_threadsafe` rather than `create_task`: the timer runs on the executor, which
    is not necessarily the loop's thread, and `create_task` from the wrong thread is a
    silent no-op in the worst case.
    """

    def submit(coroutine: Work) -> None:
        loop.call_soon_threadsafe(lambda: loop.create_task(coroutine))

    return submit
