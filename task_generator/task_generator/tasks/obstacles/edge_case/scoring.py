"""Live sampling of an episode, reduced to a criticality row.

:mod:`criticality` is pure maths over a trajectory. This is the part that has to touch ROS:
it subscribes to ``arena_peds`` and the robot's pose, accumulates :class:`~criticality.Sample`
at a fixed rate, and on episode end writes one scored row next to the case record.

Kept deliberately separate so the metrics stay testable without a simulator, and so a failure
to sample degrades to "no score" rather than taking the episode down with it — a missing
measurement is bad, an episode aborted by its own instrumentation is worse.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import arena_people_msgs.msg
import rclpy.node
from arena_rclpy_mixins.shared import Namespace
from rclpy.parameter import Parameter

from .criticality import Criticality, Sample, score

if TYPE_CHECKING:
    from task_generator.manager.robot_manager.robot_manager import RobotManager

#: Sampling rate. 10 Hz resolves a 1.1 m/s walker to ~11 cm, well under the clearances being
#: measured, without the record growing faster than the episode.
DEFAULT_RATE_HZ = 10.0
#: Seconds a pose may stay exactly unchanged before the robot counts as standing rather than
#: as waiting for its next TF update.
STALE_POSE_S = 0.5


def differenced_velocity(
    prev: tuple[float, float],
    prev_vel: tuple[float, float],
    here: tuple[float, float],
    now: float,
    moved_t: float,
) -> tuple[tuple[float, float], float]:
    """Velocity from the pose stream, and when the pose last changed.

    A sample whose pose equals the previous one is a TF frame not yet refreshed, not a stop:
    it keeps the previous velocity until the pose has been unchanged for `STALE_POSE_S`, then
    reads as standing. A changed pose is differenced over the time since the last change.
    """
    if here == prev:
        return ((0.0, 0.0) if now - moved_t >= STALE_POSE_S else prev_vel), moved_t
    dt = now - moved_t
    if dt <= 1e-6:
        return prev_vel, now
    return ((here[0] - prev[0]) / dt, (here[1] - prev[1]) / dt), now

#: Pedestrian radius used for every clearance and TTC computation, metres.
#:
#: ``arena_people_msgs/Pedestrian`` carries name, id, pose, twist and animation state - there
#: is **no radius field**, so this is not a fallback, it is the value always used. Clearance
#: and TTC are therefore measured against a nominal body, not each agent's configured
#: ``agent_radius``; a case that tails `agent_radius` will not see that tail reflected here.
DEFAULT_PED_RADIUS = 0.3

#: Injected agents are named ``edge_0..edge_N`` (see impl._INJECT_PREFIX). Matched by prefix
#: rather than exact name because the spawn path may namespace or suffix the entity.
INJECT_PREFIX = "edge_"


class EpisodeScorer(rclpy.node.Node):
    """Samples one episode and reduces it to a criticality row.

    Same process and executor as the task generator, and the tick timer runs on sim time, so
    sampling is automatically gated while the simulator is paused - a paused episode must not
    accumulate freeze time.
    """

    def __init__(
        self,
        robot_manager: RobotManager,
        *,
        rate_hz: float = DEFAULT_RATE_HZ,
        peds_topic: str | None = None,
        peds_source: Callable[[], object] | None = None,
    ) -> None:
        super().__init__(
            "edge_case_scorer",
            namespace=str(robot_manager.namespace),
            # Mirrors CollisionTrackerNode: sim time so the tick is gated while the simulator
            # is paused (a paused episode must not accumulate freeze time), and no global
            # arguments so the task generator's own CLI remaps do not land on this node.
            use_global_arguments=False,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        self._rm = robot_manager
        self._samples: list[Sample] = []
        self._saw_peds = False
        self._peds_warned = False
        self._moved_t = 0.0
        self._peds: dict[str, tuple[tuple[float, float], tuple[float, float], float]] = {}
        self._last: tuple[float, float] | None = None

        # `Namespace(node.get_namespace())`, not `robot_manager.namespace`: pedestrians are published
        # on the env namespace, a level above the robot manager's, as CollisionTrackerNode does.
        env_ns = Namespace(robot_manager.node.get_namespace())
        self._peds_topic = peds_topic or str(env_ns("arena_peds"))
        self._peds_sub = self.create_subscription(arena_people_msgs.msg.Pedestrians, self._peds_topic, self._on_peds, 10)
        self._peds_retry_t: float | None = None
        #: Direct in-process roster read when the human simulator lives in this process: no
        #: middleware to fail.
        self._peds_source = peds_source
        self._timer = self.create_timer(1.0 / max(rate_hz, 1e-3), self._tick)

    # Collection
    # ----------

    @staticmethod
    def _roster_from_msg(msg: arena_people_msgs.msg.Pedestrians) -> dict[str, tuple[tuple[float, float], tuple[float, float], float]]:
        peds = {}
        for p in msg.pedestrians:
            peds[p.name] = (
                (p.pose.position.x, p.pose.position.y),
                (p.twist.linear.x, p.twist.linear.y),
                DEFAULT_PED_RADIUS,
            )
        return peds

    def _on_peds(self, msg: arena_people_msgs.msg.Pedestrians) -> None:
        self._peds = self._roster_from_msg(msg)

    def _tick(self) -> None:
        pose = self._rm.pose
        if pose is None:
            return  # respawn window; not a sample, and not an error
        if self._peds_source is not None:
            # The direct read wins whenever it has content; the subscription stays as the
            # fallback for out-of-process human simulators and tests.
            try:
                msg = self._peds_source()
            except Exception:
                msg = None
            peds = getattr(msg, "pedestrians", None)
            if peds:
                self._peds = self._roster_from_msg(msg)
        now = self.get_clock().now().nanoseconds * 1e-9
        here = (pose.position.x, pose.position.y)

        # The pose stream carries no velocity, so it is differenced over the last sample whose pose
        # actually changed (TF can refresh slower than this timer).
        vel = (0.0, 0.0)
        if self._samples:
            prev = self._samples[-1]
            vel, self._moved_t = differenced_velocity(prev.robot, prev.robot_vel, here, now, self._moved_t)
        else:
            self._moved_t = now

        self._samples.append(
            Sample(
                t=now,
                robot=here,
                robot_vel=vel,
                robot_radius=float(self._rm.radius),
                peds=dict(self._peds),
            )
        )
        if not self._saw_peds and self._peds:
            self._saw_peds = True
            if self._peds_warned:
                self.get_logger().warn(f"edge_case scorer: arena_peds healed after {now - self._samples[0].t:.1f} s of silence ({len(self._peds)} peds)")
        if not self._saw_peds and now - self._samples[0].t > 10.0 and (self._peds_retry_t is None or now - self._peds_retry_t >= 10.0):
            # Recreate the subscription when it stays silent while the publisher keeps going.
            self._peds_retry_t = now
            try:
                self.destroy_subscription(self._peds_sub)
            except Exception:  # a dead handle must not kill the sampling tick
                pass
            self._peds_sub = self.create_subscription(arena_people_msgs.msg.Pedestrians, self._peds_topic, self._on_peds, 10)
            self.get_logger().warn(f"edge_case scorer: arena_peds silent for {now - self._samples[0].t:.0f} s - recreating the subscription")
        if not self._saw_peds and not self._peds_warned and now - self._samples[0].t > 15.0:
            # Warn while it happens, not only at flush time.
            self._peds_warned = True
            self.get_logger().warn("edge_case scorer: no pedestrians on arena_peds after 15 s of sampling - this episode will be unscorable")

    # Reduction
    # ---------

    @property
    def samples(self) -> list[Sample]:
        return self._samples

    def result(self, *, encounter: tuple[float, float] | None = None, only: Sequence[str] | None = None) -> Criticality:
        return score(self._samples, encounter=encounter, only=only)

    def injected_seen(self) -> list[str]:
        """Names on the stream that look like injected edge-case agents.

        The adapter publishes each pedestrian under its env-prefixed `sim_path`, so the bare
        `edge_0` is the last path component, not the whole name.
        """
        seen: set[str] = set()
        for sample in self._samples:
            seen.update(name for name in sample.peds if name.rsplit("/", 1)[-1].startswith(INJECT_PREFIX))
        return sorted(seen)

    def reset(self) -> None:
        self._samples.clear()
        self._peds.clear()
        self._saw_peds = False
        self._peds_warned = False

    def shutdown(self) -> None:
        self._timer.cancel()
        self.destroy_node()


def write_score(
    record_dir: Path | str,
    case: dict[str, Any],
    result: Criticality,
    *,
    scope: str = "all",
    scored_agents: Sequence[str] = (),
    ambient: Criticality | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Append one scored row to ``scores.jsonl``, keyed to its case.

    A separate file from ``cases.jsonl`` on purpose: a case is written when the episode is
    *built* and a score only exists once it has *run*, so forcing them into one row would mean
    either holding the case back or rewriting it. Joined on `(run_seed, episode_id)`.
    """
    out = Path(record_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scores.jsonl"

    row = {
        "run_seed": case.get("run_seed"),
        "episode_id": case.get("episode_id"),
        "world": case.get("world"),
        "seed": case.get("seed"),
        # Repeated so a scores file is readable alone, and so designed-vs-achieved sits side
        # by side - the comparison this whole panel exists to make.
        "scenario": case.get("scenario"),
        "case_id": case.get("case_id"),
        "prompt_id": case.get("prompt_id"),
        "steps": case.get("steps"),
        "designed_ttc": case.get("designed_ttc"),
        "min_ttc_s": result.min_ttc_s,
        "min_ttc_with": result.min_ttc_with,
        "min_ttc_at": result.min_ttc_at,
        "min_clearance_m": result.min_clearance_m,
        "min_clearance_with": result.min_clearance_with,
        "min_clearance_at": result.min_clearance_at,
        "pet_s": result.pet_s,
        "designed_pet": case.get("designed_pet"),
        "freeze_duration_s": result.freeze_duration_s,
        "intrusion_time_s": result.intrusion_time_s,
        "collided": result.collided,
        "collision_fault": result.collision_fault or None,
        "collision_with": result.collision_with or None,
        "collision_at": result.collision_at,
        "collision_robot_closing": result.collision_robot_closing,
        "collision_ped_closing": result.collision_ped_closing,
        "observed": result.observed,
        "samples": result.samples,
        "duration_s": result.duration_s,
        # What the panel above was actually computed over. Without this a row is ambiguous:
        # a min-TTC of 0.4 s means something different if it came from the injected agent
        # than if it came from whichever bystander happened to be closest.
        "scope": scope,
        "scored_agents": list(scored_agents),
        # The same panel over *every* pedestrian, recorded whenever the scope was narrower.
        # This is what makes "did the tail do it, or was the crowd already this tight?"
        # answerable from one file instead of requiring a separate baseline run.
        "min_ttc_s_ambient": None if ambient is None else ambient.min_ttc_s,
        "min_clearance_m_ambient": None if ambient is None else ambient.min_clearance_m,
        "collided_ambient": None if ambient is None else ambient.collided,
        # The ambient panel's own fault verdict, so an ambient contact is attributed rather than
        # grounded to 0; score60 decides when to use it.
        "collision_fault_ambient": None if ambient is None else (ambient.collision_fault or None),
    }
    if extra:
        # Never allowed to overwrite a measured field: a mis-keyed extra silently replacing
        # `min_ttc_s` would be indistinguishable from a real measurement.
        row.update({k: v for k, v in extra.items() if k not in row})
    with path.open("a") as fh:
        fh.write(json.dumps(row, separators=(",", ":"), sort_keys=True, default=str) + "\n")
    return path
