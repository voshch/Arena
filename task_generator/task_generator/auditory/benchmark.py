from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import attrs
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from task_generator_msgs.msg import HeardSoundEvent, SoundEvent

from task_generator.auditory.qos_profiles import transient_event_qos


@attrs.define
class CpuSample:
    timestamp_sec: float
    cpu_percent: float
    process_count: int


@attrs.define
class RunMetrics:
    label: str
    duration_sec: float
    cpu_samples: list[CpuSample] = attrs.field(factory=list)
    sound_events: int = 0
    heard_events: int = 0
    injected_events: int = 0
    latencies_ms: list[float] = attrs.field(factory=list)

    def summary(self) -> dict[str, object]:
        cpu = [sample.cpu_percent for sample in self.cpu_samples]
        lat = sorted(self.latencies_ms)
        return {
            "label": self.label,
            "duration_sec": self.duration_sec,
            "cpu_percent_avg": _mean(cpu),
            "cpu_percent_max": max(cpu) if cpu else 0.0,
            "process_count_max": max((sample.process_count for sample in self.cpu_samples), default=0),
            "sound_events": self.sound_events,
            "heard_events": self.heard_events,
            "injected_events": self.injected_events,
            "latency_ms_avg": _mean(lat),
            "latency_ms_p50": _percentile(lat, 50.0),
            "latency_ms_p95": _percentile(lat, 95.0),
            "latency_ms_max": max(lat) if lat else 0.0,
        }


class AuditoryBenchmarkNode(Node):
    def __init__(
        self,
        *,
        sound_events_topic: str,
        heard_sound_events_topic: str,
        publish_events: bool,
    ) -> None:
        super().__init__("auditory_benchmark")

        self.sound_events = 0
        self.heard_events = 0
        self.injected_events = 0
        self.latencies_ms: list[float] = []
        self._event_seen_wall: dict[str, float] = {}

        self.create_subscription(
            SoundEvent,
            sound_events_topic,
            self._cb_sound_event,
            transient_event_qos(),
        )
        self.create_subscription(
            HeardSoundEvent,
            heard_sound_events_topic,
            self._cb_heard_event,
            transient_event_qos(),
        )
        self._publisher = (
            self.create_publisher(SoundEvent, sound_events_topic, transient_event_qos())
            if publish_events
            else None
        )

    def publish_synthetic_event(self) -> None:
        if self._publisher is None:
            return

        stamp = self.get_clock().now().to_msg()
        event_id = f"benchmark:{stamp.sec}:{stamp.nanosec}:{self.injected_events}"

        msg = SoundEvent()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.event_id = event_id
        msg.source_agent_id = -10_000
        msg.source_agent_name = "auditory_benchmark"
        msg.sound_type = "greeting"
        msg.label = "benchmark"
        msg.asset_id = "greeting"
        msg.source_position = Point(x=0.0, y=0.0, z=0.0)
        msg.source_volume_db = 60.0
        msg.duration.sec = 1
        msg.loop = False

        self._event_seen_wall[event_id] = time.monotonic()
        self.injected_events += 1
        self._publisher.publish(msg)

    def _cb_sound_event(self, msg: SoundEvent) -> None:
        self.sound_events += 1
        self._event_seen_wall.setdefault(msg.event_id, time.monotonic())

    def _cb_heard_event(self, msg: HeardSoundEvent) -> None:
        self.heard_events += 1
        start = self._event_seen_wall.get(msg.event_id)
        if start is not None:
            self.latencies_ms.append((time.monotonic() - start) * 1000.0)


def run_once(
    *,
    label: str,
    command: str | None,
    pid: int | None,
    duration_sec: float,
    sample_period_sec: float,
    sound_events_topic: str,
    heard_sound_events_topic: str,
    inject_rate_hz: float,
    startup_delay_sec: float,
) -> RunMetrics:
    process = None
    root_pid = pid

    if command:
        process = subprocess.Popen(
            command,
            shell=True,
            preexec_fn=os.setsid,
        )
        root_pid = process.pid
        _wait_for_startup(process, startup_delay_sec, label)

    if root_pid is None:
        raise ValueError("run_once requires either command or pid")

    node = AuditoryBenchmarkNode(
        sound_events_topic=sound_events_topic,
        heard_sound_events_topic=heard_sound_events_topic,
        publish_events=inject_rate_hz > 0.0,
    )

    metrics = RunMetrics(label=label, duration_sec=duration_sec)
    last_cpu_time = _process_tree_cpu_seconds(root_pid)
    last_sample_wall = time.monotonic()
    next_sample_wall = last_sample_wall
    next_publish_wall = last_sample_wall
    publish_period = 1.0 / inject_rate_hz if inject_rate_hz > 0.0 else math.inf
    deadline = last_sample_wall + duration_sec

    try:
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"{label} command exited during benchmark with code {process.returncode}"
                )

            now = time.monotonic()
            if now >= next_publish_wall:
                node.publish_synthetic_event()
                next_publish_wall += publish_period

            rclpy.spin_once(node, timeout_sec=0.02)

            now = time.monotonic()
            if now >= next_sample_wall:
                current_cpu_time = _process_tree_cpu_seconds(root_pid)
                elapsed = max(now - last_sample_wall, 1e-9)
                cpu_percent = ((current_cpu_time - last_cpu_time) / elapsed) * 100.0
                metrics.cpu_samples.append(
                    CpuSample(
                        timestamp_sec=now,
                        cpu_percent=max(cpu_percent, 0.0),
                        process_count=len(_process_tree(root_pid)),
                    )
                )
                last_cpu_time = current_cpu_time
                last_sample_wall = now
                next_sample_wall = now + sample_period_sec
    finally:
        metrics.sound_events = node.sound_events
        metrics.heard_events = node.heard_events
        metrics.injected_events = node.injected_events
        metrics.latencies_ms = node.latencies_ms
        node.destroy_node()

        if process is not None:
            _terminate_process_group(process)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CPU and auditory event latency for the same scenario with "
            "and without the auditory module."
        )
    )
    parser.add_argument("--baseline-cmd", help="Command that launches the scenario without auditory nodes.")
    parser.add_argument("--auditory-cmd", help="Command that launches the same scenario with auditory nodes.")
    parser.add_argument("--pid", type=int, help="Observe an already-running process tree instead of launching commands.")
    parser.add_argument("--label", default="observed", help="Label used with --pid.")
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--startup-delay-sec", type=float, default=10.0)
    parser.add_argument("--sample-period-sec", type=float, default=1.0)
    parser.add_argument("--sound-events-topic", default="human_sound_events")
    parser.add_argument("--heard-sound-events-topic", default="heard_sound_events")
    parser.add_argument(
        "--inject-rate-hz",
        type=float,
        default=0.0,
        help="Optionally publish synthetic SoundEvent messages during each run.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    if args.pid is None and not (args.baseline_cmd and args.auditory_cmd):
        parser.error("provide either --pid or both --baseline-cmd and --auditory-cmd")

    rclpy.init()
    try:
        runs: list[RunMetrics] = []
        if args.pid is not None:
            runs.append(
                run_once(
                    label=args.label,
                    command=None,
                    pid=args.pid,
                    duration_sec=args.duration_sec,
                    sample_period_sec=args.sample_period_sec,
                    sound_events_topic=args.sound_events_topic,
                    heard_sound_events_topic=args.heard_sound_events_topic,
                    inject_rate_hz=args.inject_rate_hz,
                    startup_delay_sec=0.0,
                )
            )
        else:
            runs.append(
                run_once(
                    label="baseline",
                    command=args.baseline_cmd,
                    pid=None,
                    duration_sec=args.duration_sec,
                    sample_period_sec=args.sample_period_sec,
                    sound_events_topic=args.sound_events_topic,
                    heard_sound_events_topic=args.heard_sound_events_topic,
                    inject_rate_hz=args.inject_rate_hz,
                    startup_delay_sec=args.startup_delay_sec,
                )
            )
            runs.append(
                run_once(
                    label="auditory",
                    command=args.auditory_cmd,
                    pid=None,
                    duration_sec=args.duration_sec,
                    sample_period_sec=args.sample_period_sec,
                    sound_events_topic=args.sound_events_topic,
                    heard_sound_events_topic=args.heard_sound_events_topic,
                    inject_rate_hz=args.inject_rate_hz,
                    startup_delay_sec=args.startup_delay_sec,
                )
            )
    except KeyboardInterrupt:
        print("benchmark interrupted", flush=True)
        return
    finally:
        if rclpy.ok():
            rclpy.shutdown()

    summaries = [run.summary() for run in runs]
    print(json.dumps(summaries, indent=2))

    if args.output_json:
        args.output_json.write_text(json.dumps(summaries, indent=2) + "\n")

    if args.output_csv:
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)


def _process_tree(root_pid: int) -> set[int]:
    seen: set[int] = set()
    pending = [root_pid]

    while pending:
        pid = pending.pop()
        if pid in seen or not Path(f"/proc/{pid}").exists():
            continue
        seen.add(pid)
        pending.extend(_child_pids(pid))

    return seen


def _child_pids(pid: int) -> list[int]:
    children_path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(value) for value in children_path.read_text().split()]
    except OSError:
        return []


def _process_tree_cpu_seconds(root_pid: int) -> float:
    ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    total_ticks = 0

    for pid in _process_tree(root_pid):
        stat_path = Path(f"/proc/{pid}/stat")
        try:
            stat = stat_path.read_text()
        except OSError:
            continue

        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) < 15:
            continue
        total_ticks += int(fields[11]) + int(fields[12])

    return float(total_ticks) / float(ticks_per_second)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)


def _wait_for_startup(process: subprocess.Popen[bytes], startup_delay_sec: float, label: str) -> None:
    deadline = time.monotonic() + startup_delay_sec

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{label} command exited during startup with code {process.returncode}"
            )
        time.sleep(0.2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    index = (len(values) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[int(index)]
    weight = index - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


if __name__ == "__main__":
    main()
