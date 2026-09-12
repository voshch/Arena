"""End-to-end: a generated prompt scenario runs in the real stack.

Boots `arena launch` (gazebo, arena_humansim, nav2, the edge_case modes) on one generated
scenario and checks the whole chain the demos rely on: the case builds, the robot sets off and
the case clock starts, every runtime effect fires, the population walks, the robot's lidar sees
the pedestrians and HumanSim is told where the robot is, the score row lands, and nothing
crashes. Minutes per test and it owns the machine's simulator, so it only runs with
`ARENA_SIM_TESTS=1` (see `data/promptgen/tools/sim_test.sh`).
"""
from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.slow]

WS = Path("/opt/arena_ws")
TOOLS = WS / "data/promptgen/tools"
WORLD = os.environ.get("ARENA_SIM_TEST_WORLD", "hospital_1")
SCENARIO = os.environ.get("ARENA_SIM_TEST_SCENARIO", "normal__hospital_1_001")
ROBOT = os.environ.get("ARENA_SIM_TEST_ROBOT", "jackal")
LOG = Path(f"/tmp/sim_test_{SCENARIO}.log")
BOOT_S = 420.0
DEPART_S = 40.0
EFFECTS_S = 90.0


def _skip_unless_enabled() -> None:
    if os.environ.get("ARENA_SIM_TESTS") != "1":
        pytest.skip("set ARENA_SIM_TESTS=1 to boot the simulator")
    pytest.importorskip("rclpy")
    if not (WS / "source").exists():
        pytest.skip("runs inside the project container")


def _log() -> str:
    return LOG.read_text(errors="replace") if LOG.exists() else ""


def _wait(pattern: str, timeout: float, *, since: int = 0) -> re.Match[str] | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        m = re.search(pattern, _log()[since:])
        if m:
            return m
        time.sleep(2.0)
    return None


@pytest.fixture(scope="module")
def stack():
    _skip_unless_enabled()
    subprocess.run(["bash", str(TOOLS / "killsim.sh")], capture_output=True, timeout=60)
    LOG.unlink(missing_ok=True)
    cmd = (
        f"source {WS}/source >/dev/null 2>&1; export GZ_IP=127.0.0.1 GZ_RELAY=127.0.0.1; cd {WS} && "
        f"arena launch sim:=gazebo world:={WORLD} robot:={ROBOT} human:=arena task.robots:=edge_case task.obstacles:=edge_case "
        f"task.edge_case_robot.retries:=3 task.edge_case_robot.on_blocked:=continue headless:=true task.auto_reset:=true "
        f"task.edge_case.record_dir:=/tmp/sim_test_rec/{SCENARIO} task.scenario.file:={SCENARIO} > {LOG} 2>&1"
    )
    proc = subprocess.Popen(["bash", "-lc", cmd], start_new_session=True)
    try:
        yield proc
    finally:
        # The launch's own process group first; killsim.sh then sweeps what outlives it. Note
        # that killsim.sh matches on command lines ("task_generator", "arena launch", ...): the
        # process running this test must not carry those words (see sim_test.sh).
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=20)
        except Exception:  # noqa: BLE001 - already gone, or stuck: the sweep below handles it
            pass
        subprocess.run(["bash", str(TOOLS / "killsim.sh")], capture_output=True, timeout=120)


@pytest.fixture(scope="module")
def episode(stack) -> dict:
    """The first episode the case is built for (episode 1 runs before the robot is placed)."""
    started = _wait(r"EPISODE STARTED #2", BOOT_S)
    assert started, f"no second episode within {BOOT_S:.0f}s; log tail:\n" + _log()[-3000:]
    log = _log()
    first = log.index("EPISODE STARTED #1")
    block = log[first:started.start()]
    built = re.search(rf"edge_case: {re.escape(SCENARIO)} built - (\d+) injected, (\d+) plan\(s\), (\d+) base agent\(s\)", block)
    no_effects = re.search(rf"-> {re.escape(SCENARIO)}: 0 effect\(s\)", block)
    assert built or no_effects, "the obstacles mode never reported the case built:\n" + block[-3000:]
    return {"at": started.start(), "plans": int(built.group(2)) if built else 0, "agents": int(built.group(3)) if built else 0}


def test_the_case_builds_without_abort(episode) -> None:
    block = _log()[: episode["at"]]
    assert "Traceback" not in block.split("EPISODE STARTED #1")[-1], "a traceback during the reset"
    assert not re.search(r"CaseAborted|aborting the case", block.split("EPISODE STARTED #1")[-1])


def test_the_robot_sets_off_and_the_case_clock_starts(episode) -> None:
    m = _wait(r"under way \(([0-9.]+) m from its start[^)]*\) ([0-9.]+)s after", DEPART_S, since=episode["at"])
    assert m, f"the robot did not set off within {DEPART_S:.0f}s of the episode start"
    assert float(m.group(2)) < DEPART_S


def test_every_runtime_effect_fires(episode) -> None:
    if episode["plans"] == 0:
        pytest.skip("this case has no runtime effect")
    armed = re.findall(r"edge_case: armed (\S+) over \d+ agent\(s\), first rewrite at t=([0-9.]+)s( once [^\n]{0,80})?", _log()[: episode["at"]])
    assert armed, "no plans armed"
    ungated = [(label, float(at)) for label, at, gate in armed if not gate]
    for label, at in ungated:
        if at > EFFECTS_S:
            continue  # beyond what this test waits for
        assert _wait(rf"edge_case: {re.escape(label)}(?: at t=| fired at t=| trigger fired)", EFFECTS_S + at, since=episode["at"]), f"{label} (due at {at:.0f}s) never fired"


def test_the_population_walks_and_the_robot_is_seen_and_sees(episode) -> None:
    import rclpy
    from arena_humansim_msgs.msg import AgentStates
    from arena_people_msgs.msg import Pedestrians
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import LaserScan
    import tf2_ros

    if not rclpy.ok():
        rclpy.init()  # the ros conftest may already have done it
    node = Node("sim_test_probe")
    tf = tf2_ros.Buffer()
    tf2_ros.TransformListener(tf, node)

    def topic(suffix: str) -> str:
        for _ in range(40):
            names = sorted(n for n, _ in node.get_topic_names_and_types() if n.endswith(suffix))
            if names:
                return names[0]
            rclpy.spin_once(node, timeout_sec=0.5)
        raise AssertionError(f"no topic ending in {suffix}")

    first: dict[str, tuple[float, float]] = {}
    path: dict[str, float] = {}
    last: dict[str, tuple[float, float]] = {}
    peds: dict[str, tuple[float, float]] = {}
    world_state = [0]
    scans: list[LaserScan] = []

    def on_peds(msg):
        for p in msg.pedestrians:
            xy = (p.pose.position.x, p.pose.position.y)
            peds[p.name] = xy
            if p.name in last:
                path[p.name] = path.get(p.name, 0.0) + math.dist(last[p.name], xy)
            else:
                first[p.name] = xy
            last[p.name] = xy

    node.create_subscription(Pedestrians, topic("/arena_peds"), on_peds, 10)
    node.create_subscription(AgentStates, topic("/world_state"), lambda m: world_state.__setitem__(0, world_state[0] + 1), 10)
    node.create_subscription(LaserScan, topic(f"/{ROBOT}/lidar"), lambda m: (scans.clear(), scans.append(m)), QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))

    seen: list[tuple[str, float, int]] = []
    t0 = time.time()
    while time.time() - t0 < 30.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if scans and peds and int((time.time() - t0) * 2) % 2 == 0:
            s = scans[-1]
            try:
                t = tf.lookup_transform("map", s.header.frame_id, rclpy.time.Time())
            except Exception:
                continue
            q = t.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            rx, ry = t.transform.translation.x, t.transform.translation.y
            pts = [(rx + d * math.cos(yaw + s.angle_min + i * s.angle_increment), ry + d * math.sin(yaw + s.angle_min + i * s.angle_increment)) for i, d in enumerate(s.ranges) if s.range_min < d < s.range_max]
            for name, (px, py) in peds.items():
                dist = math.hypot(px - rx, py - ry)
                if 0.8 < dist < 4.0:
                    seen.append((name, dist, sum(1 for x, y in pts if math.hypot(x - px, y - py) < 0.45)))
    node.destroy_node()

    assert len(peds) >= 2, "no pedestrians published"
    walkers = [n for n in peds if not re.search(r"normal_[1-4]$|edge_\d+$|d_\d+$", n)]  # talkers and designed standers may stand
    moving = [n for n in walkers if path.get(n, 0.0) > 2.0]
    assert len(moving) >= max(1, len(walkers) // 2), f"population stands: {[(n, round(path.get(n, 0), 1)) for n in walkers]}"
    assert world_state[0] >= 20, f"HumanSim got {world_state[0]} world_state messages in 30 s: the robot's pose is not fed back"
    assert seen, "no pedestrian passed within 0.8-4 m of the lidar during the sample"
    hit = [h for _, _, h in seen if h >= 3]
    assert len(hit) >= len(seen) // 3, f"the lidar returns nothing from pedestrians: {seen[:12]}"


def test_the_episode_ends_with_a_score_row(episode) -> None:
    finished = _wait(r"EPISODE FINISHED #2", 300.0, since=episode["at"])
    assert finished, "the single leg did not finish within 300 s"
    rows = Path(f"/tmp/sim_test_rec/{SCENARIO}/scores.jsonl")
    deadline = time.time() + 20
    while time.time() < deadline and not rows.exists():
        time.sleep(1)
    assert rows.exists() and rows.read_text().strip(), "no score row written for the finished episode"
    tail = _log()[episode["at"]:]
    assert "Traceback" not in tail.replace("KeyboardInterrupt", ""), "a traceback during the episode"
