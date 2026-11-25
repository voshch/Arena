from __future__ import annotations

import json
from types import SimpleNamespace


class _Collector:
    def __init__(self) -> None:
        self.sent: list[object] = []

    def publish(self, msg: object) -> None:
        self.sent.append(msg)


def _sim(seconds: float):
    from arena_rclpy_mixins.Time import Time

    return Time.from_float(seconds)


def _stub(sim_seconds: float, start_seconds: float | None):
    from arena_runtime.env.episode_node import EpisodeNode, EpisodeRuntime
    from arena_runtime.env.task_board import TaskBoard

    stub = object.__new__(type("Stub", (EpisodeNode,), {"sim_time": _sim(0.0)}))
    stub.sim_time = _sim(sim_seconds)
    stub._episodes = EpisodeRuntime()
    if start_seconds is not None:
        stub._episodes.current.start_time = _sim(start_seconds)
    stub._episode_conditions = []
    stub._human_params = {}
    stub._pub_task_events = _Collector()
    stub._pub_state_episode = _Collector()
    stub._task_layer = SimpleNamespace(mode_namespaces=dict)
    stub._task_layer_param = SimpleNamespace(value="casper")
    stub._task_board = TaskBoard()
    stub._robots_manager = None
    stub.rosparam = {str: SimpleNamespace(get=lambda name, default: default)}
    return stub


def test_events_are_stamped_with_sim_seconds_since_the_episode_start_in_report_order() -> None:
    from arena_runtime.env.task_layer import TaskEvent

    stub = _stub(sim_seconds=17.5, start_seconds=10.0)
    stub.task_event(TaskEvent(task="fetch", kind="assigned", robot="tiago", detail={"score": 0.5}))
    stub.sim_time = _sim(20.0)
    stub.task_event(TaskEvent(task="fetch", kind="started", robot="tiago"))

    assert [(e["task"], e["kind"], e["robot"], e["t"], e["detail"]) for e in stub._episodes.current.tasks] == [
        ("fetch", "assigned", "tiago", 7.5, {"score": 0.5}),
        ("fetch", "started", "tiago", 10.0, {}),
    ]


def test_events_before_the_episode_starts_are_stamped_zero() -> None:
    from arena_runtime.env.task_layer import TaskEvent

    stub = _stub(sim_seconds=42.0, start_seconds=None)
    stub.task_event(TaskEvent(task="", kind="allocated", detail={"mode": "greedy"}))
    assert stub._episodes.current.tasks[0]["t"] == 0.0


def test_events_are_published_with_json_detail_and_the_episode_record_carries_them() -> None:
    from arena_runtime.env.task_layer import TaskEvent

    stub = _stub(sim_seconds=12.0, start_seconds=10.0)
    stub.task_event(TaskEvent(task="fetch", kind="met", robot="tiago", detail={"phases": 3}))
    stub.task_event(TaskEvent(task="wipe", kind="dropped", robot="p3at"))

    sent = stub._pub_task_events.sent
    assert [(m.task, m.kind, m.robot, m.t) for m in sent] == [("fetch", "met", "tiago", 2.0), ("wipe", "dropped", "p3at", 2.0)]
    assert json.loads(sent[0].detail) == {"phases": 3}
    assert json.loads(sent[1].detail) == {}

    assert stub._pub_state_episode.sent == []
    record = stub._record_to_msg(stub._episodes.current, {})
    assert json.loads(record.tasks) == [
        {"task": "fetch", "kind": "met", "robot": "tiago", "t": 2.0, "detail": {"phases": 3}},
        {"task": "wipe", "kind": "dropped", "robot": "p3at", "t": 2.0, "detail": {}},
    ]


def test_a_queued_record_carries_no_tasks() -> None:
    from arena_runtime.env.task_layer import EpisodeRecord, TaskEvent

    stub = _stub(sim_seconds=12.0, start_seconds=10.0)
    stub.task_event(TaskEvent(task="fetch", kind="met"))
    assert json.loads(stub._record_to_msg(EpisodeRecord(episode_id=2), {}).tasks) == []
    assert len(json.loads(stub._record_to_msg(stub._episodes.current, {}).tasks)) == 1


def _activity(phase: object, held: bool = False) -> str:
    from arena_runtime.env.episode_node import EpisodeNode

    state = SimpleNamespace(phase=phase, monitor=SimpleNamespace(held=held))
    return EpisodeNode._phase_activity(SimpleNamespace(runner=SimpleNamespace(active_state=state)))


def test_an_in_place_hold_reads_as_holding_before_it_is_held() -> None:
    from arena_simulation_setup.shared.task import GoToPhase
    from arena_simulation_setup.utils.geometry import Orientation, Pose, Position

    assert _activity(GoToPhase(hold_time=5.0)) == "holding"
    assert _activity(GoToPhase(pose=Pose(Position(1.0, 2.0, 0.0), Orientation.identity()))) == "navigating"
    assert _activity(GoToPhase(target="kitchen"), held=True) == "holding"
