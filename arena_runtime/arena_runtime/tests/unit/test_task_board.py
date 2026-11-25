from __future__ import annotations

from arena_runtime.env.task_board import AgentEntry, TaskBoard


def event(task: str, kind: str, robot: str = "", t: float = 0.0, **detail: object) -> dict[str, object]:
    return {"task": task, "kind": kind, "robot": robot, "t": t, "detail": detail}


def states(board: TaskBoard) -> dict[str, tuple[str, str]]:
    return {entry.name: (entry.task, entry.state) for entry in board.agents()}


def test_allocation_lists_every_member_idle_sorted_by_name():
    board = TaskBoard()
    board.apply(event("a", "allocated", assignments={"a": {"member": "tb2"}, "b": {"member": "tb1"}, "c": {"member": "tb2"}}))

    assert board.agents() == [AgentEntry("tb1"), AgentEntry("tb2")]
    assert board.changed


def test_owned_task_is_assigned_then_working_with_its_zones():
    board = TaskBoard()
    board.apply(event("a", "assigned", "tb1", 1.0))
    assert states(board) == {"tb1": ("a", "assigned")}

    board.apply(event("a", "started", "tb1", 2.0, phases=3, zones=["kitchen", "hall"]))
    assert board.agents() == [AgentEntry("tb1", "a", "working", ("kitchen", "hall"), 2.0)]


def test_met_task_stays_on_its_agent_until_the_next_one_is_owned():
    board = TaskBoard()
    board.apply(event("a", "started", "tb1", 1.0, zones=["kitchen"]))
    board.apply(event("a", "met", "tb1", 5.0))
    assert board.agents() == [AgentEntry("tb1", "a", "met", ("kitchen",), 5.0)]

    board.apply(event("b", "assigned", "tb1", 6.0))
    assert states(board) == {"tb1": ("b", "assigned")}


def test_failed_task_reassigned_moves_to_the_successor_and_frees_the_previous_agent():
    board = TaskBoard()
    board.apply(event("a", "started", "tb1", 1.0, zones=["kitchen"]))
    board.apply(event("a", "failed", "tb1", 4.0))
    assert states(board) == {"tb1": ("a", "failed")}

    board.apply(event("a", "reassigned", "tb2", 5.0, **{"from": "tb1"}))
    assert states(board) == {"tb1": ("", "idle"), "tb2": ("a", "assigned")}

    board.apply(event("a", "started", "tb2", 6.0, zones=["kitchen"]))
    assert states(board) == {"tb1": ("", "idle"), "tb2": ("a", "working")}


def test_previous_agent_falls_back_to_its_next_owned_task_after_a_reassignment():
    board = TaskBoard()
    board.apply(event("a", "started", "tb1", 1.0))
    board.apply(event("b", "assigned", "tb1", 1.0))
    board.apply(event("a", "reassigned", "tb2", 3.0, **{"from": "tb1"}))

    assert states(board) == {"tb1": ("b", "assigned"), "tb2": ("a", "assigned")}


def test_dropped_task_marks_its_agent_dropped():
    board = TaskBoard()
    board.apply(event("a", "started", "tb1", 1.0))
    board.apply(event("a", "dropped", "tb1", 2.0))

    assert states(board) == {"tb1": ("a", "dropped")}


def test_released_task_is_owned_without_being_started():
    board = TaskBoard()
    board.apply(event("a", "released", "tb1", 1.0))

    assert states(board) == {"tb1": ("a", "assigned")}


def test_changed_is_raised_only_by_a_differing_entry():
    board = TaskBoard()
    board.apply(event("a", "assigned", "tb1", 1.0))
    board.changed = False

    board.apply(event("a", "assigned", "tb1", 2.0))
    assert not board.changed

    board.apply(event("a", "started", "tb1", 3.0))
    assert board.changed


def test_second_episode_starts_empty_after_reset():
    board = TaskBoard()
    board.apply(event("a", "started", "tb1", 1.0, zones=["kitchen"]))
    board.apply(event("a", "met", "tb1", 2.0))
    board.changed = False

    board.reset()
    assert board.agents() == []
    assert board.changed

    board.apply(event("a", "assigned", "tb2", 0.5))
    assert board.agents() == [AgentEntry("tb2", "a", "assigned", (), 0.5)]
