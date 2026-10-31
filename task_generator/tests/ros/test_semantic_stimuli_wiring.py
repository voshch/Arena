"""SemanticChange -> human simulator wiring in the task generator node."""

from __future__ import annotations

import types


class _Human:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def notify_semantic_change(self, entity: str, field: str, value: str) -> None:
        self.calls.append((entity, field, value))

    @staticmethod
    def authored_name(realized: str) -> str:
        return realized.removeprefix("env_0/").removesuffix("/1")


def _node(human: _Human) -> object:
    from task_generator.manager.realizer import Realizer
    from task_generator.node import TaskGenerator

    stub = types.SimpleNamespace(
        _semantics_dirty=False,
        _semantic_names={"elev_a": "env_0/elev_a/1", "door_a": "env_0/door_a"},
        _realizer=Realizer(Realizer._Configuration(prefix="env_0")),
        _is_number=TaskGenerator._is_number,
        _human_simulator=human,
    )
    stub._forward_semantic_changes = lambda changes: TaskGenerator._forward_semantic_changes(stub, changes)
    return stub


def _change(entity: str, kind: str, field_kind: str, field: str, current: str) -> object:
    from arena_runtime.sim._semantics import SemanticChange

    return SemanticChange(entity, kind, field_kind, field, "", current)


def test_discrete_and_predicate_changes_reach_the_human_sim_under_authored_names() -> None:
    from task_generator.node import TaskGenerator

    human = _Human()
    node = _node(human)
    TaskGenerator._on_semantics_changed(
        node,
        [
            _change("env_0/elev_a/1", "elevator", "state", "cabin_door", "open"),
            _change("env_0/elev_a/1", "elevator", "state", "cabin_door_progress", "0.4"),
            _change("env_0/elev_a/1", "elevator", "predicate", "departing", "true"),
            _change("env_0/door_a", "door", "state", "state", "closed"),
            _change("env_0/door_a", "door", "state", "progress", "1.0"),
            _change("env_0/gate_x/1", "gate", "predicate", "locked", "false"),
            _change("env_0/zone_q", "unknown_kind", "state", "level", "3.5"),
            _change("env_0/zone_q", "unknown_kind", "state", "mode", "busy"),
        ],
    )
    assert node._semantics_dirty
    assert human.calls == [
        ("elev_a", "cabin_door", "open"),
        ("elev_a/1", "cabin_door", "open"),
        ("elev_a", "departing", "true"),
        ("elev_a/1", "departing", "true"),
        ("door_a", "state", "closed"),
        ("gate_x", "locked", "false"),
        ("gate_x/1", "locked", "false"),
        ("zone_q", "mode", "busy"),
    ]


def test_realized_values_are_env_stripped() -> None:
    from task_generator.node import TaskGenerator

    human = _Human()
    TaskGenerator._forward_semantic_changes(_node(human), [_change("env_0/elev_a/1", "elevator", "state", "target", "env_0/elev_b/2")])
    assert human.calls == [("elev_a", "target", "elev_b/2"), ("elev_a/1", "target", "elev_b/2")]


def test_semantics_change_marks_the_snapshot_dirty() -> None:
    from task_generator.node import TaskGenerator

    human = _Human()
    node = _node(human)
    TaskGenerator._on_semantics_changed(node, [_change("env_0/door_a", "door", "state", "state", "open")])
    assert node._semantics_dirty
    assert human.calls == [("door_a", "state", "open")]


def test_articulated_joint_values_stay_out_of_the_human_sim() -> None:
    from task_generator.node import TaskGenerator

    human = _Human()
    TaskGenerator._forward_semantic_changes(
        _node(human),
        [
            _change("env_0/fridge", "articulated", "state", "q.hinge", "0.4"),
            _change("env_0/fridge", "articulated", "state", "openness.hinge", "0.2"),
            _change("env_0/fridge", "articulated", "state", "state", "open"),
            _change("env_0/fridge", "articulated", "predicate", "moving", "false"),
        ],
    )
    assert human.calls == [("fridge", "state", "open"), ("fridge", "moving", "false")]
