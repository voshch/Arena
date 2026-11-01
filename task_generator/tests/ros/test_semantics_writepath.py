"""M2-B semantics write path and scenario timeline: pure-logic coverage.

Follows the stub pattern used across the node-service tests: unbound methods are
invoked against a plain object carrying only the attributes each method reads.
"""

from __future__ import annotations

import random

import pytest


# ---------------------------------------------------------------------------
# environment_manager kind grouping
# ---------------------------------------------------------------------------


def test_route_semantics_splits_gate_and_plate_off_door() -> None:
    from arena_simulation_setup.shared.semantics import parse_semantics

    from task_generator.manager.environment_manager import _route_semantics

    cfgs = parse_semantics(
        [
            {"preset": "gate", "params": {"authorized": [], "unlock_on": "alarm"}},
            {"preset": "pressure_plate", "params": {"position": [1.0, 2.0]}},
        ],
    )
    extras = _route_semantics(cfgs, "door")
    assert sorted(extras) == ["gate", "pressure_plate"]
    assert {c.name for c in extras["gate"]} == {"locked", "blocked"}
    assert {c.name for c in extras["pressure_plate"]} == {"pressed"}


def test_route_semantics_splits_plate_off_elevator() -> None:
    from arena_simulation_setup.shared.semantics import parse_semantics

    from task_generator.manager.environment_manager import _route_semantics

    cfgs = parse_semantics([{"preset": "pressure_plate", "params": {"position": [1.0, 2.0]}}])
    extras = _route_semantics(cfgs, "elevator")
    assert sorted(extras) == ["pressure_plate"]
    assert {c.name for c in extras["pressure_plate"]} == {"pressed"}


def test_plate_position_and_driven_door_are_realized_and_other_kinds_stay() -> None:
    from arena_simulation_setup.shared.semantics import parse_semantics

    from task_generator.manager.environment_manager import _attachments, _route_semantics
    from task_generator.manager.realizer import Realizer

    realizer = Realizer(Realizer._Configuration(x=5.0, y=-3.0, prefix="env_2"))
    cfgs = parse_semantics(
        [
            {"preset": "gate", "params": {"authorized": [], "unlock_on": "alarm"}},
            {"preset": "pressure_plate", "params": {"position": [1.0, 2.0], "radius": 0.5, "drives": "lab_door"}},
        ],
    )
    attached = dict(_attachments(realizer, _route_semantics(cfgs, "door"), ""))
    assert [c.params["position"] for c in attached["pressure_plate"]] == [[6.0, -1.0]]
    assert [c.params["drives"] for c in attached["pressure_plate"]] == [realizer.prefix("lab_door", "")]
    assert [c.params["radius"] for c in attached["pressure_plate"]] == [0.5]
    assert attached["gate"] == _route_semantics(cfgs, "door")["gate"]


def test_route_semantics_base_vocabulary_field_on_door_raises() -> None:
    """Door state publishes intrinsically now: annotating a door-vocabulary field is stale."""
    from arena_simulation_setup.shared.semantics import parse_semantics

    from task_generator.manager.environment_manager import _route_semantics

    cfgs = parse_semantics([{"state": "progress"}])
    with pytest.raises(ValueError, match="every door publishes 'progress' intrinsically"):
        _route_semantics(cfgs, "door")


def test_route_semantics_unknown_field_raises() -> None:
    from arena_simulation_setup.shared.semantics import parse_semantics

    from task_generator.manager.environment_manager import _route_semantics

    cfgs = parse_semantics([{"state": "not_a_real_field"}])
    with pytest.raises(ValueError, match="is not attachable to a door host"):
        _route_semantics(cfgs, "door")


# ---------------------------------------------------------------------------
# sound semantics write path (M2 engine dispatch through the real chain)
# ---------------------------------------------------------------------------


def test_fire_timeline_entry_writes_sound_sounding_predicate() -> None:
    from arena_runtime.sim._semantics import SemanticsManager
    from arena_simulation_setup.shared.semantics import parse_semantics
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    from task_generator.manager.realizer import Realizer
    from task_generator.node import TaskGenerator

    mech = type("Mech", (), {})()
    manager = SemanticsManager(mech)
    manager.set_sim("dummy")
    cfgs = parse_semantics([{"preset": "sound"}])
    assert manager.attach("sound", "env_0/alarm_sound", cfgs)

    simulator = type("Simulator", (), {})()
    simulator.set_semantic_value = manager.set_value

    class _Logger:
        def warning(self, msg: str) -> None:
            raise AssertionError(msg)

    stub = _node_stub()
    stub._semantic_names = {"alarm_sound": "env_0/alarm_sound"}
    stub._realizer = Realizer()
    stub._simulator = simulator
    stub.get_logger = lambda: _Logger()
    stub._resolve_semantic_entity = TaskGenerator._resolve_semantic_entity.__get__(stub)
    stub._apply_semantic_checked = TaskGenerator._apply_semantic_checked.__get__(stub)
    stub._set_semantic = TaskGenerator._set_semantic.__get__(stub)
    stub._norm_token = TaskGenerator._norm_token
    stub._resolve_timeline_value = TaskGenerator._resolve_timeline_value.__get__(stub)
    stub._timeline_state = [{"rng": random.Random("7:0")}]

    entry = TimelineEntry(set=[{"entity": "alarm_sound", "field": "sounding", "value": "true"}], at=0.0)
    TaskGenerator._fire_timeline_entry(stub, 0, entry)

    snap = manager.snapshot()[0]
    assert snap.entity == "env_0/alarm_sound"
    assert snap.predicates["sounding"] is True


def _stimulus_stub(outcomes: dict[str, object]) -> tuple[object, list[tuple[object, str, float]], list[str]]:
    from arena_simulation_setup.shared.judge import Sample

    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.manager.realizer import Realizer
    from task_generator.node import TaskGenerator

    calls: list[tuple[object, str, float]] = []
    warnings: list[str] = []

    class _Human:
        async def notify_stimulus(self, agent: int | str, stimulus: str, intensity: float) -> bool:
            calls.append((agent, stimulus, intensity))
            outcome = outcomes.get(str(agent), True)
            if isinstance(outcome, Exception):
                raise outcome
            return bool(outcome)

    class _Logger:
        def warning(self, msg: str) -> None:
            warnings.append(msg)

    environment = type("Environment", (), {"notify_stimulus": EnvironmentManager.notify_stimulus})()
    environment._human_simulator = _Human()
    stub = _node_stub()
    stub._environment_manager = environment
    stub._stimulus_tasks = set()
    stub._stimuli_fired = {}
    stub._semantics_dirty = False
    stub._sample = Sample(t=0.0)
    stub._realizer = Realizer(Realizer._Configuration(x=0.0, y=0.0, prefix="env_3"))
    stub._stringify_float = TaskGenerator._stringify_float
    stub.get_logger = lambda: _Logger()
    stub._submit_stimulus = TaskGenerator._submit_stimulus.__get__(stub)
    stub._timeline_semantic_state = TaskGenerator._timeline_semantic_state.__get__(stub)
    return stub, calls, warnings


def _fire_and_drain(stub: object, entry: object) -> None:
    import asyncio

    from task_generator.node import TaskGenerator

    async def main() -> None:
        TaskGenerator._fire_timeline_entry(stub, 0, entry)
        while stub._stimulus_tasks:
            await asyncio.gather(*stub._stimulus_tasks, return_exceptions=True)
            await asyncio.sleep(0)

    asyncio.run(main())


def test_fire_timeline_entry_sends_one_stimulus_per_agent() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"at": 8.0, "stimulus": [{"agents": ["d_1", "d_2"], "need": "form"}, {"agents": ["edge_0"], "need": "go", "intensity": 0.5}]})
    stub, calls, warnings = _stimulus_stub({})
    _fire_and_drain(stub, entry)
    assert calls == [("d_1", "form", 1.0), ("d_2", "form", 1.0), ("edge_0", "go", 0.5)]
    assert warnings == []


def test_refused_or_failing_stimulus_is_logged_not_raised() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"at": 0.0, "stimulus": [{"agents": ["ghost", "broken", "d_1"], "need": "go"}]})
    stub, calls, warnings = _stimulus_stub({"ghost": False, "broken": RuntimeError("service down")})
    _fire_and_drain(stub, entry)
    assert [agent for agent, _, _ in calls] == ["ghost", "broken", "d_1"]
    assert sorted(warnings) == sorted(
        [
            "timeline stimulus 'go' for 'ghost' was not taken",
            "timeline stimulus 'go' for 'broken' failed: RuntimeError('service down')",
        ],
    )
    assert stub._stimulus_tasks == set()
    assert stub._stimuli_fired == {"stimulus.d_1.go": 0.0}


def _reset_stub(stub: object) -> None:
    from task_generator.node import TaskGenerator

    stub._timeline = []
    stub._timeline_state = []
    stub._timeline_t0 = None
    stub._episode_conditions = []
    stub._zone_overrides = {}
    stub._sound_levels = {}
    stub.reset_timeline = TaskGenerator.reset_timeline.__get__(stub)


def test_reset_cancels_a_stimulus_still_in_flight() -> None:
    import asyncio

    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    from task_generator.manager.environment_manager import EnvironmentManager
    from task_generator.node import TaskGenerator

    stub, _calls, warnings = _stimulus_stub({})
    reached: list[str] = []

    class _Hanging:
        async def notify_stimulus(self, agent: int | str, stimulus: str, intensity: float) -> bool:
            reached.append(str(agent))
            await asyncio.Event().wait()
            return True

    environment = type("Environment", (), {"notify_stimulus": EnvironmentManager.notify_stimulus})()
    environment._human_simulator = _Hanging()
    stub._environment_manager = environment
    _reset_stub(stub)

    async def main() -> list[asyncio.Task]:
        TaskGenerator._fire_timeline_entry(stub, 0, TimelineEntry.parse({"at": 0.0, "stimulus": [{"agents": ["edge_0"], "need": "go"}]}))
        await asyncio.sleep(0)
        tasks = list(stub._stimulus_tasks)
        stub.reset_timeline()
        await asyncio.gather(*tasks, return_exceptions=True)
        return tasks

    (task,) = asyncio.run(main())
    assert reached == ["edge_0"]
    assert task.cancelled()
    assert stub._stimulus_tasks == set()
    assert stub._stimuli_fired == {}
    assert warnings == []


def test_a_stimulus_answered_before_a_reset_is_not_recorded_after_it() -> None:
    import asyncio

    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    from task_generator.node import TaskGenerator

    stub, calls, warnings = _stimulus_stub({})
    _reset_stub(stub)

    async def main() -> bool:
        TaskGenerator._fire_timeline_entry(stub, 0, TimelineEntry.parse({"at": 0.0, "stimulus": [{"agents": ["edge_0"], "need": "go"}]}))
        (task,) = stub._stimulus_tasks
        await asyncio.sleep(0)
        answered = task.done()
        stub.reset_timeline()
        await asyncio.sleep(0)
        return answered

    assert asyncio.run(main()) is True
    assert calls == [("edge_0", "go", 1.0)]
    assert stub._stimuli_fired == {}
    assert warnings == []


def _timeline_fields(stub: object) -> dict[str, str]:
    state = stub._timeline_semantic_state()
    return dict(zip(state.discrete_names, state.discrete_values, strict=True))


def test_accepted_stimulus_records_the_sim_time_of_its_firing_tick() -> None:
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"at": 0.0, "stimulus": [{"agents": ["edge_0", "ghost"], "need": "go"}]})
    stub, _calls, _warnings = _stimulus_stub({"ghost": False})
    stub._sample = Sample(t=12.5)
    _fire_and_drain(stub, entry)
    assert stub._semantics_dirty is True
    state = stub._timeline_semantic_state()
    assert (state.entity, state.kind) == ("env_3/timeline", "timeline")
    assert _timeline_fields(stub) == {"stimulus.edge_0.go": "12.5"}


def test_a_stimulus_fired_again_keeps_its_first_firing_time() -> None:
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    go = TimelineEntry.parse({"at": 0.0, "stimulus": [{"agents": ["edge_0"], "need": "go"}]})
    form = TimelineEntry.parse({"at": 0.0, "stimulus": [{"agents": ["edge_0"], "need": "form"}]})
    stub, _calls, _warnings = _stimulus_stub({})
    stub._sample = Sample(t=3.0)
    _fire_and_drain(stub, go)
    stub._sample = Sample(t=9.0)
    _fire_and_drain(stub, go)
    _fire_and_drain(stub, form)
    assert _timeline_fields(stub) == {"stimulus.edge_0.go": "3.0", "stimulus.edge_0.form": "9.0"}


def test_semantic_snapshot_carries_the_timeline_entity() -> None:
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry
    from builtin_interfaces.msg import Time

    from task_generator.node import TaskGenerator

    stub, _calls, _warnings = _stimulus_stub({})
    published: list[object] = []
    stub.sim_time = type("SimTime", (), {"to_msg": lambda self: Time(sec=4)})()
    stub._env_id = 3
    stub._world_manager = type("World", (), {"loaded_world": "map_empty", "publish_world_markers": lambda self, snapshots: None})()
    stub._simulator = type("Simulator", (), {"semantics_snapshot": lambda self: []})()
    stub._zone_semantic_states = list
    stub._robot_semantic_states = list
    stub._pub_state_semantics = type("Publisher", (), {"publish": lambda self, msg: published.append(msg)})()
    TaskGenerator._publish_semantics_snapshot(stub)
    assert [(e.entity, e.kind, list(e.discrete_names)) for e in published[-1].entities] == [("env_3/timeline", "timeline", [])]
    stub._sample = Sample(t=4.0)
    _fire_and_drain(stub, TimelineEntry.parse({"at": 0.0, "stimulus": [{"agents": ["d_1"], "need": "form"}]}))
    TaskGenerator._publish_semantics_snapshot(stub)
    assert [(e.entity, list(e.discrete_names), list(e.discrete_values)) for e in published[-1].entities] == [("env_3/timeline", ["stimulus.d_1.form"], ["4.0"])]
    assert stub._semantics_dirty is False


# ---------------------------------------------------------------------------
# value coercion / stringification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "value", "coerced"),
    [
        ("predicate", "true", True),
        ("predicate", "False", False),
        ("continuous", "1.5", 1.5),
        ("discrete", "lobby", "lobby"),
    ],
)
def test_coerce_zone_value(kind: str, value: str, coerced: object) -> None:
    from task_generator.node import TaskGenerator

    assert TaskGenerator._coerce_zone_value(kind, value) == coerced


@pytest.mark.parametrize(("kind", "value"), [("predicate", "maybe"), ("continuous", "fast")])
def test_coerce_zone_value_malformed(kind: str, value: str) -> None:
    from task_generator.node import TaskGenerator

    with pytest.raises(ValueError, match="malformed value"):
        TaskGenerator._coerce_zone_value(kind, value)


def test_norm_token_booleans_render_lowercase() -> None:
    from task_generator.node import TaskGenerator

    assert TaskGenerator._norm_token(True) == "true"
    assert TaskGenerator._norm_token(False) == "false"
    assert TaskGenerator._norm_token(1.5) == "1.5"


# ---------------------------------------------------------------------------
# timeline value resolution (seeded determinism)
# ---------------------------------------------------------------------------


def _rng_stub(seed: str) -> object:
    from task_generator.node import TaskGenerator

    stub = _node_stub()
    stub._timeline_state = [{"rng": random.Random(seed)}]
    return stub, TaskGenerator


@pytest.mark.parametrize("value", ["true", "a..b"])
def test_resolve_timeline_value_verbatim(value: str) -> None:
    stub, cls = _rng_stub("5:0")
    assert cls._resolve_timeline_value(stub, value, 0) == value


def test_resolve_timeline_value_range_is_seeded() -> None:
    stub_a, cls = _rng_stub("5:0")
    stub_b, _ = _rng_stub("5:0")
    a = cls._resolve_timeline_value(stub_a, "1.0..2.0", 0)
    b = cls._resolve_timeline_value(stub_b, "1.0..2.0", 0)
    assert a == b
    assert 1.0 <= float(a) <= 2.0


def test_resolve_timeline_value_range_differs_by_seed() -> None:
    stub_a, cls = _rng_stub("5:0")
    stub_b, _ = _rng_stub("5:1")
    assert cls._resolve_timeline_value(stub_a, "0.0..1000.0", 0) != cls._resolve_timeline_value(stub_b, "0.0..1000.0", 0)


# ---------------------------------------------------------------------------
# timeline evaluation (at / every / when)
# ---------------------------------------------------------------------------


def _node_stub() -> object:
    """A bare TaskGenerator stand-in: no robots manager, so the record carries no phases."""
    return type("Stub", (), {"_robots_manager": None, "_phases_json": lambda self: "{}"})()


def _timeline_stub(entries: list, seed: int = 7) -> tuple[object, object, list[int]]:
    from task_generator.node import TaskGenerator

    fired: list[int] = []
    stub = _node_stub()
    stub._zone_overrides = {}
    stub._sound_levels = {}
    stub._stimulus_tasks = set()
    stub._stimuli_fired = {}
    stub._fire_timeline_entry = lambda idx, entry: fired.append(idx)
    TaskGenerator.register_timeline(stub, entries, seed)
    return stub, TaskGenerator, fired


def test_timeline_at_fires_once() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "active", "value": "true"}], at=12.0)
    stub, cls, fired = _timeline_stub([entry])
    cls._evaluate_timeline(stub, 100.0)  # captures t0 = 100, rel = 0
    assert fired == []
    cls._evaluate_timeline(stub, 111.0)  # rel = 11 < 12
    assert fired == []
    cls._evaluate_timeline(stub, 112.5)  # rel = 12.5 >= 12
    assert fired == [0]
    cls._evaluate_timeline(stub, 200.0)  # already fired, no re-fire
    assert fired == [0]


def test_timeline_every_fires_each_period() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], every=5.0)
    stub, cls, fired = _timeline_stub([entry])
    cls._evaluate_timeline(stub, 0.0)  # t0 = 0
    cls._evaluate_timeline(stub, 4.0)
    assert fired == []
    cls._evaluate_timeline(stub, 5.0)
    assert fired == [0]
    cls._evaluate_timeline(stub, 10.0)
    assert fired == [0, 0]


def test_timeline_every_first_fires_at_offset() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], every=6.0, offset=2.0)
    stub, cls, fired = _timeline_stub([entry])
    cls._evaluate_timeline(stub, 0.0)
    cls._evaluate_timeline(stub, 1.9)
    assert fired == []
    cls._evaluate_timeline(stub, 2.0)
    assert fired == [0]
    cls._evaluate_timeline(stub, 7.9)
    assert fired == [0]
    cls._evaluate_timeline(stub, 8.0)
    assert fired == [0, 0]


def test_timeline_every_respects_until() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], every=5.0, until=6.0)
    stub, cls, fired = _timeline_stub([entry])
    cls._evaluate_timeline(stub, 0.0)
    cls._evaluate_timeline(stub, 5.0)
    cls._evaluate_timeline(stub, 20.0)  # past until, boundary 10 > 6 skipped
    assert fired == [0]


def test_timeline_when_fires_on_rising_edge() -> None:
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], when={"entity": "s", "field": "level", "is": 1})
    stub, cls, fired = _robot_when_stub([entry], {})
    value = {"level": "0.0"}
    stub._sample = Sample(t=0.0, field=lambda entity, field: value[field] if entity == "s" else None)
    cls._evaluate_timeline(stub, 0.0)
    assert fired == []
    value["level"] = "1.0"
    cls._evaluate_timeline(stub, 1.0)
    assert fired == [0]
    cls._evaluate_timeline(stub, 2.0)
    assert fired == [0]


def test_timeline_when_takes_a_structured_family_node_with_a_hold() -> None:
    from arena_simulation_setup.shared.judge import Sample
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"when": {"some": "fridge.openness", "ge": 0.9, "for": 2}, "set": [{"entity": "a", "field": "x", "value": "1"}]})
    stub, cls, fired = _robot_when_stub([entry], {})
    members = {"left": "0.0", "right": "0.0"}

    def sample(t: float) -> Sample:
        return Sample(t=t, family=lambda entity, family: dict(members) if (entity, family) == ("fridge", "openness") else None)

    for t, right, expected in ((0.0, "0.0", []), (1.0, "0.95", []), (2.5, "0.95", []), (3.0, "0.95", [0]), (4.0, "0.95", [0])):
        members["right"] = right
        stub._sample = sample(t)
        cls._evaluate_timeline(stub, t)
        assert fired == expected, t


def _robot_when_stub(entries: list, robots: dict[str, tuple[float, float, float]]) -> tuple[object, object, list[int]]:
    from arena_simulation_setup.shared.judge import Sample

    from task_generator.node import TaskGenerator

    stub, cls, fired = _timeline_stub(entries)
    stub._judge_zones = {}
    stub._sample = Sample(t=0.0, robots=robots, known=frozenset(robots))
    stub._norm_token = TaskGenerator._norm_token
    stub._when_true = TaskGenerator._when_true.__get__(stub)
    return stub, cls, fired


def _place(stub: object, robots: dict[str, tuple[float, float, float]]) -> None:
    from arena_simulation_setup.shared.judge import Sample

    stub._sample = Sample(t=0.0, robots=robots, known=frozenset(robots))


def test_timeline_robot_within_a_point_fires_for_a_robot_named_after_its_model() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"when": "robot within 2.0 of 5.0,0.0", "stimulus": [{"agents": ["edge_0"], "need": "go"}]})
    stub, cls, fired = _robot_when_stub([entry], {"jackal": (0.0, 0.0, 0.0)})
    cls._evaluate_timeline(stub, 0.0)
    assert fired == []
    _place(stub, {"jackal": (3.5, 0.0, 0.0)})
    cls._evaluate_timeline(stub, 1.0)
    assert fired == [0]
    cls._evaluate_timeline(stub, 2.0)
    assert fired == [0]


def test_timeline_robot_within_fires_when_any_robot_of_the_env_holds() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"when": "robot within 1.0 of 0.0,0.0", "set": [{"entity": "a", "field": "x", "value": "1"}]})
    stub, cls, fired = _robot_when_stub([entry], {"jackal": (5.0, 0.0, 0.0), "burger": (4.0, 0.0, 0.0)})
    cls._evaluate_timeline(stub, 0.0)
    assert fired == []
    _place(stub, {"jackal": (5.0, 0.0, 0.0), "burger": (0.5, 0.0, 0.0)})
    cls._evaluate_timeline(stub, 1.0)
    assert fired == [0]


def test_timeline_not_robot_within_fires_once_the_robot_leaves() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"when": "not robot within 3.0 of 0.0,0.0", "stimulus": [{"agents": ["d_1"], "need": "break"}]})
    stub, cls, fired = _robot_when_stub([entry], {"jackal": (1.0, 0.0, 0.0)})
    cls._evaluate_timeline(stub, 0.0)
    assert fired == []
    _place(stub, {"jackal": (4.0, 0.0, 0.0)})
    cls._evaluate_timeline(stub, 1.0)
    assert fired == [0]


def test_timeline_robot_atom_without_a_placed_robot_does_not_fire() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry.parse({"when": "not robot within 3.0 of 0.0,0.0", "set": [{"entity": "a", "field": "x", "value": "1"}]})
    stub, cls, fired = _robot_when_stub([entry], {})
    cls._evaluate_timeline(stub, 0.0)
    cls._evaluate_timeline(stub, 1.0)
    assert fired == []


def test_reset_timeline_clears_state() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    from task_generator.node import TaskGenerator

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], at=1.0)
    stub, cls, _fired = _timeline_stub([entry])
    TaskGenerator.register_sound_levels(stub, {"env_0/alarm_sound": 88.0})
    stub._stimuli_fired = {"stimulus.edge_0.go": 3.0}
    TaskGenerator.reset_timeline(stub)
    assert stub._stimuli_fired == {}
    assert stub._timeline == []
    assert stub._timeline_state == []
    assert stub._timeline_t0 is None
    assert stub._sound_levels == {}


# ---------------------------------------------------------------------------
# M3 episode conditions carry
# ---------------------------------------------------------------------------


def test_register_conditions_sets_episode_conditions() -> None:
    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from task_generator.node import TaskGenerator

    stub = _node_stub()
    cond = EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a"})
    TaskGenerator.register_conditions(stub, [cond])
    assert stub._episode_conditions == [cond]


def test_reset_timeline_clears_episode_conditions() -> None:
    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], at=1.0)
    stub, cls, _fired = _timeline_stub([entry])
    cls.register_conditions(stub, [EpisodeCondition.parse({"op": "never", "p": "robot in pharmacy"})])
    cls.reset_timeline(stub)
    assert stub._episode_conditions == []


def test_record_to_msg_serializes_episode_conditions() -> None:
    import json

    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from task_generator.node import EpisodeRecord, TaskGenerator

    stub = _node_stub()
    stub._episode_conditions = [
        EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a", "text": "Deliver the package."}),
    ]
    msg = TaskGenerator._record_to_msg(stub, EpisodeRecord())
    assert json.loads(msg.conditions) == [{"op": "eventually", "p": "robot in ward_a", "text": "Deliver the package."}]


def test_record_to_msg_empty_conditions_serializes_empty_list() -> None:
    import json

    from task_generator.node import EpisodeRecord, TaskGenerator

    stub = _node_stub()
    stub._episode_conditions = []
    msg = TaskGenerator._record_to_msg(stub, EpisodeRecord())
    assert json.loads(msg.conditions) == []
