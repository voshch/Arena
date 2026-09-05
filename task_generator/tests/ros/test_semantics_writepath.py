"""M2-B semantics write path and scenario timeline: pure-logic coverage.

Follows the stub pattern used across the node-service tests: unbound methods are
invoked against a plain object carrying only the attributes each method reads.
"""
from __future__ import annotations

import random

import pytest


@pytest.fixture(autouse=True)
def _ros_gate() -> None:
    try:
        import rclpy  # noqa: F401
    except ImportError:
        pytest.skip("ROS2 not available")


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

    stub = type("Stub", (), {})()
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


# ---------------------------------------------------------------------------
# value coercion / stringification
# ---------------------------------------------------------------------------


def test_coerce_zone_value_predicate_true_false() -> None:
    from task_generator.node import TaskGenerator

    assert TaskGenerator._coerce_zone_value("predicate", "true") is True
    assert TaskGenerator._coerce_zone_value("predicate", "False") is False


def test_coerce_zone_value_predicate_malformed() -> None:
    from task_generator.node import TaskGenerator

    with pytest.raises(ValueError, match="malformed value"):
        TaskGenerator._coerce_zone_value("predicate", "maybe")


def test_coerce_zone_value_continuous_and_malformed() -> None:
    from task_generator.node import TaskGenerator

    assert TaskGenerator._coerce_zone_value("continuous", "1.5") == 1.5
    with pytest.raises(ValueError, match="malformed value"):
        TaskGenerator._coerce_zone_value("continuous", "fast")


def test_coerce_zone_value_discrete_passthrough() -> None:
    from task_generator.node import TaskGenerator

    assert TaskGenerator._coerce_zone_value("discrete", "lobby") == "lobby"


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

    stub = type("Stub", (), {})()
    stub._timeline_state = [{"rng": random.Random(seed)}]
    return stub, TaskGenerator


def test_resolve_timeline_value_verbatim() -> None:
    stub, cls = _rng_stub("5:0")
    assert cls._resolve_timeline_value(stub, "true", 0) == "true"


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


def test_resolve_timeline_value_bad_range_verbatim() -> None:
    stub, cls = _rng_stub("5:0")
    assert cls._resolve_timeline_value(stub, "a..b", 0) == "a..b"


# ---------------------------------------------------------------------------
# timeline evaluation (at / every / when)
# ---------------------------------------------------------------------------


def _timeline_stub(entries: list, seed: int = 7) -> tuple[object, object, list[int]]:
    from task_generator.node import TaskGenerator

    fired: list[int] = []
    stub = type("Stub", (), {})()
    stub._zone_overrides = {}
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


def test_timeline_every_respects_until() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], every=5.0, until=6.0)
    stub, cls, fired = _timeline_stub([entry])
    cls._evaluate_timeline(stub, 0.0)
    cls._evaluate_timeline(stub, 5.0)
    cls._evaluate_timeline(stub, 20.0)  # past until, boundary 10 > 6 skipped
    assert fired == [0]


def test_timeline_when_fires_on_rising_edge() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], when={"entity": "s", "field": "active", "is": "true"})
    stub, cls, fired = _timeline_stub([entry])
    edge = {"value": False}
    stub._when_true = lambda when: edge["value"]
    cls._evaluate_timeline(stub, 0.0)  # false
    assert fired == []
    edge["value"] = True
    cls._evaluate_timeline(stub, 1.0)  # false -> true edge
    assert fired == [0]
    cls._evaluate_timeline(stub, 2.0)  # stays true, no re-fire
    assert fired == [0]


def test_reset_timeline_clears_state() -> None:
    from arena_simulation_setup.tree.World.Scenario import TimelineEntry

    from task_generator.node import TaskGenerator

    entry = TimelineEntry(set=[{"entity": "a", "field": "x", "value": "1"}], at=1.0)
    stub, cls, _fired = _timeline_stub([entry])
    TaskGenerator.reset_timeline(stub)
    assert stub._timeline == []
    assert stub._timeline_state == []
    assert stub._timeline_t0 is None


# ---------------------------------------------------------------------------
# M3 episode conditions carry
# ---------------------------------------------------------------------------


def test_register_conditions_sets_episode_conditions() -> None:
    from arena_simulation_setup.shared.conditions import EpisodeCondition
    from task_generator.node import TaskGenerator

    stub = type("Stub", (), {})()
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

    stub = type("Stub", (), {})()
    stub._episode_conditions = [
        EpisodeCondition.parse({"op": "eventually", "p": "robot in ward_a", "text": "Deliver the package."}),
    ]
    msg = TaskGenerator._record_to_msg(stub, EpisodeRecord())
    assert json.loads(msg.conditions) == [{"op": "eventually", "p": "robot in ward_a", "text": "Deliver the package."}]


def test_record_to_msg_empty_conditions_serializes_empty_list() -> None:
    import json

    from task_generator.node import EpisodeRecord, TaskGenerator

    stub = type("Stub", (), {})()
    stub._episode_conditions = []
    msg = TaskGenerator._record_to_msg(stub, EpisodeRecord())
    assert json.loads(msg.conditions) == []
