"""Gate ledger and registry semantics, including replays of the defects found live on gz."""

from __future__ import annotations

import pytest

from arena_runtime.lockstep_gates import (
    ChannelRegistry,
    ChannelSpec,
    GateLedger,
    env_match,
    parse_channel_entry,
    step_slice,
)

DT = 1.0 / 60.0
HEARTBEAT = "arena_runtime_msgs/msg/LockstepHeartbeat"


def _spec(name: str, period: float, *, hard: bool = True, topic: str | None = None) -> ChannelSpec:
    return ChannelSpec(name=name, topic_template=topic or f"/{name}", type_name=HEARTBEAT, period_s=period, hard=hard)


def _desired(*specs: ChannelSpec) -> dict[str, tuple[ChannelSpec, str]]:
    return {s.topic_template: (s, s.name) for s in specs}


def _tick(ledger: GateLedger) -> list:
    """One scheduler iteration up to the gate: step to the earliest due channel and return the hard due set."""
    ledger.advance(ledger.next_delta())
    return [ch for ch in ledger.due() if ch.hard]


# spec parsing


def test_parse_entry_roles() -> None:
    hard = parse_channel_entry("engine|/arena/{env}/agent_states|std_msgs/msg/Header|0.05|hard")
    soft = parse_channel_entry("peds|/arena/{env}/arena_peds|std_msgs/msg/Header|0.05|soft")
    assert hard.hard and not soft.hard
    assert hard.topic_template == "/arena/{env}/agent_states"
    assert hard.period_s == pytest.approx(0.05)


@pytest.mark.parametrize(
    "entry",
    [
        "too|few|fields",
        "x|/t|std_msgs/msg/Header|0.1|medium",
        "x|/t|std_msgs/msg/Header|fast|hard",
        "x|/t|std_msgs/msg/Header|0|hard",
        "x|/t|std_msgs/msg/Header|-1|soft",
    ],
)
def test_parse_entry_rejects(entry: str) -> None:
    with pytest.raises(ValueError):
        parse_channel_entry(entry)


def test_spec_rejects_nonpositive_period() -> None:
    with pytest.raises(ValueError):
        ChannelSpec(name="x", topic_template="/x", type_name=HEARTBEAT, period_s=0.0, hard=True)


def test_step_slice() -> None:
    assert step_slice(0.0, DT) == 1.0
    assert step_slice(1.0, DT) == pytest.approx(0.05)
    assert step_slice(0.1, DT) == pytest.approx(DT)
    assert step_slice(100.0, DT) == 1.0


def test_env_match_is_slash_prefix_containment() -> None:
    assert env_match("/arena/env_0", "/arena/env_0/task_generator_node")
    assert env_match("/arena/env_0/task_generator_node", "/arena/env_0")
    assert env_match("arena/env_0", "/arena/env_0/")
    assert not env_match("/arena/env_1", "/arena/env_10")
    # a basename never matches the fqn the scheduler drops by (live seam bug, 2026-08-11)
    assert not env_match("env_0", "/arena/env_0/task_generator_node")


# registry


def test_registry_replace_and_clear() -> None:
    reg = ChannelRegistry()
    reg.register("a", "/arena/env_0", [_spec("x", 0.1)])
    reg.register("a", "/arena/env_0", [_spec("y", 0.1)])
    assert list(reg.desired([])) == ["/y"]
    reg.register("a", "/arena/env_0", [])
    assert reg.desired([]) == {}
    assert reg.version == 3


def test_registry_rejects_empty_caller() -> None:
    reg = ChannelRegistry()
    with pytest.raises(ValueError):
        reg.register("", "", [_spec("x", 0.1)])
    assert reg.version == 0


def test_registry_global_expands_per_env_and_concrete_is_verbatim() -> None:
    reg = ChannelRegistry()
    reg.register("launch", "", [_spec("engine", 0.05, topic="/arena/{env}/agent_states")])
    reg.register("/arena/env_1/task_generator_node", "/arena/env_1", [_spec("roster", 0.15, topic="/arena/env_1/roster")])
    desired = reg.desired([0, 2])
    assert set(desired) == {"/arena/env_0/agent_states", "/arena/env_2/agent_states", "/arena/env_1/roster"}
    assert desired["/arena/env_0/agent_states"][1] == "engine/env_0"
    assert desired["/arena/env_1/roster"][1] == "roster"


def test_registry_first_registrant_wins_per_topic() -> None:
    reg = ChannelRegistry()
    first = _spec("cli_peds", 0.15, topic="/arena/env_0/arena_peds")
    second = _spec("peds", 0.0999, hard=False, topic="/arena/env_0/arena_peds")
    reg.register("cli", "", [first])
    reg.register("/arena/env_0/task_generator_node", "/arena/env_0", [second])
    assert reg.desired([0])["/arena/env_0/arena_peds"] == (first, "cli_peds")


def test_registry_drop_env_by_fqn_containment() -> None:
    reg = ChannelRegistry()
    reg.register("/arena/env_0/task_generator_node", "/arena/env_0", [_spec("engine", 0.05)])
    reg.register("/arena/env_0/jackal/lockstep/nav", "/arena/env_0/task_generator_node", [_spec("nav", 0.05)])
    reg.register("/arena/env_1/task_generator_node", "/arena/env_1", [_spec("other", 0.05)])
    reg.register("launch", "", [_spec("global", 0.05)])
    version = reg.version
    assert set(reg.drop_env("/arena/env_0/task_generator_node")) == {
        "/arena/env_0/task_generator_node",
        "/arena/env_0/jackal/lockstep/nav",
    }
    assert reg.version == version + 1
    assert set(reg.desired([1])) == {"/other", "/global"}
    assert reg.drop_env("/arena/env_7") == []
    assert reg.version == version + 1


# ledger


def test_refresh_adds_removes_and_keeps_state() -> None:
    ledger = GateLedger(dt=DT, base=10.0)
    a = _spec("a", 0.1)
    fast = _spec("fast", DT / 4)
    removed, added = ledger.refresh(_desired(a, fast))
    assert removed == [] and {ch.name for ch in added} == {"a", "fast"}
    assert ledger.channels["/fast"].period == pytest.approx(DT)
    assert ledger.channels["/a"].next_due == pytest.approx(0.1)

    ledger.observe(ledger.channels["/a"], 10.3)
    renamed = ChannelSpec(name="a2", topic_template="/a", type_name=HEARTBEAT, period_s=0.1, hard=True)
    removed, added = ledger.refresh({"/a": (renamed, "a2")})
    assert {ch.name for ch in removed} == {"a", "fast"}
    assert [ch.name for ch in added] == ["a2"]
    assert ledger.channels["/a"].latest_stamp is None

    removed, added = ledger.refresh({"/a": (renamed, "a2")})
    assert removed == [] and added == []


def test_next_delta_floors_to_one_step() -> None:
    ledger = GateLedger(dt=DT, base=0.0)
    ledger.refresh(_desired(_spec("a", 0.1)))
    assert ledger.next_delta() == pytest.approx(0.1)
    ledger.advance(0.1)
    assert ledger.next_delta() == pytest.approx(DT)


def test_well_behaved_producer_never_waits() -> None:
    ledger = GateLedger(dt=DT, base=100.0)
    ledger.refresh(_desired(_spec("beat", 0.1)))
    (beat,) = ledger.channels.values()
    for k in range(1, 50):
        hard_due = _tick(ledger)
        assert hard_due == [beat]
        assert ledger.waiting(hard_due) == [beat]
        ledger.observe(beat, 100.0 + k * 0.1)
        assert ledger.waiting(hard_due) == []
        ledger.complete(hard_due)
    assert ledger.now == pytest.approx(4.9)


def test_coverage_predicate_is_clock_relative() -> None:
    """A stamp covers the tick the ledger stands on when it is less than one period old.
    The window grid picks step sizes only and must not enter the predicate."""
    ledger = GateLedger(dt=DT, base=100.0)
    ledger.refresh(_desired(_spec("engine", 0.1)))
    (engine,) = ledger.channels.values()
    assert not ledger.covers(engine)
    ledger.advance(0.1)
    ledger.observe(engine, 100.0)
    assert not ledger.covers(engine)
    ledger.observe(engine, 100.001)
    assert ledger.covers(engine)
    ledger.observe(engine, 100.07)
    assert ledger.covers(engine)
    # the same stamp against any window boundary, and stale once the clock moves on
    engine.next_due = 1234.0
    assert ledger.covers(engine)
    engine.next_due = 0.0
    assert ledger.covers(engine)
    ledger.advance(0.1)
    assert not ledger.covers(engine)


def _elapsed_producer(gate_period: float, producer_interval: float, ticks: int) -> int:
    """Producer that publishes stamp=clock whenever >= interval of sim time elapsed since
    its last publish, driven by a scheduler that only advances the clock to window ends.
    Returns how many windows cleared before the first stall."""
    ledger = GateLedger(dt=DT, base=0.0)
    ledger.refresh(_desired(_spec("roster", gate_period)))
    (roster,) = ledger.channels.values()
    last_pub = 0.0
    cleared = 0
    for _ in range(ticks):
        hard_due = _tick(ledger)
        if ledger.now - last_pub >= producer_interval:
            last_pub = ledger.now
            ledger.observe(roster, ledger.now)
        if ledger.waiting(hard_due):
            return cleared
        ledger.complete(hard_due)
        cleared += 1
    return cleared


def test_gate_period_must_exceed_elapsed_based_producer_interval() -> None:
    """hunav publishes on elapsed sim time (0.1 s), so a 0.0999 gate freezes the clock
    just short of its next publish. The CLI preset and the adapter register 0.15."""
    assert _elapsed_producer(gate_period=0.15, producer_interval=0.1, ticks=200) == 200
    assert _elapsed_producer(gate_period=0.0999, producer_interval=0.1, ticks=200) < 200


ENGINE_PERIOD = 0.05
ENGINE_BASE = 100.0
NS = 1_000_000_000


def _engine_replay(physics_dt: float, phase: float, windows: int) -> tuple[float, int]:
    """Scheduler and clock-driven engine in lockstep: the scheduler steps in whole
    physics steps (gz rounds the request), the engine ticks up to the grid point the
    clock has covered since its epoch and stamps that tick's start. Returns the sim
    time reached and the engine's tick count."""
    period_ns = int(ENGINE_PERIOD * NS)
    epoch_ns = int(ENGINE_BASE * NS) - int(phase * NS)
    ledger = GateLedger(dt=physics_dt, base=ENGINE_BASE)
    ledger.refresh(_desired(_spec("engine", ENGINE_PERIOD)))
    (engine,) = ledger.channels.values()
    ticks = 0
    for _ in range(windows):
        before = ledger.now
        ledger.advance(max(1, round(ledger.next_delta() / physics_dt)) * physics_dt)
        assert ledger.now > before, "the scheduler must always advance the clock"
        hard_due = [ch for ch in ledger.due() if ch.hard]
        covered = (round(ledger.tick * NS) - epoch_ns) // period_ns + 1
        assert covered >= ticks, "the clock never owes fewer ticks than the engine ran"
        if covered > ticks:
            ticks = covered
            ledger.observe(engine, (epoch_ns + (ticks - 1) * period_ns) / NS)
        assert ledger.waiting(hard_due) == [], f"gate wedged at {ledger.now:.4f} with {ticks} ticks"
        ledger.complete(hard_due)
    return ledger.now, ticks


@pytest.mark.parametrize("physics_dt", [0.0333, 0.01, 0.004, 0.001])
@pytest.mark.parametrize("phase", [i * ENGINE_PERIOD / 8 for i in range(8)])
def test_engine_gate_never_wedges_across_phases_and_step_sizes(physics_dt: float, phase: float) -> None:
    """The engine epoch lands anywhere in its period relative to the run, and a step
    can advance less than that period. Neither may freeze the clock (live 2026-09-03)."""
    windows = 60
    now, ticks = _engine_replay(physics_dt, phase, windows)
    assert now >= windows * physics_dt
    assert ticks == pytest.approx(now / ENGINE_PERIOD, abs=2)


def test_gate_holds_while_the_engine_lags() -> None:
    """Coverage is honest under lag: a stamp a full period stale never clears the tick."""
    ledger = GateLedger(dt=DT, base=0.0)
    ledger.refresh(_desired(_spec("engine", ENGINE_PERIOD)))
    (engine,) = ledger.channels.values()
    for k in range(1, 20):
        hard_due = _tick(ledger)
        assert ledger.now == pytest.approx(k * ENGINE_PERIOD)
        ledger.observe(engine, (k - 1) * ENGINE_PERIOD)
        assert ledger.waiting(hard_due) == [engine]
        ledger.observe(engine, k * ENGINE_PERIOD)
        assert ledger.waiting(hard_due) == []
        ledger.complete(hard_due)


def test_rebase_after_external_advance() -> None:
    ledger = GateLedger(dt=DT, base=5.0)
    ledger.refresh(_desired(_spec("a", 0.1), _spec("b", 0.25)))
    ledger.complete(_tick(ledger))
    assert ledger.now == pytest.approx(0.1)
    # jitter within half a step and a clock that lags are not drift
    assert ledger.rebase(5.1 + DT / 4) == 0.0
    assert ledger.rebase(5.05) == 0.0
    assert ledger.now == pytest.approx(0.1)
    # an unpause window ran the sim 2 s: every window restarts from the new now
    drift = ledger.rebase(7.1)
    assert drift == pytest.approx(2.0)
    assert ledger.now == pytest.approx(2.1)
    assert ledger.tick == pytest.approx(7.1)
    assert ledger.channels["/a"].next_due == pytest.approx(2.2)
    assert ledger.channels["/b"].next_due == pytest.approx(2.35)


def test_soft_channel_schedules_but_never_gates() -> None:
    ledger = GateLedger(dt=DT, base=0.0)
    ledger.refresh(_desired(_spec("engine", 0.1), _spec("peds", 0.04, hard=False)))
    ledger.advance(ledger.next_delta())
    assert ledger.now == pytest.approx(0.04)
    due = ledger.due()
    assert [ch.name for ch in due] == ["peds"]
    assert [ch for ch in due if ch.hard] == []


def test_due_survives_float_accumulation() -> None:
    ledger = GateLedger(dt=DT, base=1234.5)
    ledger.refresh(_desired(_spec("a", 0.1), _spec("b", 0.15), _spec("c", 0.0999)))
    expected = {"/a": 0, "/b": 0, "/c": 0}
    for _ in range(5000):
        ledger.advance(ledger.next_delta())
        due = ledger.due()
        assert due, "a step landed on no window end"
        for ch in due:
            ledger.observe(ch, ledger.tick)
            expected[ch.topic] += 1
        assert ledger.waiting(due) == []
        ledger.complete(due)
    for ch in ledger.channels.values():
        assert expected[ch.topic] == pytest.approx(ledger.now / ch.period, abs=1)
