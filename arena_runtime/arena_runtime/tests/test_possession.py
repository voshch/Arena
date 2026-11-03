from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from arena_runtime.humans.possession import (
    PossessionTable,
    bare_joint_names_valid,
    snapshot_roster,
    validate_stream,
)


def _ped(
    name: str,
    joint_names: Sequence[str] = (),
    joint_positions: Sequence[float] = (),
    model_uri: str = "",
    ped_id: int = 0,
    gait_phase: float = 0.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        id=ped_id,
        gait_phase=gait_phase,
        joint_state=SimpleNamespace(name=list(joint_names), position=list(joint_positions)),
        model_uri=model_uri,
    )


# stream validation


def test_unknown_name_dropped() -> None:
    ped = _ped("ghost")
    validated, unknown, bad_joints = validate_stream([ped], gate_open=True, known_names={"alice"})
    assert validated == []
    assert unknown == {"ghost"}
    assert bad_joints == frozenset()


def test_bare_name_mismatch_dropped() -> None:
    # hri_producer suffixes joint names for the wire, so a streamed entry using
    # the suffixed form (instead of bare GaitGenerator.JOINT_NAMES) must drop.
    ped = _ped("alice", joint_names=["l_elbow_body"], joint_positions=[0.5])
    validated, unknown, bad_joints = validate_stream([ped], gate_open=True, known_names={"alice"})
    assert validated == []
    assert bad_joints == {"alice"}
    assert unknown == frozenset()


def test_out_of_limit_angles_pass_through() -> None:
    ped = _ped("alice", joint_names=["l_elbow", "waist"], joint_positions=[99.0, -99.0])
    validated, _, _ = validate_stream([ped], gate_open=True, known_names={"alice"})
    assert len(validated) == 1
    positions = dict(zip(validated[0].joint_state.name, validated[0].joint_state.position, strict=True))
    assert positions["l_elbow"] == pytest.approx(99.0)
    assert positions["waist"] == pytest.approx(-99.0)


def test_model_uri_stripped() -> None:
    ped = _ped("alice", model_uri="package://sneaky/model.sdf")
    validated, _, _ = validate_stream([ped], gate_open=True, known_names={"alice"})
    assert validated[0].model_uri == ""


def test_gate_closed_drops_everything() -> None:
    ped = _ped("alice", joint_names=["waist"], joint_positions=[0.1])
    validated, unknown, bad_joints = validate_stream([ped], gate_open=False, known_names={"alice"})
    assert validated == []
    assert unknown == frozenset()
    assert bad_joints == frozenset()


def test_validated_entry_is_independent_copy() -> None:
    ped = _ped("alice", joint_names=["waist"], joint_positions=[0.1])
    validated, _, _ = validate_stream([ped], gate_open=True, known_names={"alice"})
    validated[0].joint_state.position[0] = 999.0
    assert ped.joint_state.position[0] == 0.1


def test_snapshot_roster_copy_semantics() -> None:
    """publish_arena_peds mutates the snapshot, fills must not leak into the cache."""
    cached = _ped("alice")
    cache = {"alice": cached}
    out = snapshot_roster(cache)
    out[0].joint_state.name = ["waist"]
    out[0].joint_state.position = [0.42]
    assert cache["alice"].joint_state.name == []
    assert cache["alice"].joint_state.position == []


def test_bare_joint_names_valid_accepts_known_and_empty() -> None:
    assert bare_joint_names_valid([])
    assert bare_joint_names_valid(["waist", "l_elbow"])
    assert not bare_joint_names_valid(["waist_body"])


# possession table


def test_claim_and_renew() -> None:
    table = PossessionTable()
    accepted, unknown, bad_joints, released = table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=0.0)
    assert [ped.name for ped in accepted] == ["alice"]
    assert unknown == frozenset()
    assert bad_joints == frozenset()
    assert released == []
    assert table.possessed(0.5) == {"alice"}
    table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=0.9)
    assert table.expire(1.8) == []
    assert table.possessed(1.8) == {"alice"}
    expired = table.expire(2.0)
    assert [name for name, _ in expired] == ["alice"]
    assert table.possessed(2.0) == set()


def test_manifest_omission_releases() -> None:
    table = PossessionTable()
    table.merge([_ped("alice"), _ped("bob")], known_names={"alice", "bob"}, gate_open=True, now=0.0)
    _, _, _, released = table.merge([_ped("alice")], known_names={"alice", "bob"}, gate_open=True, now=0.1)
    assert [name for name, _ in released] == ["bob"]
    assert released[0][1].name == "bob"
    assert table.possessed(0.1) == {"alice"}


def test_empty_batch_releases_all() -> None:
    table = PossessionTable()
    table.merge([_ped("alice"), _ped("bob")], known_names={"alice", "bob"}, gate_open=True, now=0.0)
    _, _, _, released = table.merge([], known_names={"alice", "bob"}, gate_open=True, now=0.1)
    assert {name for name, _ in released} == {"alice", "bob"}
    assert table.possessed(0.1) == set()


def test_timeout_expiry() -> None:
    table = PossessionTable()
    table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=5.0)
    assert table.expire(6.0) == []
    ((name, state),) = table.expire(6.01)
    assert name == "alice"
    assert state.name == "alice"
    assert table.expire(7.0) == []


def test_bad_entry_keeps_claim_until_timeout() -> None:
    table = PossessionTable()
    table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=0.0)
    accepted, _, bad_joints, released = table.merge([_ped("alice", joint_names=["bogus"], joint_positions=[0.1])], known_names={"alice"}, gate_open=True, now=0.5)
    assert accepted == []
    assert bad_joints == {"alice"}
    assert released == []
    assert table.possessed(0.5) == {"alice"}
    assert [name for name, _ in table.expire(1.5)] == ["alice"]


def test_substitute_swaps_possessed_only() -> None:
    table = PossessionTable()
    table.merge([_ped("alice", joint_names=["waist"], joint_positions=[0.2], ped_id=7)], known_names={"alice"}, gate_open=True, now=0.0)
    bus_alice = _ped("alice", ped_id=7)
    bus_bob = _ped("bob", ped_id=8)
    batch = [bus_alice, bus_bob]
    out = table.substitute(batch, 0.5)
    assert [ped.id for ped in out] == [7, 8]
    assert out[0] is not bus_alice
    assert out[0].joint_state.name == ["waist"]
    assert out[1] is bus_bob
    assert len(batch) == 2
    assert batch[0] is bus_alice
    assert batch[1] is bus_bob
    assert bus_alice.joint_state.name == []


def test_substitute_preserves_bus_id() -> None:
    table = PossessionTable()
    table.merge([_ped("alice", ped_id=999)], known_names={"alice"}, gate_open=True, now=0.0)
    out = table.substitute([_ped("alice", ped_id=7)], 0.1)
    assert out[0].id == 7


def test_substitute_restamps_gait_phase() -> None:
    table = PossessionTable(phase={3: 2.5}.__getitem__)
    table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=0.0)
    out = table.substitute([_ped("alice", ped_id=3)], 0.1)
    assert out[0].gait_phase == pytest.approx(2.5)


def test_substitute_ignores_expired() -> None:
    table = PossessionTable()
    table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=0.0)
    bus = _ped("alice")
    out = table.substitute([bus], 2.0)
    assert out[0] is bus


def test_deep_copy_isolation_both_directions() -> None:
    table = PossessionTable()
    src = _ped("alice", joint_names=["waist"], joint_positions=[0.1])
    accepted, _, _, _ = table.merge([src], known_names={"alice"}, gate_open=True, now=0.0)
    accepted[0].joint_state.position[0] = 99.0
    assert table.states(0.1)["alice"].joint_state.position[0] == pytest.approx(0.1)
    states = table.states(0.1)
    states["alice"].joint_state.position[0] = 55.0
    assert table.states(0.2)["alice"].joint_state.position[0] == pytest.approx(0.1)
    out = table.substitute([_ped("alice")], 0.3)
    out[0].joint_state.position[0] = 77.0
    assert table.states(0.4)["alice"].joint_state.position[0] == pytest.approx(0.1)


def test_gate_closed_merge_is_inert() -> None:
    table = PossessionTable()
    table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=0.0)
    accepted, unknown, bad_joints, released = table.merge([], known_names={"alice"}, gate_open=False, now=0.1)
    assert accepted == []
    assert unknown == frozenset()
    assert bad_joints == frozenset()
    assert released == []
    assert table.possessed(0.1) == {"alice"}


def test_clear_drops_everything_silently() -> None:
    table = PossessionTable()
    table.merge([_ped("alice")], known_names={"alice"}, gate_open=True, now=0.0)
    assert table.clear() is None
    assert table.possessed(0.0) == set()
    assert table.expire(10.0) == []
