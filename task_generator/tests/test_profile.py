"""Pose profile library, agent-type section ops, laws and the agent-type yaml resolution."""

from __future__ import annotations

import math
from pathlib import Path

import attrs
import pytest
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from task_generator.simulators.human.gait import LIMITS, GaitGenerator
from task_generator.simulators.human.profile import (
    DEFAULT_WALK,
    JOINT_NAMES,
    CadenceLaw,
    GainLaw,
    PhaseWarp,
    Signal,
    agent_type_pose_section,
    default_profile,
    load_library,
    resolve_pose_profile,
)

_WALKING = 1
_RUNNING = 2
PAIRS = (("l_r_hip", "r_r_hip"), ("l_knee", "r_knee"), ("l_p_shoulder", "r_p_shoulder"), ("l_elbow", "r_elbow"))


def _walk(profile, phi: float = 0.7, speed: float = 1.2) -> dict[str, float]:
    gen = GaitGenerator()
    return gen.compute(0, _WALKING, speed, 0.0, phase=phi, profile=profile)


def test_library_ships_walk_and_run() -> None:
    library = load_library()
    assert {DEFAULT_WALK, "run"} <= set(library)
    walk, run = library[DEFAULT_WALK], library["run"]
    assert run.joints == walk.joints
    assert run.gain.factor == 1.6 and walk.gain.factor == 1.0
    assert default_profile().walk == walk and default_profile().run == run


def test_default_laws_match_the_frozen_gait_formulas() -> None:
    walk = default_profile().walk
    for speed in (0.0, 0.3, 1.0, 1.6, 2.5, 4.0):
        assert walk.cadence(speed) == max(0.4, min(2.2, 0.4 + 0.55 * speed))
        assert walk.gain(speed) == max(0.2, min(1.0, speed / 1.2))
        assert default_profile().run.gain(speed) == 1.6 * max(0.2, min(1.0, speed / 1.2))


def test_cadence_law_fields_override_field_by_field() -> None:
    profile = resolve_pose_profile({"walk": {"cadence": {"base": 0.6, "max": 1.0}}})
    assert profile.walk.cadence == CadenceLaw(base=0.6, per_speed=0.55, lo=0.4, hi=1.0)
    assert profile.walk.cadence(2.0) == 1.0
    assert profile.walk.cadence(0.0) == 0.6
    assert profile.run.cadence == CadenceLaw()


def test_law_keys_are_min_and_max() -> None:
    with pytest.raises(ValueError, match="cadence takes"):
        resolve_pose_profile({"walk": {"cadence": {"lo": 0.1}}})
    assert resolve_pose_profile({"walk": {"gain": {"min": 0.5, "max": 0.9}}}).walk.gain == GainLaw(lo=0.5, hi=0.9)


def test_cadence_drives_the_integrator() -> None:
    slow = resolve_pose_profile({"walk": {"cadence": {"base": 0.1, "per_speed": 0.0, "min": 0.1, "max": 0.1}}})
    gen = GaitGenerator()
    start = gen.phase(3)
    gen.compute(3, _WALKING, 1.0, 0.5, profile=slow)
    assert gen.phase(3) == pytest.approx(start + 2.0 * math.pi * 0.1 * 0.5)


def test_scale_one_and_offset_zero_are_exact_noops() -> None:
    base = _walk(default_profile())
    for ops in ({"knee": {"scale": 1.0}}, {"knee": {"offset": 0}}, {"hip": {"offset": {"mean": 0.0}}}, {"elbow": {"scale": 1.0, "offset": 0.0}}):
        assert _walk(resolve_pose_profile({"walk": {"joints": ops}})) == base


def test_set_scale_offset_touch_only_the_named_joints() -> None:
    base = _walk(default_profile())
    scaled = _walk(resolve_pose_profile({"walk": {"joints": {"knee": {"scale": 0.5}}}}))
    for name in JOINT_NAMES:
        if name in ("l_knee", "r_knee"):
            assert scaled[name] == pytest.approx(0.5 * base[name])
        else:
            assert scaled[name] == base[name]

    offset = _walk(resolve_pose_profile({"walk": {"joints": {"hip": {"offset": {"mean": 0.1}}}}}))
    for name in JOINT_NAMES:
        if name in ("l_r_hip", "r_r_hip"):
            assert offset[name] == pytest.approx(base[name] + 0.1)
        else:
            assert offset[name] == base[name]

    replaced = _walk(resolve_pose_profile({"walk": {"joints": {"elbow": {"set": {"mean": 0.3, "harmonics": [[0.2, 0.0]]}}}}}), phi=0.7)
    for name in JOINT_NAMES:
        if name == "l_elbow":
            assert replaced[name] == pytest.approx(0.3 + 0.2 * math.sin(0.7))
        elif name == "r_elbow":
            assert replaced[name] == pytest.approx(0.3 + 0.2 * math.sin(0.7 + math.pi))
        else:
            assert replaced[name] == base[name]


def test_offset_adds_harmonics_as_phasors() -> None:
    added = Signal(0.1, ((0.3, 0.2), (0.05, 1.0))).plus(Signal(0.2, ((0.4, -1.1),)))
    assert added.mean == pytest.approx(0.3)
    for phi in (0.0, 0.9, 2.5, 5.1):
        direct = 0.3 + 0.3 * math.sin(phi + 0.2) + 0.4 * math.sin(phi - 1.1) + 0.05 * math.sin(2 * phi + 1.0)
        assert added.evaluate(phi, 0.0, 1.0) == pytest.approx(direct)


def test_set_on_an_unbound_joint_adds_it() -> None:
    profile = resolve_pose_profile({"walk": {"joints": {"ankle": {"set": {"mean": 0.1, "harmonics": [[0.2, 0.0]]}}, "y_waist": {"set": 0.25}}}})
    angles = _walk(profile, phi=0.4)
    assert angles["l_ankle"] == pytest.approx(0.1 + 0.2 * math.sin(0.4))
    assert angles["r_ankle"] == pytest.approx(0.1 + 0.2 * math.sin(0.4 + math.pi))
    assert angles["y_waist"] == 0.25
    assert _walk(default_profile(), phi=0.4)["l_ankle"] == 0.0


def test_symmetric_default_keeps_exact_antiphase_and_per_side_set_breaks_it() -> None:
    symmetric = resolve_pose_profile({"walk": {"joints": {"knee": {"scale": 0.8}}}})
    one_sided = resolve_pose_profile({"walk": {"symmetric": False, "joints": {"l_knee": {"set": {"mean": -0.3, "harmonics": [[0.1, 0.0]]}}}}})
    phi = 0.9
    for profile, antiphase in ((symmetric, True), (one_sided, False)):
        at_phi = _walk(profile, phi=phi)
        at_phi_pi = _walk(profile, phi=phi + math.pi)
        for left, right in PAIRS:
            same = math.isclose(at_phi[left], at_phi_pi[right], abs_tol=1e-9)
            assert same == (antiphase or left != "l_knee"), f"{left}/{right} antiphase={same}"
    assert _walk(one_sided, phi=phi)["r_knee"] == _walk(default_profile(), phi=phi)["r_knee"]


def test_symmetric_profile_rejects_a_side_name() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        resolve_pose_profile({"walk": {"joints": {"l_knee": {"scale": 0.5}}}})
    with pytest.raises(ValueError, match="symmetric"):
        resolve_pose_profile({"walk": {"joints": {"l_ankle": {"set": 0.1}}}})


def test_asymmetric_key_applies_to_both_sides_independently() -> None:
    profile = resolve_pose_profile({"walk": {"symmetric": False, "joints": {"knee": {"scale": 0.5}}}})
    base = _walk(default_profile())
    angles = _walk(profile)
    assert angles["l_knee"] == pytest.approx(0.5 * base["l_knee"])
    assert angles["r_knee"] == pytest.approx(0.5 * base["r_knee"])


def test_unknown_joint_op_and_base_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown joint"):
        resolve_pose_profile({"walk": {"joints": {"tail": {"scale": 0.5}}}})
    with pytest.raises(ValueError, match="takes"):
        resolve_pose_profile({"walk": {"joints": {"knee": {"multiply": 0.5}}}})
    with pytest.raises(ValueError, match="unknown base"):
        resolve_pose_profile({"walk": {"base": "moonwalk"}})
    with pytest.raises(ValueError, match="pose takes"):
        resolve_pose_profile({"crawl": {}})
    with pytest.raises(ValueError, match="idle"):
        resolve_pose_profile({"idle": {"joints": {"waist": {"scale": 2.0}}}})


def test_run_same_as_walk_inherits_ops_at_the_run_gain() -> None:
    profile = resolve_pose_profile({"walk": {"joints": {"knee": {"scale": 0.5}}}, "run": {"same_as": "walk"}})
    assert profile.run.joints == profile.walk.joints
    assert profile.run.gain == GainLaw(factor=1.6)
    gen = GaitGenerator()
    walk = gen.compute(0, _WALKING, 1.0, 0.0, phase=0.3, profile=profile)
    run = gen.compute(0, _RUNNING, 1.0, 0.0, phase=0.3, profile=profile)
    assert run["l_knee"] == pytest.approx(1.6 * walk["l_knee"])
    assert resolve_pose_profile({}).run.joints is not profile.run.joints


def test_phase_warp_identity_at_half_and_split_elsewhere() -> None:
    assert PhaseWarp(0.5)(1.234) == 1.234
    warp = PhaseWarp(0.25)
    assert warp(0.0) == 0.0
    assert warp(math.pi / 2) == pytest.approx(math.pi)
    assert warp(math.pi) == pytest.approx(math.pi + math.pi / 3)
    assert warp(2 * math.pi + math.pi / 2) == pytest.approx(3 * math.pi)
    assert warp(-math.pi / 2) == pytest.approx(-2 * math.pi + math.pi + (3 * math.pi / 2 - math.pi / 2) / 1.5)
    with pytest.raises(ValueError, match="split"):
        PhaseWarp(1.0)


def test_phase_warp_default_is_bit_exact_and_split_changes_the_pose() -> None:
    base = _walk(default_profile(), phi=1.1)
    assert _walk(resolve_pose_profile({"walk": {"phase_warp": {"split": 0.5}}}), phi=1.1) == base
    warped = resolve_pose_profile({"walk": {"phase_warp": {"split": 0.4}}})
    assert _walk(warped, phi=1.1) == _walk(default_profile(), phi=1.1 / 0.8)


def test_agent_type_pose_section_resolves_paths_builtins_and_extends(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "adult.yaml").write_text("name: adult\npose:\n  walk:\n    joints:\n      knee: {scale: 0.9}\n      hip: {offset: 0.05}\n  run:\n    same_as: walk\n")
    (builtin / "elder.yaml").write_text("name: elder\nextends: adult\npose:\n  walk:\n    cadence: {base: 0.3}\n    joints:\n      knee: {scale: 0.6}\n")
    (builtin / "robot.yaml").write_text("name: robot\n")
    scenario = tmp_path / "limper.yaml"
    scenario.write_text("extends: elder\npose:\n  walk:\n    symmetric: false\n")

    assert agent_type_pose_section("robot", builtin) == {}
    assert agent_type_pose_section("adult", builtin) == {"walk": {"joints": {"knee": {"scale": 0.9}, "hip": {"offset": 0.05}}}, "run": {"same_as": "walk"}}
    assert agent_type_pose_section("elder", builtin) == {"walk": {"cadence": {"base": 0.3}, "joints": {"knee": {"scale": 0.6}, "hip": {"offset": 0.05}}}, "run": {"same_as": "walk"}}
    assert agent_type_pose_section(str(scenario), builtin) == {"walk": {"symmetric": False, "cadence": {"base": 0.3}, "joints": {"knee": {"scale": 0.6}, "hip": {"offset": 0.05}}}, "run": {"same_as": "walk"}}
    assert agent_type_pose_section("ghost", builtin) is None
    assert agent_type_pose_section(str(tmp_path / "missing.yaml"), builtin) is None
    assert agent_type_pose_section("adult", None) is None

    profile = resolve_pose_profile(agent_type_pose_section("elder", builtin))
    assert profile.walk.cadence.base == 0.3
    assert profile.run.cadence.base == 0.3
    assert _walk(profile)["l_knee"] == pytest.approx(0.6 * _walk(default_profile())["l_knee"])


def test_agent_type_extends_cycle_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b\n")
    (tmp_path / "b.yaml").write_text("extends: a\n")
    with pytest.raises(ValueError, match="cycle"):
        agent_type_pose_section("a", tmp_path)


_IDLE = 0
_SEATED_HIP = 1.4211244138595862


def test_default_profile_carries_no_limits_and_the_builtin_idle() -> None:
    profile = default_profile()
    assert profile.walk.limits == {} and profile.run.limits == {}
    assert profile.idle is None


def test_profile_limits_override_the_global_table_per_joint() -> None:
    hip = JOINT_NAMES.index("l_r_hip")
    assert LIMITS[hip] == (-0.4, 0.7)
    wide = resolve_pose_profile({"walk": {"limits": {"l_r_hip": [-0.4, 2.0]}, "joints": {"hip": {"set": 1.5}}}})
    narrow = resolve_pose_profile({"walk": {"joints": {"hip": {"set": 1.5}}}})
    assert wide.walk.limits == {"l_r_hip": (-0.4, 2.0)}
    assert _walk(wide)["l_r_hip"] == 1.5
    assert _walk(wide)["r_r_hip"] == 0.7
    assert _walk(narrow)["l_r_hip"] == 0.7
    assert _walk(resolve_pose_profile({"walk": {"base": "seated"}}))["l_r_hip"] == _SEATED_HIP
    with pytest.raises(ValueError, match="limits take"):
        resolve_pose_profile({"walk": {"limits": {"l_r_hip": [2.0, -0.4]}}})
    with pytest.raises(ValueError, match="limits take"):
        resolve_pose_profile({"walk": {"limits": {"tail": [0.0, 1.0]}}})


def test_section_limits_merge_over_the_library_limits() -> None:
    profile = resolve_pose_profile({"walk": {"base": "seated", "limits": {"l_knee": [-3.0, 0.0]}}})
    assert profile.walk.limits["l_r_hip"] == (-0.4, 2.0)
    assert profile.walk.limits["l_knee"] == (-3.0, 0.0)


def test_seated_library_entry_holds_the_last_sit_frame() -> None:
    seated = load_library()["seated"]
    assert not seated.symmetric
    assert seated.cadence == CadenceLaw() and seated.gain == GainLaw()
    assert set(seated.joints) == set(JOINT_NAMES)
    assert all(js.signal.harmonics == () and js.shift == 0.0 for js in seated.joints.values())
    assert seated.joints["l_r_hip"].signal.mean == _SEATED_HIP
    assert seated.joints["l_knee"].signal.mean == pytest.approx(-1.5754, abs=1e-4)
    assert seated.joints["r_knee"].signal.mean == pytest.approx(-1.6450, abs=1e-4)
    for joint, js in seated.joints.items():
        lo, hi = seated.limits.get(joint, LIMITS[JOINT_NAMES.index(joint)])
        assert lo <= js.signal.mean <= hi, joint


def test_idle_from_a_library_profile_holds_the_pose_at_unit_gain() -> None:
    profile = resolve_pose_profile({"idle": {"base": "seated"}})
    assert profile.idle is not None and profile.idle.name == "seated"
    gen = GaitGenerator()
    start = gen.phase(5)
    frames = [gen.compute(5, _IDLE, 0.0, 0.1, profile=profile) for _ in range(3)]
    assert gen.phase(5) == pytest.approx(start + 3 * 2.0 * math.pi * 0.25 * 0.1)
    assert all(frame == frames[0] for frame in frames)
    assert frames[0]["l_r_hip"] == _SEATED_HIP
    assert frames[0]["l_knee"] == pytest.approx(-1.5754, abs=1e-4)
    assert gen.compute(5, 3, 2.0, 0.1, profile=profile) == frames[0]
    assert gen.compute(5, _IDLE, 0.0, 0.1)["l_r_hip"] == 0.0


def test_seated_wheelchair_library_entry_is_a_mirrored_held_posture() -> None:
    chair = load_library()["seated_wheelchair"]
    assert not chair.symmetric
    assert chair.cadence == CadenceLaw(base=0.6, per_speed=0.4, lo=0.6, hi=1.4)
    assert all(js.signal.harmonics == () and js.shift == 0.0 for js in chair.joints.values())
    mean = {joint: js.signal.mean for joint, js in chair.joints.items()}
    for joint in ("p_collar", "p_shoulder", "elbow", "p_hip", "r_hip", "knee", "ankle"):
        assert mean[f"l_{joint}"] == mean[f"r_{joint}"], joint
    for joint in ("y_shoulder", "r_shoulder", "y_ankle"):
        assert mean[f"l_{joint}"] == -mean[f"r_{joint}"], joint
    assert (mean["l_r_hip"], mean["l_knee"]) == (1.5, -0.91)
    for joint, value in mean.items():
        lo, hi = chair.limits.get(joint, LIMITS[JOINT_NAMES.index(joint)])
        assert lo <= value <= hi, joint
    profile = resolve_pose_profile({"idle": {"base": "seated_wheelchair"}})
    idle = GaitGenerator().compute(4, _IDLE, 0.0, 0.1, profile=profile)
    assert idle == {joint: mean.get(joint, 0.0) for joint in GaitGenerator.JOINT_NAMES}


def test_shipped_wheelchair_manual_pose_strokes_both_arms_together_on_the_held_seat() -> None:
    try:
        builtin = Path(get_package_share_directory("arena_humansim")) / "config" / "agent_types"
    except PackageNotFoundError:
        pytest.skip("arena_humansim is not installed")
    section = agent_type_pose_section("wheelchair_manual", builtin)
    assert section is not None and section["walk"]["base"] == "seated_wheelchair"
    profile = resolve_pose_profile(section)
    seat = {joint: js.signal.mean for joint, js in load_library()["seated_wheelchair"].joints.items()}
    stroked = set(section["walk"]["joints"])
    assert {"l_p_shoulder", "r_p_shoulder", "l_elbow", "r_elbow"} <= stroked
    gen = GaitGenerator()
    poses = [gen.compute(9, _WALKING, 0.9, 0.0, phase=2.0 * math.pi * k / 16, profile=profile) for k in range(16)]
    for pose in poses:
        for joint in JOINT_NAMES:
            if joint not in stroked:
                assert pose[joint] == seat.get(joint, 0.0), joint
        assert pose["l_p_shoulder"] == pytest.approx(pose["r_p_shoulder"], abs=1e-9)
        assert pose["l_elbow"] == pytest.approx(pose["r_elbow"], abs=1e-9)
        assert pose["l_y_shoulder"] == pytest.approx(-pose["r_y_shoulder"], abs=0.1)
        assert pose["l_r_shoulder"] == pytest.approx(-pose["r_r_shoulder"], abs=0.1)
    assert max(pose["l_p_shoulder"] for pose in poses) - min(pose["l_p_shoulder"] for pose in poses) > 0.1
    assert gen.compute(9, _RUNNING, 0.9, 0.0, phase=1.0, profile=profile) == gen.compute(9, _WALKING, 0.9, 0.0, phase=1.0, profile=profile)
    idle = gen.compute(9, _IDLE, 0.0, 0.0, phase=1.0, profile=profile)
    assert idle == {joint: seat.get(joint, 0.0) for joint in GaitGenerator.JOINT_NAMES}


def test_idle_builtin_rejects_overrides() -> None:
    with pytest.raises(ValueError, match="takes no overrides"):
        resolve_pose_profile({"idle": {"base": "idle", "symmetric": False}})
    with pytest.raises(ValueError, match="same_as"):
        resolve_pose_profile({"idle": {"same_as": "walk"}})


def test_wheelchair_section_keeps_the_seat_and_swings_both_arms_in_phase() -> None:
    arm = {"set": {"mean": 0.3, "harmonics": [[0.6, 0.0]]}}
    elbow = {"set": {"mean": 0.9, "harmonics": [[0.5, 3.1]]}}
    section = {
        "walk": {"base": "seated", "symmetric": False, "gain": {"min": 1.0, "max": 1.0}, "joints": {"l_p_shoulder": arm, "r_p_shoulder": arm, "l_elbow": elbow, "r_elbow": elbow}},
        "idle": {"base": "seated"},
        "run": {"same_as": "walk", "gain": {"factor": 1.0}},
    }
    profile = resolve_pose_profile(section)
    seated = {joint: js.signal.mean for joint, js in load_library()["seated"].joints.items()}
    assert (seated["l_r_hip"], seated["r_r_hip"]) == pytest.approx((1.4211, 1.4118), abs=1e-4)
    assert (seated["l_knee"], seated["r_knee"]) == pytest.approx((-1.5754, -1.6450), abs=1e-4)
    gen = GaitGenerator()
    for speed in (0.1, 0.9, 2.0):
        for phi in (0.0, 1.0, 2.5, 4.0):
            walk = gen.compute(9, _WALKING, speed, 0.0, phase=phi, profile=profile)
            run = gen.compute(9, _RUNNING, speed, 0.0, phase=phi, profile=profile)
            for pose in (walk, run):
                for joint in ("l_r_hip", "r_r_hip", "l_knee", "r_knee"):
                    assert pose[joint] == seated[joint], f"{joint} at speed {speed}"
                assert pose["l_p_shoulder"] == pose["r_p_shoulder"] == pytest.approx(0.3 + 0.6 * math.sin(phi))
                assert pose["l_elbow"] == pose["r_elbow"] == pytest.approx(0.9 + 0.5 * math.sin(phi + 3.1))
            assert run == walk
    for phi in (0.0, 2.5):
        idle = gen.compute(9, _IDLE, 0.0, 0.0, phase=phi, profile=profile)
        assert idle == {joint: seated.get(joint, 0.0) for joint in GaitGenerator.JOINT_NAMES}  # the wrists are not profiled, 0.0
    assert profile.run.gain == GainLaw(lo=1.0, hi=1.0, factor=1.0)
    assert profile.run.limits == profile.walk.limits


def test_run_same_as_walk_takes_law_overrides_over_the_resolved_walk() -> None:
    walk = {"joints": {"knee": {"scale": 0.5}}, "cadence": {"base": 0.3}}
    unscaled = resolve_pose_profile({"walk": walk, "run": {"same_as": "walk", "gain": {"factor": 1.0}}})
    gen = GaitGenerator()
    for speed in (0.3, 1.0, 2.5):
        assert gen.compute(0, _RUNNING, speed, 0.0, phase=0.8, profile=unscaled) == gen.compute(0, _WALKING, speed, 0.0, phase=0.8, profile=unscaled)

    tuned = resolve_pose_profile({"walk": walk, "run": {"same_as": "walk", "cadence": {"max": 3.0}, "phase_warp": {"split": 0.4}, "limits": {"l_knee": [-3.0, 0.0]}, "gain": {"min": 0.5}}})
    assert tuned.run.joints == tuned.walk.joints
    assert tuned.run.cadence == CadenceLaw(base=0.3, hi=3.0)
    assert tuned.run.gain == GainLaw(lo=0.5, factor=1.6)
    assert tuned.run.phase_warp == PhaseWarp(0.4) and tuned.walk.phase_warp == PhaseWarp(0.5)
    assert tuned.run.limits == {"l_knee": (-3.0, 0.0)} and tuned.walk.limits == {}


@pytest.mark.parametrize("extra", [{"base": "run"}, {"symmetric": False}, {"joints": {"knee": {"scale": 0.5}}}])
def test_run_same_as_walk_rejects_base_symmetric_and_joints(extra: dict) -> None:
    with pytest.raises(ValueError, match="same_as takes law overrides only"):
        resolve_pose_profile({"run": {"same_as": "walk", **extra}})


def _limp_types(tmp_path: Path) -> Path:
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    (builtin / "adult.yaml").write_text("name: adult\n")
    (builtin / "limp.yaml").write_text("name: limp\nextends: adult\nlocomotion:\n  kinematics: holonomic\n  phase_warp: {split: 0.4}\n")
    (builtin / "limp_posed.yaml").write_text("name: limp_posed\nextends: limp\npose:\n  walk:\n    joints:\n      knee: {scale: 0.8}\n  run:\n    same_as: walk\n")
    (builtin / "limp_explicit.yaml").write_text("name: limp_explicit\nextends: limp_posed\npose:\n  walk:\n    phase_warp: {split: 0.45}\n")
    (builtin / "limp_sampled.yaml").write_text("name: limp_sampled\nextends: limp_posed\nlocomotion:\n  phase_warp:\n    split: {mean: 0.35, std: 0.02, clip_low: 0.3, clip_high: 0.45}\n")
    (builtin / "limp_cadence.yaml").write_text("name: limp_cadence\nextends: limp\nlocomotion:\n  cadence: {max: 1.2}\n")
    return builtin


def test_locomotion_split_is_the_default_walk_warp_and_reaches_run_through_same_as(tmp_path: Path) -> None:
    builtin = _limp_types(tmp_path)
    assert agent_type_pose_section("limp_posed", builtin) == {"walk": {"joints": {"knee": {"scale": 0.8}}, "phase_warp": {"split": 0.4}}, "run": {"same_as": "walk"}}
    profile = resolve_pose_profile(agent_type_pose_section("limp_posed", builtin))
    assert profile.walk.phase_warp == PhaseWarp(0.4)
    assert profile.run.phase_warp == PhaseWarp(0.4)
    assert _walk(profile, phi=1.1)["l_knee"] == pytest.approx(0.8 * _walk(default_profile(), phi=1.1 / 0.8)["l_knee"])

    sampled = resolve_pose_profile(agent_type_pose_section("limp_sampled", builtin))
    assert sampled.walk.phase_warp == PhaseWarp(0.35) and sampled.run.phase_warp == PhaseWarp(0.35)


def test_explicit_pose_phase_warp_wins_over_the_locomotion_split(tmp_path: Path) -> None:
    builtin = _limp_types(tmp_path)
    assert agent_type_pose_section("limp_explicit", builtin)["walk"]["phase_warp"] == {"split": 0.45}
    profile = resolve_pose_profile(agent_type_pose_section("limp_explicit", builtin))
    assert profile.walk.phase_warp == PhaseWarp(0.45)
    assert profile.run.phase_warp == PhaseWarp(0.45)


def test_locomotion_split_without_a_pose_section_warps_the_default_walk(tmp_path: Path) -> None:
    builtin = _limp_types(tmp_path)
    assert agent_type_pose_section("adult", builtin) == {}
    for name in ("limp", "limp_cadence"):
        assert agent_type_pose_section(name, builtin) == {"walk": {"phase_warp": {"split": 0.4}}}
        profile = resolve_pose_profile(agent_type_pose_section(name, builtin))
        assert profile.walk == attrs.evolve(default_profile().walk, phase_warp=PhaseWarp(0.4))
        assert _walk(profile, phi=1.1) == _walk(default_profile(), phi=1.1 / 0.8)
        assert profile.idle is None
    scenario = tmp_path / "local_limp.yaml"
    scenario.write_text("extends: limp\n")
    assert agent_type_pose_section(str(scenario), builtin) == {"walk": {"phase_warp": {"split": 0.4}}}


def test_locomotion_split_rejects_a_non_number(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("locomotion:\n  phase_warp: {split: wide}\n")
    (tmp_path / "out_of_range.yaml").write_text("locomotion:\n  phase_warp: {split: 1.0}\n")
    with pytest.raises(ValueError, match="split must be a number"):
        agent_type_pose_section("bad", tmp_path)
    with pytest.raises(ValueError, match="split must lie in"):
        resolve_pose_profile(agent_type_pose_section("out_of_range", tmp_path))


def test_limp_section_breaks_antiphase_on_the_right_leg_only() -> None:
    section = {
        "walk": {
            "symmetric": False,
            "phase_warp": {"split": 0.58},
            "joints": {"r_knee": {"scale": 0.4}, "r_r_hip": {"scale": 0.7}, "waist": {"offset": {"mean": 0.0, "harmonics": [[0.06, 1.6]]}}},
        },
        "run": {"same_as": "walk"},
    }
    profile = resolve_pose_profile(section)
    plain = resolve_pose_profile({"walk": {"symmetric": False, "phase_warp": {"split": 0.58}}})
    assert profile.walk.phase_warp == PhaseWarp(split=0.58)
    gen = GaitGenerator()
    for phi in (0.3, 1.9, 3.3, 5.1):
        limp = gen.compute(4, _WALKING, 1.2, 0.0, phase=phi, profile=profile)
        base = gen.compute(4, _WALKING, 1.2, 0.0, phase=phi, profile=plain)
        assert limp["r_knee"] == pytest.approx(0.4 * base["r_knee"])
        assert limp["r_r_hip"] == pytest.approx(0.7 * base["r_r_hip"])
        for joint in ("l_knee", "l_r_hip", "l_p_shoulder", "r_p_shoulder", "l_elbow", "r_elbow"):
            assert limp[joint] == base[joint]
        assert limp["waist"] != base["waist"]
