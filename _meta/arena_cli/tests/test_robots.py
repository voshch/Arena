"""Unit tests for the arena_robots feature backend's submodule map."""

from pathlib import Path

from arena_cli.complete import Context, _expand
from arena_cli.features import planners, robots

GITMODULES = """\
[submodule "arena_robots/robots/mpo700/meshes"]
\tpath = arena_robots/robots/mpo700/meshes
\turl = https://example.invalid/mpo700.git
\trobot = mpo700
[submodule "arena_robots/components/lidar/sick_s300/meshes"]
\tpath = arena_robots/components/lidar/sick_s300/meshes
\turl = https://example.invalid/sick_s300.git
\tcomponent = lidar/sick_s300
[submodule "arena_robots/components/arm/ur/meshes"]
\tpath = arena_robots/components/arm/ur/meshes
\turl = https://example.invalid/ur.git
\tcomponent = arm/ur
"""

ASSEMBLY = """\
prefix: ""
mounts:
  front:
    parent: base_footprint
    accepts: [lidar]
defaults:
  lidar:
    - variant: sick_s300
      mount: front
    - variant: sick_s300
      mount: rear
      overrides: {name: lidar_rear, topic: scan/rear}
"""


def _arena(tmp_path: Path, assembly: str | None = ASSEMBLY, robot: str = "mpo700") -> Path:
    sdk = tmp_path / "arena_robots"
    (sdk / ".gitmodules").parent.mkdir(parents=True, exist_ok=True)
    (sdk / ".gitmodules").write_text(GITMODULES)
    robot_dir = tmp_path / robots.ROBOTS_PREFIX / robot
    robot_dir.mkdir(parents=True)
    if assembly is not None:
        (robot_dir / "assembly.yaml").write_text(assembly)
    for family, body in (("lidar/sick_s300", "type: lidar"), ("arm/ur", "variants: [ur5e, ur10]")):
        family_dir = tmp_path / robots.COMPONENTS_PREFIX / family
        family_dir.mkdir(parents=True)
        (family_dir / "component.yaml").write_text(f"{body}\n")
    return tmp_path


def test_assembly_defaults_reads_type_and_variant(tmp_path):
    arena = _arena(tmp_path)
    assert robots.assembly_defaults(arena / robots.ROBOTS_PREFIX / "mpo700") == {"lidar/sick_s300"}


def test_assembly_defaults_ignores_variants_outside_the_defaults_block(tmp_path):
    arena = _arena(tmp_path, assembly="overrides:\n  lidar:\n    - variant: rplidar\ndefaults:\n")
    assert robots.assembly_defaults(arena / robots.ROBOTS_PREFIX / "mpo700") == set()


def test_assembly_defaults_without_assembly_file(tmp_path):
    arena = _arena(tmp_path, assembly=None)
    assert robots.assembly_defaults(arena / robots.ROBOTS_PREFIX / "mpo700") == set()


def test_robot_inherits_default_component_submodules(tmp_path):
    subs = robots.robot_submodules(_arena(tmp_path))
    assert subs["mpo700"] == [
        "arena_robots/arena_robots/robots/mpo700/meshes",
        "arena_robots/arena_robots/components/lidar/sick_s300/meshes",
    ]
    assert subs["lidar/sick_s300"] == ["arena_robots/arena_robots/components/lidar/sick_s300/meshes"]


def test_inherited_path_is_shared_so_rm_keeps_it(tmp_path):
    shared = robots._path_robots(_arena(tmp_path))
    assert shared["arena_robots/arena_robots/components/lidar/sick_s300/meshes"] == {
        "mpo700",
        "lidar/sick_s300",
    }


def test_variant_refs_resolve_to_their_family(tmp_path):
    arena = _arena(tmp_path, assembly="defaults:\n  arm:\n    - variant: ur5e\n")
    assert robots.robot_submodules(arena)["mpo700"] == [
        "arena_robots/arena_robots/robots/mpo700/meshes",
        "arena_robots/arena_robots/components/arm/ur/meshes",
    ]


def test_unknown_variant_is_skipped(tmp_path):
    arena = _arena(tmp_path, assembly="defaults:\n  arm:\n    - variant: nonesuch\n")
    assert robots.robot_submodules(arena)["mpo700"] == [
        "arena_robots/arena_robots/robots/mpo700/meshes",
    ]


def test_uri_owner():
    assert robots._uri_owner("robots/mpo700/meshes/body.dae") == "mpo700"
    assert robots._uri_owner("components/lidar/sick_s300/meshes/s300.dae") == "lidar/sick_s300"
    assert robots._uri_owner("configs/walls/brick.yaml") is None


def _check_out(arena: Path, path: str) -> None:
    (arena / path).mkdir(parents=True, exist_ok=True)
    (arena / path / ".git").write_text("gitdir: elsewhere\n")


def test_submodule_status_follows_the_checkout_without_a_git_repository(tmp_path):
    arena = _arena(tmp_path)
    _check_out(arena, "arena_robots/arena_robots/robots/mpo700/meshes")
    assert robots.submodule_status(arena) == {
        "arena_robots/arena_robots/robots/mpo700/meshes": "init",
        "arena_robots/arena_robots/components/lidar/sick_s300/meshes": "uninit",
        "arena_robots/arena_robots/components/arm/ur/meshes": "uninit",
    }


def test_robot_missing_a_default_component_is_pending(tmp_path, capsys):
    arena = _arena(tmp_path)
    _check_out(arena, "arena_robots/arena_robots/robots/mpo700/meshes")
    robots.cmd_ls(arena, None)
    assert "[ ] mpo700" in capsys.readouterr().out.splitlines()
    assert "mpo700" not in robots._ready_robots(arena)


def test_robot_with_every_submodule_checked_out_is_ready(tmp_path, capsys):
    arena = _arena(tmp_path)
    _check_out(arena, "arena_robots/arena_robots/robots/mpo700/meshes")
    _check_out(arena, "arena_robots/arena_robots/components/lidar/sick_s300/meshes")
    robots.cmd_ls(arena, None)
    lines = capsys.readouterr().out.splitlines()
    assert "[x] mpo700" in lines
    assert "[x] lidar/sick_s300" in lines
    assert "[ ] arm/ur" in lines
    assert "mpo700" in robots._ready_robots(arena)


def test_sim_preflight_suite_gives_each_robot_its_own_preflight_stage():
    suite = planners.preflight_suite(["go1", "boxer"])
    assert [(stage["name"], stage["robot"]) for stage in suite["stages"]] == [("preflight-go1", "go1"), ("preflight-boxer", "boxer")]
    assert {stage["config"]["scenario"]["file"] for stage in suite["stages"]} == {"preflight"}
    assert suite["launch"]["env.bootstrap_timeout"] == 90


def test_sim_preflight_suite_names_a_request_stage_by_a_path_safe_slug():
    suite = planners.preflight_suite(["rbkairos[arm_mount=arm/fr3,gripper_mount=gripper/franka_hand]"])
    assert [(stage["name"], stage["robot"]) for stage in suite["stages"]] == [("preflight-rbkairos-arm_mount-arm-fr3-gripper_mount-gripper-franka_hand", "rbkairos[arm_mount=arm/fr3,gripper_mount=gripper/franka_hand]")]


def test_request_needs_its_base_robot_and_each_placed_component(tmp_path):
    arena = _arena(tmp_path)
    _check_out(arena, "arena_robots/arena_robots/robots/mpo700/meshes")
    _check_out(arena, "arena_robots/arena_robots/components/lidar/sick_s300/meshes")
    ready = robots._ready_robots(arena)
    assert robots.request_refs(arena, "mpo700[top=arm/ur5e,front=none]") == ("mpo700", ["arm/ur"])
    assert not robots._spec_ready(arena, "mpo700[top=arm/ur5e]", ready)
    _check_out(arena, "arena_robots/arena_robots/components/arm/ur/meshes")
    assert robots._spec_ready(arena, "mpo700[top=arm/ur5e]", ready)
    assert robots._spec_ready(arena, "mpo700", ready)


def test_planner_preflight_suite_keeps_each_contestant_robot():
    assert [(stage["name"], stage["robot"]) for stage in planners.preflight_suite()["stages"]] == [("preflight", "auto")]


_RUNTIME_MANIFEST = {"launch": {"arena_bringup/arena_runtime.launch.py": {"sim": {"desc": "simulator", "default": "gazebo"}}}, "values": {"sim": ["gazebo", "isaac", "mujoco", "flatsim"]}}


def test_preflight_verbs_complete_the_simulator() -> None:
    for spec in (robots.COMMANDS["check"].complete, planners.COMMANDS["test"].complete):
        assert "sim:=" in {value for value, _desc, _group in _expand(spec, [], "", Context("", _RUNTIME_MANIFEST)).items}
        assert {"gazebo", "isaac", "mujoco", "flatsim"} <= {value for value, _desc, _group in _expand(spec, [], "sim:=", Context("", _RUNTIME_MANIFEST)).items}
