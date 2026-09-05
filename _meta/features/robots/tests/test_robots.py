"""Unit tests for the arena_robots feature backend's submodule map."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "arena_robots_feature", Path(__file__).resolve().parents[1] / "robots.py"
)
robots = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(robots)


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
        "mpo700", "lidar/sick_s300",
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
