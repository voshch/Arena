from __future__ import annotations

import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "uv_workspace.py"
spec = importlib.util.spec_from_file_location("uv_workspace", TOOL)
assert spec is not None and spec.loader is not None
uvw = importlib.util.module_from_spec(spec)
sys.modules["uv_workspace"] = uvw
spec.loader.exec_module(uvw)

PACKAGE_XML = '<?xml version="1.0"?>\n<package format="3"><name>{name}</name><version>0.0.0</version></package>\n'

ROOT_TOML = """# header comment
[tool.uv]
package = false

[project]
name = "root"
version = "0.1.0"
dependencies = []
"""


def write(path: pathlib.Path, text: str = "") -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def project_toml(name: str, deps: str = "", extra: str = "", virtual: bool = True, dynamic: str = "") -> str:
    text = f'[project]\nname = "{name}"\nversion = "0.1.0"\ndependencies = [{deps}]\n{extra}'
    if dynamic:
        text += f"dynamic = [{dynamic}]\n"
    if virtual:
        text += "\n[tool.uv]\npackage = false\n"
    return text


def make_package(root: pathlib.Path, rel: str, pyproject: str | None = None, name: str | None = None) -> pathlib.Path:
    pkg = root / rel
    write(pkg / "package.xml", PACKAGE_XML.format(name=name or pathlib.PurePosixPath(rel).name))
    write(pkg / "pyproject.toml", pyproject if pyproject is not None else project_toml(name or pathlib.PurePosixPath(rel).name))
    return pkg


def lock_entry(requirement: str) -> str:
    match = re.fullmatch(r"([A-Za-z0-9._-]+)(.*)", requirement)
    assert match is not None
    name, specifier = uvw.canonical(match.group(1)), match.group(2).strip()
    return f'{{ name = "{name}", specifier = "{specifier}" }}' if specifier else f'{{ name = "{name}" }}'


def fresh_lock(root: pathlib.Path, repo: str = ".", registry: list[tuple[str, str]] | tuple[tuple[str, str], ...] = ()) -> None:
    """Write a uv.lock matching the repo's current manifests, plus the given registry packages."""
    manifest = tomllib.loads((root / repo / "pyproject.toml").read_text())
    projects = [(manifest, ".")]
    for member in uvw.repo_members(root).get(repo, []):
        projects.append((tomllib.loads((root / member / "pyproject.toml").read_text()), member))
    lines = ["version = 1", 'requires-python = "==3.12.*"', ""]
    exclude_newer = manifest.get("tool", {}).get("uv", {}).get("exclude-newer")
    if exclude_newer:
        lines += ["[options]", f'exclude-newer = "{exclude_newer}"', ""]
    names = sorted(uvw.canonical(data["project"]["name"]) for data, _ in projects)
    if len(names) > 1:
        lines += ["[manifest]", "members = [" + ", ".join(f'"{name}"' for name in names) + "]", ""]
    for data, rel in projects:
        lines += ["[[package]]", f'name = "{uvw.canonical(data["project"]["name"])}"', 'version = "0.0.0"', f'source = {{ virtual = "{rel}" }}', ""]
        deps = data["project"].get("dependencies", [])
        groups = {uvw.canonical(group): entries for group, entries in data.get("dependency-groups", {}).items() if entries}
        if deps or groups:
            lines += ["[package.metadata]", "requires-dist = [" + ", ".join(lock_entry(dep) for dep in deps) + "]", ""]
        if groups:
            lines += ["[package.metadata.requires-dev]"] + [f"{group} = [" + ", ".join(lock_entry(dep) for dep in entries) + "]" for group, entries in groups.items()] + [""]
    for name, version in registry:
        lines += ["[[package]]", f'name = "{name}"', f'version = "{version}"', 'source = { registry = "https://pypi.org/simple" }', ""]
    write(root / repo / "uv.lock", "\n".join(lines))


def refresh(root: pathlib.Path) -> pathlib.Path:
    """Bring the root members table and the superproject lock in line with the tree."""
    members = uvw.relative_members(uvw.repo_members(root).get(".", []), root, root)
    write(root / "pyproject.toml", uvw.with_root_members((root / "pyproject.toml").read_text(), members))
    fresh_lock(root)
    return root


def run(capsys: pytest.CaptureFixture[str], root: pathlib.Path, *args: str) -> tuple[int, str, str]:
    code = uvw.main(["--root", str(root), *args])
    out = capsys.readouterr()
    return code, out.out, out.err


@pytest.fixture
def repo(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "repo"
    write(root / "pyproject.toml", ROOT_TOML)
    make_package(root, "pkg_a")
    return root


def test_discovery_finds_package_dir_member(repo: pathlib.Path) -> None:
    assert uvw.compute_members(repo) == ["pkg_a"]


def test_discovery_finds_src_member_when_package_dir_has_no_project(repo: pathlib.Path) -> None:
    make_package(repo, "pkg_b", pyproject="[tool.other]\nx = 1\n")
    write(repo / "pkg_b/src/pyproject.toml", project_toml("pkg_b"))
    assert uvw.compute_members(repo) == ["pkg_a", "pkg_b/src"]


def test_discovery_skips_package_without_project_table(repo: pathlib.Path) -> None:
    make_package(repo, "pkg_c", pyproject="[tool.other]\nx = 1\n")
    assert uvw.compute_members(repo) == ["pkg_a"]


@pytest.mark.parametrize("marker", ["COLCON_IGNORE", "AMENT_IGNORE"])
def test_discovery_prunes_ignore_marker_dirs(repo: pathlib.Path, marker: str) -> None:
    make_package(repo, "pkg_b")
    write(repo / "pkg_b" / marker)
    assert uvw.compute_members(repo) == ["pkg_a"]


def test_discovery_prunes_ignore_marker_ancestor(repo: pathlib.Path) -> None:
    write(repo / "vendor" / "COLCON_IGNORE")
    make_package(repo, "vendor/inner")
    assert uvw.compute_members(repo) == ["pkg_a"]


@pytest.mark.parametrize("rel", ["arena_planners/planners/p", "arena_isaac/isaac", "arena_tools/t", "arena_robots/deps/r"])
def test_discovery_prunes_excluded_subtrees(repo: pathlib.Path, rel: str) -> None:
    make_package(repo, rel)
    assert uvw.compute_members(repo) == ["pkg_a"]


def test_discovery_keeps_sibling_of_excluded_subtree(repo: pathlib.Path) -> None:
    make_package(repo, "arena_planners/arena_planners")
    assert uvw.compute_members(repo) == ["arena_planners/arena_planners", "pkg_a"]


@pytest.mark.parametrize("skipped", ["build", "install", "log", ".venv", "node_modules", "_meta/repos"])
def test_discovery_prunes_skip_dirs(repo: pathlib.Path, skipped: str) -> None:
    make_package(repo, f"{skipped}/pkg_z")
    assert uvw.compute_members(repo) == ["pkg_a"]


def test_discovery_adds_extra_member_without_package_xml(repo: pathlib.Path) -> None:
    write(repo / "_meta/arena_cli/pyproject.toml", project_toml("arena-cli", virtual=False))
    assert uvw.compute_members(repo) == ["_meta/arena_cli", "pkg_a"]


def test_discovery_ignores_extra_member_without_project_table(repo: pathlib.Path) -> None:
    write(repo / "_meta/arena_cli/pyproject.toml", "[tool.x]\na = 1\n")
    assert uvw.compute_members(repo) == ["pkg_a"]


def test_members_command_prints_one_per_line(repo: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_package(repo, "pkg_b")
    code, out, _ = run(capsys, repo, "members")
    assert code == 0
    assert out.splitlines() == ["pkg_a", "pkg_b"]


def test_canonical_collapses_separator_runs() -> None:
    assert uvw.canonical("Foo_-.Bar") == "foo-bar"


def clean_repo(root: pathlib.Path) -> pathlib.Path:
    write(root / "pyproject.toml", ROOT_TOML)
    make_package(root, "pkg_a")
    return refresh(root)


def test_check_passes_for_clean_repo(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, clean_repo(tmp_path), "check")
    assert (code, out, err) == (0, "", "")


def test_check_reports_missing_project_fields(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "pkg_a/pyproject.toml", '[project]\ndescription = "x"\n\n[tool.uv]\npackage = false\n')
    code, out, _ = run(capsys, root, "check")
    assert code == 1
    lines = out.splitlines()
    assert "pkg_a: [project].name must be a static string" in lines
    assert "pkg_a: [project].version must be a static string" in lines
    assert "pkg_a: [project].dependencies must be a list" in lines


def test_check_reports_non_string_version(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "pkg_a/pyproject.toml", '[project]\nname = "a"\nversion = 3\ndependencies = []\n\n[tool.uv]\npackage = false\n')
    refresh(root)
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["pkg_a: [project].version must be a static string"]


@pytest.mark.parametrize("field", ["version", "dependencies", "optional-dependencies", "requires-python"])
def test_check_reports_forbidden_dynamic(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], field: str) -> None:
    root = clean_repo(tmp_path)
    write(root / "pkg_a/pyproject.toml", project_toml("a", dynamic=f'"readme", "{field}"'))
    code, out, _ = run(capsys, root, "check")
    assert code == 1
    assert f"pkg_a: [project].dynamic must not contain {field}" in out


def test_check_allows_harmless_dynamic(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "pkg_a/pyproject.toml", project_toml("a", dynamic='"readme", "scripts"'))
    refresh(root)
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_reports_virtual_member_without_package_false(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "pkg_a/pyproject.toml", project_toml("a", virtual=False))
    refresh(root)
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["pkg_a: [tool.uv] package = false is required"]


def test_check_reports_real_member_with_package_false(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    make_package(root, "arena_planners/arena_planners", pyproject=project_toml("ap", virtual=True))
    refresh(root)
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["arena_planners/arena_planners: [tool.uv] package = false is not allowed for an installed member"]


def test_check_accepts_real_member_without_package_false(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "_meta/arena_cli/pyproject.toml", project_toml("arena-cli", virtual=False))
    refresh(root)
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def with_setup(root: pathlib.Path, setup_body: str, pyproject: str | None = None) -> None:
    write(root / "pkg_a/setup.py", f"from setuptools import setup\n\nsetup({setup_body})\n")
    if pyproject is not None:
        write(root / "pkg_a/pyproject.toml", pyproject)
        refresh(root)


@pytest.mark.parametrize("kwarg", ["version", "install_requires", "extras_require", "tests_require"])
def test_check_reports_removed_setup_kwargs(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], kwarg: str) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, f"name='a', {kwarg}=[]")
    code, out, _ = run(capsys, root, "check")
    assert code == 1
    assert f"pkg_a: setup.py passes {kwarg}, declare it in pyproject.toml only" in out


@pytest.mark.parametrize(
    ("kwarg", "field"),
    [
        ("description", "description"),
        ("long_description", "readme"),
        ("license", "license"),
        ("maintainer", "maintainers"),
        ("maintainer_email", "maintainers"),
        ("author", "authors"),
        ("author_email", "authors"),
        ("keywords", "keywords"),
        ("classifiers", "classifiers"),
        ("project_urls", "urls"),
    ],
)
def test_check_reports_setup_kwarg_without_dynamic_or_static_field(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], kwarg: str, field: str) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, f"name='a', {kwarg}='x'")
    code, out, _ = run(capsys, root, "check")
    assert code == 1
    assert f"pkg_a: setup.py {kwarg} needs {field} in [project].dynamic or a static [project].{field}" in out


def test_check_accepts_setup_kwarg_declared_dynamic(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, "name='a', description='x', author='y', keywords=['k']", project_toml("a", dynamic='"description", "authors", "keywords"'))
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_accepts_setup_kwarg_declared_static(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, "name='a', description='x'", project_toml("a", extra='description = "x"\n'))
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_ignores_setup_kwargs_outside_the_mapping(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, "name='a', packages=['a'], data_files=[], zip_safe=True, **extra")
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_entry_points_literal_console_scripts_needs_scripts(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, "entry_points={'console_scripts': ['a=a:main']}")
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["pkg_a: setup.py entry_points needs scripts in [project].dynamic or a static [project].scripts"]


def test_check_entry_points_literal_passes_with_matching_fields(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    body = "entry_points={'console_scripts': [], 'gui_scripts': [], 'ament.x': []}"
    with_setup(root, body, project_toml("a", dynamic='"scripts", "gui-scripts"', extra="entry-points = {}\n"))
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_entry_points_literal_requires_each_group(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, "entry_points={'gui_scripts': [], 'ament.x': []}")
    _, out, _ = run(capsys, root, "check")
    assert sorted(out.splitlines()) == [
        "pkg_a: setup.py entry_points needs entry-points in [project].dynamic or a static [project].entry-points",
        "pkg_a: setup.py entry_points needs gui-scripts in [project].dynamic or a static [project].gui-scripts",
    ]


def test_check_entry_points_non_literal_requires_all_three(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, "entry_points=ENTRY_POINTS")
    _, out, _ = run(capsys, root, "check")
    assert len(out.splitlines()) == 3
    for field in ("scripts", "gui-scripts", "entry-points"):
        assert any(f"needs {field} in" in line for line in out.splitlines())


def test_check_entry_points_non_literal_passes_with_all_dynamic(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    with_setup(root, "entry_points=ENTRY_POINTS", project_toml("a", dynamic='"scripts", "gui-scripts", "entry-points"'))
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_exempts_src_member_from_setup_py_rule(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    make_package(root, "pkg_b", pyproject="[tool.other]\nx = 1\n")
    write(root / "pkg_b/src/pyproject.toml", project_toml("pkg_b"))
    write(root / "pkg_b/setup.py", "from setuptools import setup\nsetup(version='1', install_requires=[], description='x')\n")
    refresh(root)
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_skips_setup_py_without_package_xml(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "_meta/arena_cli/pyproject.toml", project_toml("arena-cli", virtual=False))
    write(root / "_meta/arena_cli/setup.py", "from setuptools import setup\nsetup(version='1')\n")
    refresh(root)
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_check_reports_unparsable_setup_py(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "pkg_a/setup.py", "setup(\n")
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["pkg_a: setup.py does not parse"]


COMPOSE_ROOT = """[project]
name = "arena-rosnav"
version = "0.1.0"
description = "ignored"
requires-python = "==3.12.*"
dependencies = [
    "pip",
    "PyYAML>=6",
]

[dependency-groups]
dev = ["pytest", "ruff"]
empty = []

[tool.uv]
package = false
default-groups = ["dev"]
exclude-newer = "2026-10-03T00:00:00Z"
compile-bytecode = true
cache-keys = 3

[tool.uv.pip]
strict = true
"""


def compose_repo(root: pathlib.Path) -> pathlib.Path:
    write(root / "pyproject.toml", COMPOSE_ROOT)
    make_package(root, "pkg_a")
    make_package(root, "deep/pkg_b")
    fresh_lock(root)
    return root


def test_compose_writes_members_relative_to_default_out(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    code, out, _ = run(capsys, root, "compose")
    assert code == 0
    assert out.strip() == f"composed 2 members and 0 pins into {root.resolve() / '.uv-workspace' / 'pyproject.toml'}"
    data = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert data["tool"]["uv"]["workspace"]["members"] == ["../deep/pkg_b", "../pkg_a"]


def test_compose_members_relative_to_custom_out(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    code, _, _ = run(capsys, root, "compose", "--out", "gen/ws")
    assert code == 0
    data = tomllib.loads((root / "gen/ws/pyproject.toml").read_text())
    assert data["tool"]["uv"]["workspace"]["members"] == ["../../deep/pkg_b", "../../pkg_a"]


def test_compose_accepts_absolute_out(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path / "repo")
    target = tmp_path / "elsewhere"
    run(capsys, root, "compose", "--out", str(target))
    data = tomllib.loads((target / "pyproject.toml").read_text())
    assert data["tool"]["uv"]["workspace"]["members"] == ["../repo/deep/pkg_b", "../repo/pkg_a"]


def test_compose_round_trips_copied_values(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    run(capsys, root, "compose")
    source = tomllib.loads(COMPOSE_ROOT)
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert composed["project"] == {key: value for key, value in source["project"].items() if key != "description"}
    assert composed["dependency-groups"] == source["dependency-groups"]
    uv = dict(composed["tool"]["uv"])
    del uv["workspace"]
    assert uv == source["tool"]["uv"]
    assert isinstance(composed["tool"]["uv"]["exclude-newer"], str)


def test_compose_without_optional_sections(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path / "pyproject.toml", '[project]\nname = "x"\n')
    code, _, _ = run(capsys, tmp_path, "compose")
    composed = tomllib.loads((tmp_path / ".uv-workspace/pyproject.toml").read_text())
    assert code == 0
    assert composed == {"project": {"name": "x"}, "tool": {"uv": {"workspace": {"members": []}}}}


def test_compose_leaves_out_absent_and_uninitialized_members(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    (root / "arena_robots").mkdir()
    code, out, _ = run(capsys, root, "compose")
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert code == 0
    assert "2 members" in out
    assert all("arena_robots" not in member for member in composed["tool"]["uv"]["workspace"]["members"])


def test_compose_second_run_reports_up_to_date_without_rewriting(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    run(capsys, root, "compose")
    target = root / ".uv-workspace/pyproject.toml"
    first = target.stat().st_mtime_ns
    content = target.read_bytes()
    code, out, _ = run(capsys, root, "compose")
    assert code == 0
    assert "up to date" in out
    assert target.stat().st_mtime_ns == first
    assert target.read_bytes() == content


def test_compose_rewrites_when_membership_changes(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    run(capsys, root, "compose")
    make_package(root, "pkg_c")
    _, out, _ = run(capsys, root, "compose")
    assert "composed 3 members" in out


def test_compose_ignores_its_own_output_directory(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    make_package(root, ".uv-workspace/inner")
    _, out, _ = run(capsys, root, "members")
    assert ".uv-workspace/inner" not in out


def test_compose_errors_for_tracked_uv_sources(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path / "pyproject.toml", '[project]\nname = "x"\n\n[tool.uv.sources]\nx = { workspace = true }\n')
    code, _, err = run(capsys, tmp_path, "compose")
    assert code == 2
    assert "[tool.uv.sources]" in err
    assert not (tmp_path / ".uv-workspace").exists()


def test_compose_replaces_root_workspace_with_tree_members(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = compose_repo(tmp_path)
    write(root / "pyproject.toml", COMPOSE_ROOT + '\n[tool.uv.workspace]\nmembers = ["pkg_a"]\n')
    code, _, _ = run(capsys, root, "compose")
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert code == 0
    assert composed["tool"]["uv"]["workspace"] == {"members": ["../deep/pkg_b", "../pkg_a"]}


@pytest.mark.parametrize(
    ("body", "key"),
    [
        ('[tool.uv]\nratio = 1.5\n', "tool.uv.ratio"),
        ('[tool.uv]\nwhen = 2026-10-03\n', "tool.uv.when"),
        ('[tool.uv]\ninline = [{a = 1}]\n', "tool.uv.inline"),
        ('[tool.uv]\nnested = [[1], [2]]\n', "tool.uv.nested"),
        ('[tool.uv.pip.deeper]\nx = 1\n', "tool.uv.pip.deeper"),
        ('[dependency-groups]\ndev = [{include-group = "x"}]\n', "dependency-groups.dev"),
        ('[dependency-groups]\nratio = 1.5\n', "dependency-groups.ratio"),
    ],
)
def test_compose_errors_for_unsupported_value_type(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], body: str, key: str) -> None:
    write(tmp_path / "pyproject.toml", f'[project]\nname = "x"\n\n{body}')
    code, _, err = run(capsys, tmp_path, "compose")
    assert code == 2
    assert key in err


def test_compose_errors_without_root_pyproject(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, tmp_path, "compose")
    assert code == 2
    assert "not found" in err


def test_compose_quotes_special_strings(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path / "pyproject.toml", '[project]\nname = "x"\ndependencies = ["a @ file:///p q", "b\\\\c"]\n')
    run(capsys, tmp_path, "compose")
    composed = tomllib.loads((tmp_path / ".uv-workspace/pyproject.toml").read_text())
    assert composed["project"]["dependencies"] == ["a @ file:///p q", "b\\c"]


def test_check_reports_tracked_uv_sources_in_root(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    write(root / "pyproject.toml", (root / "pyproject.toml").read_text() + "\n[tool.uv.sources]\nx = 1\n")
    code, out, _ = run(capsys, root, "check")
    assert code == 1
    assert out.splitlines() == ["pyproject.toml: tracked root must not contain [tool.uv.sources]"]


def test_check_errors_without_root_pyproject(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, tmp_path, "check")
    assert code == 2
    assert "not found" in err


def test_check_validates_every_computed_member(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    make_package(root, "pkg_b", pyproject=project_toml("b", virtual=False))
    refresh(root)
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["pkg_b: [tool.uv] package = false is required"]


SUPER_ROOT = """[project]
name = "arena-rosnav"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = ["pip"]

[dependency-groups]
dev = ["pytest"]

[tool.uv]
package = false
default-groups = ["dev"]
exclude-newer = "2026-10-03T00:00:00Z"

[tool.uv.workspace]
members = []

[tool.ruff]
line-length = 120
"""


def make_submodule(root: pathlib.Path, repo: str, as_dir: bool = False) -> pathlib.Path:
    path = root / repo
    if as_dir:
        (path / ".git").mkdir(parents=True)
    else:
        write(path / ".git", "gitdir: /elsewhere\n")
    return path


def super_repo(root: pathlib.Path) -> pathlib.Path:
    """Superproject with one in-tree member and a submodule hosting two members, manifests and locks fresh."""
    write(root / "pyproject.toml", SUPER_ROOT)
    make_package(root, "pkg_a", pyproject=project_toml("pkg_a", deps='"numpy>=1.24", "PyYAML"'))
    make_submodule(root, "sub")
    make_package(root, "sub/pkg_s", pyproject=project_toml("pkg_s", deps='"attrs"'))
    make_package(root, "sub/nested/pkg_t", pyproject=project_toml("pkg_t"))
    repos = uvw.repo_members(root)
    for repo, text in uvw.expected_manifests(root, repos, uvw.load_root(root)).items():
        write(root / repo / "pyproject.toml", text)
    fresh_lock(root, registry=[("numpy", "2.1.0"), ("pyyaml", "6.0.3"), ("pytest", "9.0.0")])
    fresh_lock(root, "sub", registry=[("attrs", "26.1.0"), ("numpy", "2.1.0")])
    return root


def test_repo_members_groups_members_by_enclosing_checkout(tmp_path: pathlib.Path) -> None:
    make_package(tmp_path, "pkg_a")
    make_submodule(tmp_path, "sub")
    make_package(tmp_path, "sub/deep/pkg_s")
    make_submodule(tmp_path, "dirsub", as_dir=True)
    make_package(tmp_path, "dirsub/pkg_d")
    assert uvw.repo_members(tmp_path) == {".": ["pkg_a"], "dirsub": ["dirsub/pkg_d"], "sub": ["sub/deep/pkg_s"]}


def test_repo_manifest_copies_resolution_settings_and_depends_on_members(tmp_path: pathlib.Path) -> None:
    make_package(tmp_path, "ext/my_sub/b", pyproject=project_toml("Pkg_B"))
    make_package(tmp_path, "ext/my_sub/a/x", pyproject=project_toml("x"))
    text = uvw.repo_manifest_text("ext/my_sub", ["ext/my_sub/b", "ext/my_sub/a/x"], tmp_path, tomllib.loads(SUPER_ROOT))
    assert tomllib.loads(text) == {
        "project": {"name": "my-sub-workspace", "version": "0.0.0", "requires-python": "==3.12.*", "dependencies": ["pkg-b", "x"]},
        "tool": {
            "uv": {
                "package": False,
                "exclude-newer": "2026-10-03T00:00:00Z",
                "sources": {"pkg-b": {"workspace": True}, "x": {"workspace": True}},
                "workspace": {"members": ["a/x", "b"]},
            }
        },
    }


def test_root_members_rewrite_keeps_rest_of_file_byte_identical() -> None:
    rewritten = uvw.with_root_members(SUPER_ROOT, ["b", "a"])
    assert rewritten == SUPER_ROOT.replace("members = []", 'members = [\n    "b",\n    "a",\n]')


def test_root_members_rewrite_replaces_multiline_array() -> None:
    text = SUPER_ROOT.replace("members = []", 'members = [\n    "old",\n    "older",\n]')
    assert uvw.with_root_members(text, []) == SUPER_ROOT


def test_root_members_rewrite_inserts_key_into_existing_table() -> None:
    text = SUPER_ROOT.replace("members = []\n", 'exclude = ["x"]\n')
    rewritten = uvw.with_root_members(text, ["a"])
    assert tomllib.loads(rewritten)["tool"]["uv"]["workspace"] == {"members": ["a"], "exclude": ["x"]}
    assert rewritten.replace('members = [\n    "a",\n]\n', "") == text


def test_root_members_rewrite_appends_missing_table() -> None:
    text = SUPER_ROOT.replace("[tool.uv.workspace]\nmembers = []\n\n", "")
    rewritten = uvw.with_root_members(text, ["a"])
    assert rewritten.startswith(text.rstrip("\n"))
    assert tomllib.loads(rewritten)["tool"]["uv"]["workspace"] == {"members": ["a"]}


def test_compose_pins_union_of_repo_locks_after_root_constraints(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    write(root / "pyproject.toml", (root / "pyproject.toml").read_text().replace("package = false\n", 'package = false\nconstraint-dependencies = ["setuptools<80"]\n', 1))
    code, out, _ = run(capsys, root, "compose")
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert code == 0
    assert "composed 3 members and 4 pins" in out
    assert composed["tool"]["uv"]["constraint-dependencies"] == ["setuptools<80", "attrs==26.1.0", "numpy==2.1.0", "pytest==9.0.0", "pyyaml==6.0.3"]
    assert composed["tool"]["uv"]["workspace"]["members"] == ["../pkg_a", "../sub/nested/pkg_t", "../sub/pkg_s"]


def test_compose_without_submodule_checkout_takes_no_pins_from_it(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    shutil.rmtree(root / "sub")
    (root / "sub").mkdir()
    run(capsys, root, "compose")
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert composed["tool"]["uv"]["constraint-dependencies"] == ["numpy==2.1.0", "pytest==9.0.0", "pyyaml==6.0.3"]
    assert composed["tool"]["uv"]["workspace"]["members"] == ["../pkg_a"]


def test_compose_errors_on_conflicting_repo_pins(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    fresh_lock(root, "sub", registry=[("attrs", "26.1.0"), ("numpy", "2.2.0")])
    code, _, err = run(capsys, root, "compose")
    assert code == 2
    assert "numpy: 2.1.0 in uv.lock, 2.2.0 in sub/uv.lock" in err
    assert not (root / ".uv-workspace").exists()


def test_compose_leaves_forked_package_unpinned(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    fresh_lock(root, "sub", registry=[("attrs", "26.1.0"), ("attrs", "25.4.0")])
    code, _, err = run(capsys, root, "compose")
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert code == 0
    assert "sub/uv.lock locks several versions of attrs, left unpinned" in err
    assert not any(pin.startswith("attrs") for pin in composed["tool"]["uv"]["constraint-dependencies"])


def test_compose_errors_when_present_repo_has_no_lock(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    (root / "sub/uv.lock").unlink()
    code, _, err = run(capsys, root, "compose")
    assert code == 2
    assert "sub/uv.lock not found" in err


def test_compose_pins_only_registry_packages(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    write(root / "sub/uv.lock", (root / "sub/uv.lock").read_text() + '\n[[package]]\nname = "gitdep"\nversion = "1.0"\nsource = { git = "https://example.invalid/x" }\n')
    run(capsys, root, "compose")
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert not any(pin.startswith(("gitdep", "pkg-", "arena-rosnav")) for pin in composed["tool"]["uv"]["constraint-dependencies"])


def test_check_passes_for_fresh_superproject_and_submodule(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, super_repo(tmp_path), "check")
    assert (code, out, err) == (0, "", "")


def test_check_reports_stale_root_members(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    make_package(root, "pkg_b")
    _, out, _ = run(capsys, root, "check")
    assert "pyproject.toml: [tool.uv.workspace] members are out of date, run python3 _meta/tools/uv_workspace.py lock" in out.splitlines()


@pytest.mark.parametrize("edit", ["missing", "stale"])
def test_check_reports_submodule_manifest_out_of_date(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], edit: str) -> None:
    root = super_repo(tmp_path)
    if edit == "missing":
        (root / "sub/pyproject.toml").unlink()
    else:
        make_package(root, "sub/pkg_u")
    code, out, _ = run(capsys, root, "check")
    assert code == 1
    assert out.splitlines() == ["sub/pyproject.toml is out of date, run python3 _meta/tools/uv_workspace.py lock"]


def test_check_reports_missing_repo_lock(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    (root / "sub/uv.lock").unlink()
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["sub/uv.lock is missing, run python3 _meta/tools/uv_workspace.py lock"]


@pytest.mark.parametrize(
    ("path", "old", "new", "reason"),
    [
        ("sub/pkg_s/pyproject.toml", '"attrs"', '"attrs", "rich"', "pkg-s dependencies"),
        ("pkg_a/pyproject.toml", '"numpy>=1.24"', '"numpy>=2"', "pkg-a dependencies"),
        ("pkg_a/pyproject.toml", 'name = "pkg_a"', 'name = "pkg_renamed"', "members"),
        ("pyproject.toml", 'dev = ["pytest"]', 'dev = ["pytest", "hypothesis"]', "arena-rosnav dependency-groups"),
        ("pyproject.toml", 'dependencies = ["pip"]', 'dependencies = ["pip", "lark"]', "arena-rosnav dependencies"),
    ],
)
def test_check_reports_stale_lock(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str], path: str, old: str, new: str, reason: str) -> None:
    root = super_repo(tmp_path)
    write(root / path, (root / path).read_text().replace(old, new, 1))
    code, out, _ = run(capsys, root, "check")
    assert code == 1
    assert any(line.endswith("_meta/tools/uv_workspace.py lock") and reason in line and "uv.lock is stale" in line for line in out.splitlines()), out


def test_check_matches_specifiers_regardless_of_order_and_spacing(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    lock = (root / "uv.lock").read_text().replace('specifier = ">=1.24"', 'specifier = "<3, >=1.24"')
    write(root / "uv.lock", lock)
    write(root / "pkg_a/pyproject.toml", (root / "pkg_a/pyproject.toml").read_text().replace('"numpy>=1.24"', '"numpy >=1.24,<3"'))
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def test_lock_refuses_uninitialized_submodule(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = super_repo(tmp_path)
    write(root / ".gitmodules", '[submodule "sub"]\n\tpath = sub\n[submodule "later"]\n\tpath = later\n[submodule "arena_isaac"]\n\tpath = arena_isaac\n')
    (root / "later").mkdir()
    code, _, err = run(capsys, root, "lock")
    assert code == 2
    assert "initialize later" in err


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv on PATH")
def test_lock_writes_manifests_and_derives_repo_locks_that_pass_check(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "ws"
    write(root / "pyproject.toml", SUPER_ROOT.replace('dependencies = ["pip"]', "dependencies = []").replace('dev = ["pytest"]', "dev = []"))
    make_package(root, "pkg_a")
    make_submodule(root, "single")
    make_package(root, "single/pkg_s")
    make_submodule(root, "multi")
    make_package(root, "multi/pkg_m")
    make_package(root, "multi/pkg_n")
    write(root / ".gitmodules", '[submodule "single"]\n\tpath = single\n[submodule "multi"]\n\tpath = multi\n')
    code, out, err = run(capsys, root, "lock")
    assert code == 0, err
    assert {"wrote pyproject.toml", "wrote multi/pyproject.toml", "wrote single/pkg_s/pyproject.toml"} <= set(out.splitlines())
    assert {"derived uv.lock", "derived multi/uv.lock", "derived single/pkg_s/uv.lock"} <= set(out.splitlines())
    assert not (root / "single/pyproject.toml").exists()
    assert tomllib.loads((root / "uv.lock").read_text())["manifest"]["members"] == ["arena-rosnav", "pkg-a"]
    assert tomllib.loads((root / "multi/uv.lock").read_text())["manifest"]["members"] == ["multi-workspace", "pkg-m", "pkg-n"]
    single_lock = tomllib.loads((root / "single/pkg_s/uv.lock").read_text())
    assert single_lock["requires-python"] == "==3.12.*"
    assert single_lock["options"]["exclude-newer"] == "2026-10-03T00:00:00Z"
    assert subprocess.run(["uv", "lock", "--check", "--offline", "--project", str(root / ".uv-workspace")], check=False).returncode == 0
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


def single_repo(root: pathlib.Path) -> pathlib.Path:
    """Superproject plus a submodule with one member, manifests and locks fresh."""
    write(root / "pyproject.toml", SUPER_ROOT)
    make_package(root, "pkg_a", pyproject=project_toml("pkg_a"))
    make_submodule(root, "solo")
    make_package(root, "solo/pkg_s", pyproject=project_toml("pkg_s", deps='"attrs"'))
    for home, text in uvw.expected_manifests(root, uvw.repo_members(root), uvw.load_root(root)).items():
        write(root / home / "pyproject.toml", text)
    fresh_lock(root, registry=[("pytest", "9.0.0")])
    fresh_lock(root, "solo/pkg_s", registry=[("attrs", "26.1.0")])
    return root


def test_lock_home_is_lone_member_of_single_member_submodule(tmp_path: pathlib.Path) -> None:
    assert uvw.lock_home("sub", ["sub/pkg"], tmp_path) == "sub/pkg"
    assert uvw.lock_home("sub", ["sub/a", "sub/b"], tmp_path) == "sub"
    assert uvw.lock_home(".", ["pkg"], tmp_path) == "."


def test_ament_python_lone_member_locks_at_a_generated_repo_root_without_requires_python(tmp_path: pathlib.Path) -> None:
    write(tmp_path / "pyproject.toml", SUPER_ROOT)
    make_submodule(tmp_path, "aud")
    pkg = make_package(tmp_path, "aud/aud", pyproject=project_toml("aud", deps='"attrs"'))
    write(pkg / "package.xml", (pkg / "package.xml").read_text().replace("</package>", "  <export><build_type>ament_python</build_type></export>\n</package>"))
    repos = uvw.repo_members(tmp_path)
    assert uvw.lock_home("aud", repos["aud"], tmp_path) == "aud"
    expected = tomllib.loads(uvw.expected_manifests(tmp_path, repos, uvw.load_root(tmp_path))["aud"])
    assert expected["project"]["requires-python"] == "==3.12.*"
    assert expected["tool"]["uv"]["workspace"]["members"] == ["aud"]
    assert "requires-python" not in tomllib.loads((pkg / "pyproject.toml").read_text())["project"]
    assert uvw.check_member("aud/aud", tmp_path) == []
    write(pkg / "pyproject.toml", project_toml("aud", deps='"attrs"', extra='requires-python = "==3.12.*"\n'))
    assert uvw.check_member("aud/aud", tmp_path) == ["aud/aud: [project].requires-python is not allowed, colcon cannot read an ament_python setup.py that carries it"]


def test_shared_lone_member_locks_at_a_generated_repo_root_without_requires_python(tmp_path: pathlib.Path) -> None:
    write(tmp_path / "pyproject.toml", SUPER_ROOT)
    make_submodule(tmp_path, "arena_planners")
    make_package(tmp_path, "arena_planners/arena_planners", pyproject=project_toml("arena_planners", virtual=False))
    repos = uvw.repo_members(tmp_path)
    assert uvw.lock_home("arena_planners", repos["arena_planners"], tmp_path) == "arena_planners"
    expected = tomllib.loads(uvw.expected_manifests(tmp_path, repos, uvw.load_root(tmp_path))["arena_planners"])
    assert expected["project"]["name"] == "arena-planners-workspace"
    assert expected["project"]["requires-python"] == "==3.12.*"
    assert expected["tool"]["uv"]["workspace"]["members"] == ["arena_planners"]
    assert "requires-python" not in tomllib.loads((tmp_path / "arena_planners/arena_planners/pyproject.toml").read_text())["project"]


def test_check_reports_requires_python_on_shared_member(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = clean_repo(tmp_path)
    make_package(root, "arena_planners/arena_planners", pyproject=project_toml("ap", extra='requires-python = "==3.12.*"\n', virtual=False))
    refresh(root)
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["arena_planners/arena_planners: [project].requires-python is not allowed, other Python environments install this member"]


def test_single_member_manifest_gains_root_settings_and_keeps_its_text() -> None:
    text = project_toml("pkg_s", deps='"attrs"') + '\n[tool.ruff]\nline-length = 100\n'
    updated = uvw.single_member_text(text, tomllib.loads(SUPER_ROOT))
    data = tomllib.loads(updated)
    assert data["project"]["requires-python"] == "==3.12.*"
    assert data["tool"]["uv"] == {"package": False, "exclude-newer": "2026-10-03T00:00:00Z"}
    assert data["tool"]["ruff"] == {"line-length": 100}
    assert uvw.single_member_text(updated, tomllib.loads(SUPER_ROOT)) == updated


def test_single_member_manifest_replaces_stale_settings_in_place() -> None:
    text = project_toml("pkg_s").replace('version = "0.1.0"\n', 'version = "0.1.0"\nrequires-python = ">=3.10"\n') + 'exclude-newer = "2020-01-01T00:00:00Z"\n'
    updated = uvw.single_member_text(text, tomllib.loads(SUPER_ROOT))
    assert updated == text.replace(">=3.10", "==3.12.*").replace("2020-01-01T00:00:00Z", "2026-10-03T00:00:00Z")


def test_table_value_lands_before_comment_heading_the_next_table() -> None:
    text = '[project]\nname = "a"\ndependencies = [\n    "x",\n]\n\n# sources below\n[tool.uv.sources]\nx = { path = "x" }\n'
    assert uvw.with_table_value(text, "project", "requires-python", '"==3.12.*"') == text.replace(']\n\n#', ']\nrequires-python = "==3.12.*"\n\n#')


def test_single_member_manifest_appends_missing_tool_uv_table() -> None:
    text = '[project]\nname = "real"\nversion = "0.1.0"\ndependencies = []\n'
    data = tomllib.loads(uvw.single_member_text(text, tomllib.loads(SUPER_ROOT)))
    assert data["tool"]["uv"] == {"exclude-newer": "2026-10-03T00:00:00Z"}


def test_lone_member_at_repo_root_keeps_its_manifest(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path / "pyproject.toml", SUPER_ROOT)
    make_package(tmp_path, "pkg_a", pyproject=project_toml("pkg_a"))
    make_submodule(tmp_path, "train")
    make_package(tmp_path, "train", pyproject=project_toml("train", deps='"attrs"', virtual=False))
    repos = uvw.repo_members(tmp_path)
    assert uvw.lock_home("train", repos["train"], tmp_path) == "train"
    expected = uvw.expected_manifests(tmp_path, repos, uvw.load_root(tmp_path))["train"]
    assert tomllib.loads(expected)["project"]["name"] == "train"
    assert tomllib.loads(expected)["project"]["requires-python"] == "==3.12.*"
    assert uvw.stray_files(tmp_path, "train", repos["train"]) == []


def test_compose_pins_from_single_member_lock(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = single_repo(tmp_path)
    code, _, _ = run(capsys, root, "compose")
    composed = tomllib.loads((root / ".uv-workspace/pyproject.toml").read_text())
    assert code == 0
    assert composed["tool"]["uv"]["constraint-dependencies"] == ["attrs==26.1.0", "pytest==9.0.0"]


def test_check_passes_for_fresh_single_member_submodule(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, single_repo(tmp_path), "check")
    assert (code, out) == (0, "")


def test_check_reports_single_member_without_root_settings(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = single_repo(tmp_path)
    write(root / "solo/pkg_s/pyproject.toml", project_toml("pkg_s", deps='"attrs"'))
    _, out, _ = run(capsys, root, "check")
    assert out.splitlines() == ["solo/pkg_s/pyproject.toml is out of date, run python3 _meta/tools/uv_workspace.py lock"]


def test_check_reports_stray_lock_and_generated_root_of_single_member_repo(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = single_repo(tmp_path)
    write(root / "solo/pyproject.toml", uvw.repo_manifest_text("solo", ["solo/pkg_s"], root, tomllib.loads(SUPER_ROOT)))
    write(root / "solo/uv.lock", "version = 1\n")
    _, out, _ = run(capsys, root, "check")
    assert sorted(out.splitlines()) == [
        "solo/pyproject.toml is stale, run python3 _meta/tools/uv_workspace.py lock",
        "solo/uv.lock is stale, run python3 _meta/tools/uv_workspace.py lock",
    ]


def test_check_keeps_hand_written_root_of_single_member_repo(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = single_repo(tmp_path)
    write(root / "solo/pyproject.toml", "[tool.ruff]\nline-length = 100\n")
    code, out, _ = run(capsys, root, "check")
    assert (code, out) == (0, "")


GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false"]
HOOKS = TOOL.parent / "githooks"


def git_run(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=cwd, capture_output=True, text=True, check=True).stdout


def git_repo(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    git_run(path, "init", "-q")
    return path


def hooked_tree(root: pathlib.Path) -> pathlib.Path:
    """Arena-like tree with the real tool copied in, a real superproject repo and a real nested repo hosting one member."""
    git_repo(root)
    write(root / "_meta/tools/uv_workspace.py", TOOL.read_text())
    shutil.copy2(HOOKS / "pre-commit", write(root / "_meta/tools/githooks/pre-commit"))
    write(root / "pyproject.toml", SUPER_ROOT.replace('dependencies = ["pip"]', "dependencies = []").replace('dev = ["pytest"]', "dev = []"))
    make_package(root, "pkg_a")
    git_repo(root / "sub")
    make_package(root, "sub/pkg_s")
    return root


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_compose_points_member_submodules_at_relock_hook(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = hooked_tree(tmp_path / "ws")
    git_repo(root / "other")
    git_run(root / "other", "config", "core.hooksPath", "/custom/hooks")
    make_package(root, "other/pkg_o")
    for home in (".", "sub/pkg_s", "other/pkg_o"):
        fresh_lock(root, home)
    code, out, err = run(capsys, root, "compose")
    assert code == 0
    assert "installed relock hook in sub" in out
    assert "other keeps core.hooksPath=/custom/hooks" in err
    assert git_run(root / "sub", "config", "--get", "core.hooksPath").strip() == "../_meta/tools/githooks"
    assert git_run(root / "other", "config", "--get", "core.hooksPath").strip() == "/custom/hooks"
    assert subprocess.run(["git", "config", "--get", "core.hooksPath"], cwd=root, check=False).returncode == 1
    _, out, _ = run(capsys, root, "compose")
    assert "installed relock hook" not in out
    git_run(root / "sub", "config", "core.hooksPath", "/host/path/_meta/tools/githooks")
    _, out, _ = run(capsys, root, "compose")
    assert "installed relock hook in sub" in out
    assert git_run(root / "sub", "config", "--get", "core.hooksPath").strip() == "../_meta/tools/githooks"


def test_locator_hook_runs_the_enclosing_checkouts_tool(tmp_path: pathlib.Path) -> None:
    write(tmp_path / "arena/_meta/tools/uv_workspace.py", "import sys\nprint(' '.join(sys.argv[1:]))\n")
    (tmp_path / "arena/a/b").mkdir(parents=True)
    result = subprocess.run([str(HOOKS / "pre-commit")], cwd=tmp_path / "arena/a/b", capture_output=True, text=True, check=True)
    assert result.stdout.strip() == f"--root {tmp_path / 'arena'} hook"


def test_locator_hook_without_tool_is_a_no_op(tmp_path: pathlib.Path) -> None:
    result = subprocess.run([str(HOOKS / "pre-commit")], cwd=tmp_path, capture_output=True, text=True, check=False)
    assert (result.returncode, result.stdout) == (0, "")


@pytest.mark.skipif(shutil.which("uv") is None or shutil.which("git") is None, reason="needs uv and git")
def test_submodule_commit_relocks_and_stages_its_lock(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = hooked_tree(tmp_path / "ws")
    assert run(capsys, root, "lock")[0] == 0
    git_run(root / "sub", "add", "-A")
    git_run(root / "sub", "commit", "-q", "-m", "init")
    assert "installed relock hook in sub" in run(capsys, root, "compose")[1]
    member = root / "sub/pkg_s/pyproject.toml"
    write(member, member.read_text().replace('version = "0.1.0"', 'version = "0.2.0"'))
    git_run(root / "sub", "add", "pkg_s/pyproject.toml")
    git_run(root / "sub", "commit", "-q", "-m", "bump")
    assert sorted(git_run(root / "sub", "show", "--name-only", "--format=", "HEAD").split()) == ["pkg_s/pyproject.toml", "pkg_s/uv.lock"]
    assert git_run(root / "sub", "status", "--porcelain") == ""
    assert run(capsys, root, "check")[:2] == (0, "")


@pytest.mark.skipif(shutil.which("uv") is None or shutil.which("git") is None, reason="needs uv and git")
def test_superproject_hook_stages_root_lock_and_ignores_unrelated_commits(tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = hooked_tree(tmp_path / "ws")
    assert run(capsys, root, "lock")[0] == 0
    write(root / "notes.txt", "x\n")
    git_run(root, "add", "notes.txt")
    hook = [sys.executable, str(TOOL), "--root", str(root), "hook"]
    assert subprocess.run(hook, cwd=root, capture_output=True, text=True, check=True).stdout == ""
    member = root / "pkg_a/pyproject.toml"
    write(member, member.read_text().replace('version = "0.1.0"', 'version = "0.3.0"'))
    git_run(root, "add", "pkg_a/pyproject.toml")
    result = subprocess.run(hook, cwd=root, capture_output=True, text=True, check=True)
    assert "uv-workspace: staged uv.lock" in result.stdout
    assert "uv.lock" in git_run(root, "diff", "--cached", "--name-only").split()


def test_hook_skips_relock_on_partial_tree(tmp_path: pathlib.Path) -> None:
    root = git_repo(tmp_path / "ws")
    write(root / "pyproject.toml", SUPER_ROOT)
    write(root / ".gitmodules", '[submodule "gone"]\n\tpath = gone\n')
    make_package(root, "pkg_a")
    git_run(root, "add", "pkg_a/pyproject.toml")
    result = subprocess.run([sys.executable, str(TOOL), "--root", str(root), "hook"], cwd=root, capture_output=True, text=True, check=True)
    assert "skipped relock (initialize gone)" in result.stderr
