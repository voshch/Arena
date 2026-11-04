from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import socket
import time
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("arena_mcp")

import yaml
from arena_mcp.core.errors import ToolError
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as SdkToolError

from arena_simulation_setup.mcp import airchitect
from arena_simulation_setup.metrics.nav_at_r import DEFAULT_RADIUS
from arena_simulation_setup.tree.World import WorldIdentifier, WorldView, parse_world

HOTEL_RUNS = Path("/home/arena/tmp/showcase/hotel/runs")

HOTEL_B_EDGES = [
    "edge_lobby_corridor",
    "edge_lounge_lobby",
    "edge_breakfast_lobby",
    "edge_pantry_breakfast",
    "edge_suite_a_corridor",
    "edge_suite_b_corridor",
    "edge_suite_c_corridor",
    "edge_suite_d_corridor",
    "edge_bath_a_suite_a",
    "edge_bath_b_suite_b",
    "edge_bath_c_suite_c",
    "edge_bath_d_suite_d",
    "edge_linen_corridor",
]
HOTEL_B_SHARED = [
    "door_edge_bath_a_suite_a",
    "door_edge_bath_b_suite_b",
    "door_edge_bath_c_suite_c",
    "door_edge_bath_d_suite_d",
    "door_edge_breakfast_lobby",
    "door_edge_lobby_corridor",
    "door_edge_lounge_lobby",
    "door_edge_pantry_breakfast",
    "door_edge_suite_a_corridor",
    "door_edge_suite_b_corridor",
    "door_edge_suite_c_corridor",
]
HOTEL_B_TRIES = [
    [*HOTEL_B_SHARED, "door_edge_linen_corridor", "door_auto_breakfast_room__corridor", "door_auto_breakfast_room__suite_a", "door_auto_breakfast_room__suite_d", "door_auto_suite_c__linen_store"],
    [*HOTEL_B_SHARED, "door_edge_linen_corridor", "door_auto_breakfast_room__corridor", "door_auto_breakfast_room__suite_a", "door_auto_breakfast_room__suite_d", "door_auto_corridor__lounge", "door_auto_suite_b__linen_store", "door_auto_suite_b__suite_bath_c", "door_auto_suite_c__linen_store"],
    [*HOTEL_B_SHARED, "door_edge_suite_d_corridor", "door_auto_breakfast_room__linen_store", "door_auto_suite_b__suite_d"],
    [*HOTEL_B_SHARED, "door_edge_suite_d_corridor", "door_auto_breakfast_room__linen_store"],
]


def _graph(edges):
    return {"nodes": [], "edges": [{"id": edge, "source": "a", "target": "b", "attributes": []} for edge in edges]}


def _record(graph, doors_by_level):
    return {"status": "completed", "plan": {"world": {"levels": {level: {"doors": [{"name": name, "span": [[0, 0], [1, 0]]} for name in doors]} for level, doors in doors_by_level.items()}}, "graph": graph}}


@pytest.fixture
def closed_url():
    previous = os.environ.get("AIRCHITECT_URL")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["AIRCHITECT_URL"] = f"http://127.0.0.1:{port}"
    yield os.environ["AIRCHITECT_URL"]
    if previous is None:
        del os.environ["AIRCHITECT_URL"]
    else:
        os.environ["AIRCHITECT_URL"] = previous


def _world_zip(source: Path, extra: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and "scenarios" not in path.relative_to(source).parts:
                archive.write(path, path.relative_to(source).as_posix())
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def real_world() -> Path:
    return Path(WorldIdentifier("map_empty").resolve_sync().path)


def test_floorplan_selection_keeps_the_hotel_b_try_the_old_driver_kept():
    graph = _graph(HOTEL_B_EDGES)
    best, scores = airchitect.pick_floorplan(graph, [_record(graph, {"0": doors}) for doors in HOTEL_B_TRIES])
    assert scores[best]["try"] == 4
    assert [(len(score["dropped_edges"]), len(score["auto_doors"])) for score in scores] == [(1, 4), (1, 7), (1, 2), (1, 1)]
    assert scores[best]["dropped_edges"] == ["edge_linen_corridor"]
    assert scores[best]["auto_doors"] == ["door_auto_breakfast_room__linen_store"]


def test_floorplan_selection_counts_doors_on_every_level_and_breaks_ties_by_order():
    graph = _graph(["e1", "e2", "e3"])
    split = _record(graph, {"0": ["door_e1"], "1": ["door_e2", "door_e3"]})
    assert airchitect.floorplan_score(graph, split) == {"dropped_edges": [], "auto_doors": []}
    first = _record(graph, {"0": ["door_e1", "door_auto_x__y"]})
    second = _record(graph, {"0": ["door_e1", "door_auto_y__z"]})
    best, scores = airchitect.pick_floorplan(graph, [first, second])
    assert best == 0
    assert scores[0]["dropped_edges"] == ["e2", "e3"]


@pytest.mark.skipif(not HOTEL_RUNS.is_dir(), reason="the recorded hotel runs live on the laptop only")
@pytest.mark.parametrize("run", ["hotel_a", "hotel_b"])
def test_floorplan_selection_matches_the_recorded_runs(run):
    folder = HOTEL_RUNS / run
    graph = json.loads((folder / "01_prompt_result.json").read_text())["graph"]
    tries = []
    for index in range(1, 5):
        old = json.loads((folder / f"02_floorplan_try{index}_result.json").read_text())
        levels = {level: [door["name"] for zone in data["zones"] for door in zone.get("doors") or []] for level, data in old["multi_level_world"]["levels"].items()}
        tries.append(_record(graph, levels))
    choice = json.loads((folder / "02_floorplan_choice.json").read_text())
    best, scores = airchitect.pick_floorplan(graph, tries)
    assert scores[best]["try"] == choice["chosen_try"]
    recorded = {entry["try"]: (entry["dropped"], entry["implicit"]) for entry in choice["all"]}
    assert {score["try"]: (len(score["dropped_edges"]), len(score["auto_doors"])) for score in scores} == recorded


def test_install_zip_unpacks_a_real_world_into_the_local_root(local_root, real_world):
    data = _world_zip(real_world, {"graph.yaml": b"nodes: []\nedges: []\n"})
    path = airchitect.install_zip(data, "from_zip", overwrite=False)
    assert path == local_root / "worlds" / "from_zip"
    assert (path / "world.yaml").read_bytes() == (real_world / "world.yaml").read_bytes()
    assert (path / "graph.yaml").is_file()
    assert WorldView(path).load(validate=True).levels
    assert [entry.name for entry in path.parent.iterdir()] == ["from_zip"]


def test_install_zip_refuses_an_existing_world_unless_overwrite(local_root, real_world):
    data = _world_zip(real_world)
    path = airchitect.install_zip(data, "twice", overwrite=False)
    (path / "stale.txt").write_text("left over")
    with pytest.raises(ToolError, match="already exists"):
        airchitect.install_zip(data, "twice", overwrite=False)
    assert (path / "stale.txt").is_file()
    path = airchitect.install_zip(data, "twice", overwrite=True)
    assert not (path / "stale.txt").exists()
    assert (path / "world.yaml").is_file()


@pytest.mark.parametrize("member", ["../evil.yaml", "/etc/evil.yaml", "a/../../evil.yaml"])
def test_install_zip_rejects_paths_that_leave_the_world(local_root, real_world, member):
    data = _world_zip(real_world, {member: b"x"})
    with pytest.raises(ToolError, match="unsafe path"):
        airchitect.install_zip(data, "unsafe", overwrite=False)
    assert not (local_root / "worlds" / "unsafe").exists()
    assert not (local_root / "evil.yaml").exists()


def test_install_zip_rejects_archives_without_a_root_world_yaml(local_root, real_world):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.write(real_world / "world.yaml", "nested/world.yaml")
    with pytest.raises(ToolError, match="no world.yaml"):
        airchitect.install_zip(buffer.getvalue(), "nested", overwrite=False)
    with pytest.raises(ToolError, match="not a zip"):
        airchitect.install_zip(b"<html>error page</html>", "nested", overwrite=False)
    assert not (local_root / "worlds" / "nested").exists()


def test_check_world_validates_measures_nav_at_r_and_writes_previews(tmp_path, real_world):
    world = tmp_path / "world"
    shutil.copytree(real_world, world, ignore_dangling_symlinks=True)
    report = airchitect.check_world(world, 0.3, tmp_path / "run")
    assert report["validation"]["valid"] is True and report["validation"]["warnings"] == []
    assert report["nav_at_r"]["radius"] == pytest.approx(0.3)
    levels = report["nav_at_r"]["levels"]
    assert levels and all(0.0 < level["value"] <= 1.0 for level in levels.values())
    assert report["previews"] and all(Path(preview).read_bytes().startswith(b"\x89PNG") for preview in report["previews"])
    assert any(path.name.startswith("nav_at_r") for path in (tmp_path / "run").iterdir())


def test_check_world_reports_a_world_that_fails_strict_parsing(tmp_path, real_world):
    world = tmp_path / "broken"
    shutil.copytree(real_world, world, ignore_dangling_symlinks=True)
    data = yaml.safe_load((world / "world.yaml").read_text())
    data["unexpected_key"] = 1
    (world / "world.yaml").write_text(yaml.safe_dump(data))
    report = airchitect.check_world(world, 0.3, tmp_path / "run")
    assert report["validation"]["valid"] is False
    assert "unexpected_key" in report["validation"]["reason"]


def test_check_world_measures_nav_at_r_at_the_metric_default_radius_when_none_is_given(tmp_path, real_world):
    world = tmp_path / "world"
    shutil.copytree(real_world, world, ignore_dangling_symlinks=True)
    report = airchitect.check_world(world, None, tmp_path / "run")
    assert report["nav_at_r"]["radius"] == DEFAULT_RADIUS
    assert report["nav_at_r"]["levels"]


def test_bundle_zips_a_world_the_ingest_endpoint_can_read(real_world):
    data = airchitect.bundle("map_empty")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        assert "world.yaml" in names
        parse_world(yaml.safe_load(archive.read("world.yaml")))
    assert all(not name.startswith(("/", "..")) for name in names)


def test_latest_plan_prefers_scenario_then_decoration_then_floorplan():
    plan = {"world": {"levels": {}}, "graph": {"nodes": [], "edges": []}}
    floor = {"status": "completed", "plan": {**plan, "tag": "floor"}}
    decoration = {"status": "completed", "plan": {**plan, "tag": "decoration"}}
    scenario = {"status": "completed", "plan": {**plan, "tag": "scenario"}, "scenario": {"robots": []}}
    assert airchitect.latest_plan({"floorplan": floor, "decoration": decoration, "scenario": scenario})[1:] == ({"robots": []}, "scenario")
    assert airchitect.latest_plan({"floorplan": floor, "decoration": decoration, "scenario": {"status": "error"}})[2] == "decoration"
    assert airchitect.latest_plan({"floorplan": floor, "decoration": None, "scenario": None})[0]["tag"] == "floor"
    with pytest.raises(ToolError, match="no completed floor plan"):
        airchitect.latest_plan({"floorplan": {"status": "processing"}, "decoration": None, "scenario": None})


def test_summarize_counts_a_stage_record_without_its_bulk():
    graph = _graph(HOTEL_B_EDGES)
    record = {**_record(graph, {"0": HOTEL_B_TRIES[3]}), "model": {"name": "Arena", "endpoint": "http://x"}, "timestamp": "t", "kg": {"text": "..."}}
    summary = airchitect.summarize(record)
    assert summary["model"] == "Arena" and summary["status"] == "completed" and summary["has_kg"]
    assert summary["graph"] == {"nodes": 0, "edges": len(HOTEL_B_EDGES)}
    assert summary["levels"]["0"]["doors"] == len(HOTEL_B_TRIES[3])
    assert airchitect.summarize(None) is None


def test_pick_models_takes_defaults_and_checks_overrides():
    listing = {stage: [{"name": f"{stage}_a"}, {"name": f"{stage}_b", "default": True}] for stage in (*airchitect.STAGES, "export")}
    assert airchitect.pick_models(listing, None)["floorplan"] == "floorplan_b"
    assert airchitect.pick_models(listing, {"decoration": "decoration_a"})["decoration"] == "decoration_a"
    with pytest.raises(ToolError, match="no model"):
        airchitect.pick_models(listing, {"prompt": "missing"})
    with pytest.raises(ToolError, match="no stage"):
        airchitect.pick_models(listing, {"export": "export_a"})


def test_run_directory_keeps_stage_records_and_reuses_only_completed_ones(tmp_path):
    previous = os.environ.get("ARENA_DATA_DIR")
    os.environ["ARENA_DATA_DIR"] = str(tmp_path)
    try:
        run = airchitect.Run("airchitect-test-0001")
        run.update(uuid="u1", jobs=["a"])
        run.update(jobs=["a", "b"])
        run.save("01_prompt", {"status": "completed", "graph": {}})
        run.save("03_decoration", {"status": "error", "error": "boom"})
    finally:
        if previous is None:
            del os.environ["ARENA_DATA_DIR"]
        else:
            os.environ["ARENA_DATA_DIR"] = previous
    assert run.dir == tmp_path / "mcp" / "airchitect" / "airchitect-test-0001"
    assert run.meta == {"uuid": "u1", "jobs": ["a", "b"]}
    assert run.load("01_prompt") == {"status": "completed", "graph": {}}
    assert run.load("03_decoration") is None
    assert run.load("04_scenario") is None


def test_an_unreachable_webapp_stays_served_and_fails_fast_with_the_deploy_hint(closed_url):
    assert airchitect.domain.unavailable() is None
    start = time.monotonic()
    with pytest.raises(ToolError) as caught:
        airchitect.call("GET", "models")
    assert time.monotonic() - start < 30.0
    assert "./airchitect deploy up --cpu" in (caught.value.hint or "")


def test_pick_models_requires_only_the_stages_that_run():
    listing = {stage: [{"name": f"{stage}_a"}] for stage in ("prompt", "floorplan", "decoration")}
    assert set(airchitect.pick_models(listing, None, required=("prompt", "floorplan", "decoration"))) == {"prompt", "floorplan", "decoration"}
    with pytest.raises(ToolError, match="no model for scenario"):
        airchitect.pick_models(listing, None)


def test_every_tool_installs_into_the_server():
    server = MCPServer("airchitect-test")
    airchitect.domain.install(server)
    tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}
    assert set(tools) == {spec.name for spec in airchitect.domain.tools}
    tries = tools["airchitect_generate"].input_schema["properties"]["floorplan_tries"]
    assert tries["minimum"] == 1 and tries["maximum"] == 10


def test_status_through_the_server_fails_fast_when_the_webapp_is_down(closed_url):
    server = MCPServer("airchitect-test")
    airchitect.domain.install(server)
    start = time.monotonic()
    with pytest.raises(SdkToolError, match="airchitect deploy up --cpu"):
        asyncio.run(server.call_tool("airchitect_status", {}))
    assert time.monotonic() - start < 30.0


def test_generate_checks_the_name_and_the_target_before_contacting_the_stack(local_root, real_world, closed_url):
    with pytest.raises(ToolError, match="one path component"):
        asyncio.run(airchitect.airchitect_generate("../escape", "a hotel", "residential"))
    with pytest.raises(ToolError, match="no domain"):
        asyncio.run(airchitect.airchitect_generate("hotel", "a hotel", "castle"))
    airchitect.install_zip(_world_zip(real_world), "taken", overwrite=False)
    with pytest.raises(ToolError, match="already exists"):
        asyncio.run(airchitect.airchitect_generate("taken", "a hotel", "residential"))
    with pytest.raises(ToolError, match="did not answer"):
        asyncio.run(airchitect.airchitect_generate("taken", "a hotel", "residential", overwrite=True))
