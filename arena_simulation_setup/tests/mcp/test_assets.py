from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import yaml

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.mcp import assets


sourced = pytest.mark.skipif(not os.environ.get("ARENA_WS_DIR"), reason="needs a sourced Arena workspace")


@sourced
def test_kinds_come_from_the_cli_table():
    kinds = assets.assets_kinds()
    assert {"object", "material", "world", "suite", "contest", "manifest"} <= set(kinds)
    assert kinds["world"] == "worlds"


@sourced
def test_ls_lists_installed_worlds_and_filters_by_glob():
    listed = assets.assets_ls("world", pattern="map_*")
    assert {"name": "map_empty"} in listed["assets"]
    assert all(row["name"].startswith("map_") for row in listed["assets"])
    assert listed["count"] == len(listed["assets"])


@sourced
def test_find_reports_the_winning_resolver_of_a_world():
    found = assets.assets_find("world", "sbend_corridor")
    assert found["resolves"]
    hit = next(verdict for verdict in found["verdicts"] if verdict["verdict"] == "hit")
    assert hit["path"] == str(ASS_DIR / "worlds" / "sbend_corridor")


@sourced
def test_find_of_an_unknown_kind_fails_with_the_cli_message():
    with pytest.raises(ToolError, match="unknown kind"):
        assets.assets_find("spaceship", "x")


@sourced
def test_pull_of_a_world_absent_from_the_bucket_fails():
    with pytest.raises(ToolError, match="was not pulled"):
        assets.assets_pull("world", "mcp_no_such_world_anywhere")


def test_domain_serves_only_the_arena_asset_verbs():
    assert sorted(spec.name for spec in assets.domain.tools) == ["assets_articulation", "assets_find", "assets_import", "assets_import_items", "assets_import_sources", "assets_kinds", "assets_ls", "assets_pull", "assets_push", "assets_retarget_animation"]


@sourced
def test_import_sources_list_the_datasets_by_kind():
    every = assets.assets_import_sources()
    assert ("world", "barn") in {(row["kind"], row["name"]) for row in every["sources"]}
    worlds = assets.assets_import_sources("world")
    assert worlds["count"] == len(worlds["sources"]) > 0
    assert {row["kind"] for row in worlds["sources"]} == {"world"}


@sourced
def test_import_items_lists_a_dataset_filtered_and_capped():
    listed = assets.assets_import_items("world", "barn", pattern="29?")
    assert listed["count"] == 10
    assert listed["items"] == [f"29{digit}" for digit in range(10)]
    assert listed["target"].endswith("worlds")
    assert assets.assets_import_items("world", "barn", limit=3)["items"] == ["0", "1", "2"]


@sourced
def test_import_of_an_unregistered_source_is_refused_before_a_job_starts():
    with pytest.raises(ToolError, match="no world import source 'nope'"):
        asyncio.run(assets.assets_import("world", "nope", ["1"]))


def test_smplx_importer_names_the_retarget_tool(tmp_path):
    from arena_simulation_setup.adapters.arena_animation_retarget.importer import SMPLX

    importer = SMPLX(cache=tmp_path)
    assert importer.items() == []
    with pytest.raises(RuntimeError, match="animation_retarget"):
        importer.import_item("wave", tmp_path)
    assert importer.import_items(["wave"], tmp_path) == {}


def test_retarget_of_a_missing_source_is_refused_before_a_job_starts(tmp_path):
    with pytest.raises(ToolError, match="no such source"):
        asyncio.run(assets.assets_retarget_animation([str(tmp_path / "nope.npz")], str(tmp_path)))


@pytest.fixture
def object_root(tmp_path):
    from arena_simulation_setup.tree import DynamicPaths

    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = tmp_path
    yield tmp_path
    DynamicPaths.ARENA.path = previous


def test_articulation_lists_links_joints_physics_materials_states_detents_families_and_swept_box(object_root):
    name = f"Kitchen/McpCabinet{uuid.uuid4().hex[:8]}"
    bundle = object_root / "Kitchen" / "Object" / name.split("/")[1]
    (bundle / "links" / "body").mkdir(parents=True)
    (bundle / "links" / "door").mkdir()
    articulation = {
        "links": {"body": ["body"], "door": ["door"], "knob": []},
        "joints": [
            {"name": "door", "type": "revolute", "parent": "body", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.57], "role": "door", "dynamics": {"friction": 0.4, "spring": {"stiffness": 2.0, "rest": 0.0}}},
            {"name": "knob", "type": "continuous", "parent": "door", "child": "knob", "velocity": 2.0},
        ],
        "materials": {"door": {"static_friction": 0.6}},
        "states": {"closed": {"door": 0.0}, "open": {"door": 1.5}},
        "detents": {"knob": {"up": 0.0, "down": 3.0}},
        "swept": [[-0.5, 0.5], [-1.3, 0.3], [0.0, 1.0]],
    }
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.5, 0.5], [-0.3, 0.3], [0.0, 1.0]], "articulation": articulation}))
    shown = assets.assets_articulation(name)
    assert shown["articulated"] and shown["path"] == str(bundle) and shown["root"] == "body"
    assert shown["links"] == [{"name": "body", "nodes": ["body"], "bundle": True}, {"name": "door", "nodes": ["door"], "bundle": True}, {"name": "knob", "nodes": [], "bundle": False}]
    door, knob = shown["joints"]
    assert (door["name"], door["type"], door["parent"], door["child"], door["limit"], door["speed"]) == ("door", "revolute", "body", "door", [0.0, 1.57], 1.0)
    assert (door["role"], door["dynamics"]) == ("door", {"friction": 0.4, "spring": {"stiffness": 2.0, "rest": 0.0}})
    assert (knob["type"], "limit" in knob, knob["speed"], "dynamics" in knob) == ("continuous", False, 2.0, False)
    assert shown["materials"] == {"door": {"static_friction": 0.6}}
    assert shown["states"] == {"closed": {"door": 0.0}, "open": {"door": 1.5}}
    assert shown["detents"] == {"knob": {"up": 0.0, "down": 3.0}}
    assert shown["families"] == {"openness": ["door"], "angle": ["knob"], "setting": ["knob"], "q": ["door", "knob"]}
    assert shown["initial"] == {"door": 0.0, "knob": 0.0}
    assert shown["swept"] == [[-0.5, 0.5], [-1.3, 0.3], [0.0, 1.0]]


def test_articulation_of_a_rigid_object_says_so_and_an_unknown_one_fails(object_root):
    name = f"Kitchen/McpCrate{uuid.uuid4().hex[:8]}"
    bundle = object_root / "Kitchen" / "Object" / name.split("/")[1]
    bundle.mkdir(parents=True)
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.4]]}))
    assert assets.assets_articulation(name) == {"object": name, "path": str(bundle), "articulated": False}
    with pytest.raises(ToolError, match="does not resolve"):
        assets.assets_articulation(f"Kitchen/McpGhost{uuid.uuid4().hex[:8]}")
