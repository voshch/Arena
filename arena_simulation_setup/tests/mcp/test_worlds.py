from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import sys
import uuid

import pytest
import yaml

pytest.importorskip("arena_mcp")

from arena_mcp.core import ToolError
from mcp.server.mcpserver import Image
from PIL import Image as PilImage

from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.mcp import worlds
from arena_simulation_setup.tree.World import FORMAT, LEGACY_FORMAT
from arena_simulation_setup.tree.World.Map import PREVIEW_MIN_RESOLUTION, PREVIEW_PIXEL_BUDGET
from arena_simulation_setup.tree.World.Scenario import LEGACY_SCENARIO_FORMAT

PNG = b"\x89PNG\r\n\x1a\n"

BOX = {"format": FORMAT, "levels": [{"id": "0", "zones": [{"name": "box", "polygon": [[0, 0], [6, 0], [6, 4], [0, 4]], "walls": "outline"}, {"name": "nook", "polygon": [[6, 0], [8, 0], [8, 4], [6, 4]]}]}]}


def _size(image: Image) -> tuple[int, int]:
    assert image.data is not None and image.data.startswith(PNG)
    return PilImage.open(io.BytesIO(image.data)).size


def _split(result: list) -> tuple[dict, list[Image]]:
    head, *images = result
    assert isinstance(head, dict)
    assert all(isinstance(image, Image) for image in images)
    return head, images


@pytest.mark.parametrize("name", ["", ".", "..", ".hidden", "-dash", "a/b", "../escape", "a\\b", "a\0b", "a,b[1]"])
def test_path_component_rejects_anything_but_one_plain_path_component(name):
    with pytest.raises(ToolError, match="one path component"):
        worlds.path_component(name)


def test_path_component_accepts_plain_names_and_names_what_it_checks():
    assert worlds.path_component("hotel_b-2.v1") == "hotel_b-2.v1"
    with pytest.raises(ToolError, match="scenario name"):
        worlds.path_component("a/b", "scenario name")


def test_inspect_reports_names_kg_scenarios_and_a_preview():
    summary, images = _split(worlds.worlds_inspect("map_empty"))
    assert summary["format"] == FORMAT and summary["valid"]
    (level,) = summary["levels"]
    assert level["zones"] == ["empty_zone"]
    assert sorted(level["doors"]) == ["north_door", "south_door", "west_door"]
    assert level["elevators"] == ["elevator_ground", "elevator_top"]
    assert level["objects"] == ["table_0"] and level["counts"]["objects"] == 1
    assert level["extent"]["size"] == [30.0, 23.0]
    assert summary["kg"]["derived_facts"] > 0
    assert "preflight" in summary["scenarios"]
    assert len(images) == 1 and len(summary["previews"]) == 1
    _size(images[0])


def test_inspect_with_a_level_filter_keeps_only_those_levels():
    summary, images = _split(worlds.worlds_inspect("three_storied_residential[2]"))
    assert [level["id"] for level in summary["levels"]] == ["2"]
    assert len(images) == 1
    with pytest.raises(ToolError, match="no level 9"):
        worlds.worlds_inspect("three_storied_residential[9]")


def test_inspect_of_an_unknown_world_fails_with_a_hint():
    with pytest.raises(ToolError) as raised:
        worlds.worlds_inspect("mcp_no_such_world_anywhere")
    assert "assets_ls" in (raised.value.hint or "") and "assets_find" in (raised.value.hint or "")


def test_inspect_of_an_invalid_world_carries_the_strict_reason(local_root):
    duplicate = {**BOX, "levels": [{**BOX["levels"][0], "doors": [{"name": "box", "span": [[6, 1], [6, 2]]}]}]}
    (local_root / "worlds" / "mcp_twin").mkdir(parents=True)
    (local_root / "worlds" / "mcp_twin" / "world.yaml").write_text(yaml.safe_dump(duplicate))
    summary, images = _split(worlds.worlds_inspect("mcp_twin"))
    assert summary["valid"] is False and "unique" in summary["reason"]
    assert [level["zones"] for level in summary["levels"]] == [["box", "nook"]]
    assert summary["kg"]["derived_facts"] > 0
    assert len(images) == 1


def test_preview_fits_a_large_level_into_the_pixel_budget(local_root):
    hall = {"format": FORMAT, "levels": [{"id": "0", "zones": [{"name": "hall", "polygon": [[0, 0], [200, 0], [200, 40], [0, 40]], "walls": "outline"}]}]}
    worlds.worlds_write("mcp_hall", yaml.safe_dump(hall))
    info, (image,) = _split(worlds.worlds_preview("mcp_hall"))
    assert info["resolution"] == pytest.approx(200 / PREVIEW_PIXEL_BUDGET)
    assert max(_size(image)) <= PREVIEW_PIXEL_BUDGET + 20
    small, _ = _split(worlds.worlds_preview("sbend_corridor"))
    assert small["resolution"] >= PREVIEW_MIN_RESOLUTION


def test_preview_lays_levels_side_by_side_or_draws_one():
    combined_info, (combined,) = _split(worlds.worlds_preview("three_storied_residential"))
    single_info, (single,) = _split(worlds.worlds_preview("three_storied_residential", level="1"))
    assert combined_info["level"] == "1,2,3" and single_info["level"] == "1"
    assert _size(combined)[0] * _size(combined)[1] > 2 * _size(single)[0] * _size(single)[1]
    with pytest.raises(ToolError, match="no level"):
        worlds.worlds_preview("three_storied_residential", level="0")


def test_validate_accepts_an_installed_world_and_reports_bad_text():
    assert worlds.worlds_validate(name="sbend_corridor")["valid"]
    duplicate = {**BOX, "levels": [{**BOX["levels"][0], "doors": [{"name": "box", "span": [[6, 1], [6, 2]]}]}]}
    verdict = worlds.worlds_validate(yaml_text=yaml.safe_dump(duplicate))
    assert not verdict["valid"] and "unique" in verdict["reason"]
    unknown = worlds.worlds_validate(yaml_text=yaml.safe_dump({**BOX, "rooms": []}))
    assert not unknown["valid"] and "unknown keys" in unknown["reason"]
    with pytest.raises(ToolError, match="exactly one"):
        worlds.worlds_validate(name="map_empty", yaml_text="{}")


def _legacy_world(root):
    (root / "0").mkdir(parents=True)
    shutil.copy(ASS_DIR / "worlds" / "hospital_1_bare" / "0" / "world.yaml", root / "0" / "world.yaml")
    return root


def test_validate_reads_a_format_1_directory_through_the_legacy_loader(tmp_path):
    root = _legacy_world(tmp_path / "bare")
    verdict = worlds.worlds_validate(path=str(root))
    assert verdict["format"] == LEGACY_FORMAT
    assert verdict["levels"] == ["0"]


def test_migrate_rewrites_a_format_1_world_directory_in_place(tmp_path):
    root = _legacy_world(tmp_path / "mcp_bare")
    done = worlds.worlds_migrate(str(root))
    assert done["migrated"] and done["path"] == str(root)
    assert yaml.safe_load((root / "world.yaml").read_text())["format"] == FORMAT
    assert not (root / "0" / "world.yaml").exists()
    assert not worlds.worlds_migrate(str(root))["migrated"]


def test_migrate_takes_only_what_touch_world_takes(local_root):
    root = _legacy_world(local_root / "worlds" / "mcp_resolved_only")
    with pytest.raises(ToolError, match="not found"):
        worlds.worlds_migrate("mcp_resolved_only")
    assert (root / "0" / "world.yaml").is_file()


def test_write_saves_into_the_local_root_and_refuses_to_clobber(local_root):
    written = worlds.worlds_write("mcp_box", yaml.safe_dump(BOX))
    target = local_root / "worlds" / "mcp_box"
    assert written["path"] == str(target) and written["resolves_to"] == str(target) and not written["shadowed"]
    assert (target / "map" / "0.png").is_file()
    with pytest.raises(ToolError, match="already exists"):
        worlds.worlds_write("mcp_box", yaml.safe_dump(BOX))
    assert worlds.worlds_write("mcp_box", yaml.safe_dump(BOX), overwrite=True)["replaced"]
    before = (target / "world.yaml").read_text()
    broken = {**BOX, "levels": [{**BOX["levels"][0], "zones": [{"name": "box", "walls": {"outline": True, "open_to": ["ghost"]}}]}]}
    with pytest.raises(ToolError, match="nothing was written"):
        worlds.worlds_write("mcp_box", yaml.safe_dump(broken), overwrite=True)
    assert (target / "world.yaml").read_text() == before


def test_overwrite_drops_the_maps_of_levels_that_are_gone(local_root):
    worlds.worlds_write("mcp_relevel", yaml.safe_dump(BOX))
    target = local_root / "worlds" / "mcp_relevel"
    renamed = {**BOX, "levels": [{**BOX["levels"][0], "id": "ground"}]}
    worlds.worlds_write("mcp_relevel", yaml.safe_dump(renamed), overwrite=True)
    assert sorted(path.name for path in (target / "map").iterdir() if path.suffix == ".png") == ["ground.png"]


def test_write_rejects_names_and_texts_it_cannot_place(local_root):
    with pytest.raises(ToolError, match="path component"):
        worlds.worlds_write("../escape", yaml.safe_dump(BOX))
    with pytest.raises(ToolError, match="format 1"):
        worlds.worlds_write("mcp_old", yaml.safe_dump({"zones": []}))
    with pytest.raises(ToolError, match="nothing was written"):
        worlds.worlds_write("mcp_bad", yaml.safe_dump({**BOX, "levels": [{"id": "0", "zones": [{"polygon": []}]}]}))
    assert not (local_root / "worlds" / "mcp_bad").exists()


def test_nav_at_r_reports_each_level_with_its_component_map():
    head, images = _split(worlds.worlds_nav_at_r("map_empty", radius=0.3, walls_only=True, start=[5.0, 12.0]))
    (level,) = head["levels"]
    assert level["level"] == "0" and level["radius"] == 0.3 and level["walls_only"]
    assert 0.0 < level["value"] <= 1.0 and level["start_share"] is not None
    assert len(images) == 1
    _size(images[0])
    with pytest.raises(ToolError, match="positive length"):
        worlds.worlds_nav_at_r("map_empty", radius=-1.0)


def test_nav_at_r_paints_one_map_per_level():
    head, images = _split(worlds.worlds_nav_at_r("three_storied_residential", walls_only=True))
    assert [level["level"] for level in head["levels"]] == ["1", "2", "3"]
    assert len(images) == 3


def test_kg_carries_zone_descriptions_and_derived_facts():
    derived = worlds.worlds_kg("sbend_corridor")
    stated = worlds.worlds_kg("sbend_corridor", derive="off")
    assert derived["zones"]["corridor_south"]["description"] == "Southern corridor, traversed westbound"
    assert derived["summary"]["derived_facts"] > 0 and stated["summary"]["derived_facts"] == 0
    assert len(derived["kg"]["facts"]) > len(stated["kg"]["facts"])
    with pytest.raises(ToolError, match="derive"):
        worlds.worlds_kg("sbend_corridor", derive="sometimes")


def test_kg_check_reports_entities_the_world_lacks(local_root):
    worlds.worlds_write("mcp_kg", yaml.safe_dump(BOX))
    assert worlds.worlds_kg_check("mcp_kg") == {"world": "mcp_kg", "kg": None, "problems": []}
    kg = {"format": "arena-kg/1", "uses": ["core@2"], "entities": [{"id": "room:box"}, {"id": "room:attic"}], "facts": [], "constraints": []}
    (local_root / "worlds" / "mcp_kg" / "kg.yaml").write_text(yaml.safe_dump(kg))
    problems = worlds.worlds_kg_check("mcp_kg")["problems"]
    assert len(problems) == 1 and "attic" in problems[0]
    assert worlds.worlds_kg_check("sbend_corridor")["problems"] == []


def test_vocabularies_list_the_packages_and_build_a_schema():
    packages = {package["package"]: package for package in worlds.worlds_vocabularies()["packages"]}
    assert "core@2" in packages and "room" in packages["core@2"]["nodes"]
    union = worlds.worlds_vocabularies(uses=["core@2"])
    assert "contains" in union["verbs"] and union["schema"]["properties"]["format"]["const"] == "arena-kg/1"
    with pytest.raises(ToolError, match="unknown package"):
        worlds.worlds_vocabularies(uses=["nope@9"])


def test_scenarios_list_a_scenario_without_format_as_format_1(local_root):
    worlds.worlds_write("mcp_old_scene", yaml.safe_dump(BOX))
    (local_root / "worlds" / "mcp_old_scene" / "scenarios" / "old").mkdir(parents=True)
    (local_root / "worlds" / "mcp_old_scene" / "scenarios" / "old" / "scenario.yaml").write_text(yaml.safe_dump({"robots": []}))
    assert [row["format"] for row in worlds.worlds_scenarios("mcp_old_scene")] == [LEGACY_SCENARIO_FORMAT]


def test_scenarios_and_one_scenario_with_its_instruction():
    listed = {row["scenario"]: row for row in worlds.worlds_scenarios("map_empty")}
    assert listed["preflight"]["format"] == "arena-scenario/2"
    shown = worlds.worlds_scenario("map_empty", "preflight")
    assert shown["valid"] and shown["counts"]["robots"] == 1 and shown["counts"]["humans"] == 2
    assert shown["instructions"][0].startswith("Go to")
    assert shown["content"]["format"] == "arena-scenario/2"
    with pytest.raises(ToolError, match="worlds_scenarios"):
        worlds.worlds_scenario("map_empty", "mcp_missing")


def _scenario(start: object) -> str:
    return yaml.safe_dump({"format": "arena-scenario/2", "robots": [{"start": start, "phases": [{"goto": "corridor_north"}]}]})


def test_scenario_validate_resolves_zone_references_against_the_world():
    good = worlds.worlds_scenario_validate(_scenario("corridor_south"), world="sbend_corridor")
    assert good["valid"] and good["instructions"] == ["Go to Northern corridor, traversed westbound to the goal."]
    bad = worlds.worlds_scenario_validate(_scenario("attic"), world="sbend_corridor")
    assert not bad["valid"] and "attic" in bad["reason"]
    alone = worlds.worlds_scenario_validate(_scenario("corridor_south"))
    assert not alone["valid"] and "world is given" in alone["reason"]
    legacy = worlds.worlds_scenario_validate(yaml.safe_dump({"robots": [{"start": [1.0, 1.0, 0.0], "goal": [5.0, 1.5, 0.0]}]}), world="sbend_corridor")
    assert legacy["valid"] and legacy["upgraded"]["robots"][0]["phases"] == [{"goto": [5.0, 1.5, 0.0]}]


def test_scenario_write_lands_in_a_writable_world_only(local_root):
    worlds.worlds_write("mcp_scene", yaml.safe_dump(BOX))
    text = yaml.safe_dump({"format": "arena-scenario/2", "robots": [{"start": "box", "phases": [{"goto": "nook"}]}]})
    written = worlds.worlds_scenario_write("mcp_scene", "visit", text)
    assert written["path"] == str(local_root / "worlds" / "mcp_scene" / "scenarios" / "visit" / "scenario.yaml")
    assert [row["scenario"] for row in worlds.worlds_scenarios("mcp_scene")] == ["visit"]
    with pytest.raises(ToolError, match="already exists"):
        worlds.worlds_scenario_write("mcp_scene", "visit", text)
    assert worlds.worlds_scenario_write("mcp_scene", "visit", text, overwrite=True)["replaced"]
    with pytest.raises(ToolError, match="outside the writable root"):
        worlds.worlds_scenario_write("map_empty", "visit", text)
    with pytest.raises(ToolError, match="nothing was written"):
        worlds.worlds_scenario_write("mcp_scene", "lost", text.replace("nook", "cellar").replace("box", "attic"))


def test_task_parse_returns_canonical_form_and_text_or_the_error():
    task = {"phases": [{"goto": "kitchen", "until": "robot in kitchen for 2.0"}, {"loop": [{"goto": [1, 2, 0]}], "until": "alice plays wave"}], "conditions": [{"op": "never", "p": "robot in lab"}]}
    parsed = worlds.worlds_task_parse(yaml.safe_dump(task))
    assert parsed["valid"] and parsed["kind"] is None
    assert [phase["type"] for phase in parsed["phases"]] == ["GoToPhase", "LoopPhase"]
    assert parsed["phases"][0]["until"] == "robot in kitchen for 2"
    assert parsed["conditions"] == ["Never let it happen that you are in lab."]
    assert parsed["text"].startswith("Go to kitchen")
    broken = worlds.worlds_task_parse(yaml.safe_dump({"phases": [{"goto": "kitchen", "until": "robot hovers"}]}))
    assert not broken["valid"] and "surface form" in broken["reason"]


def test_condition_parse_handles_until_and_episode_conditions():
    until = worlds.worlds_condition_parse("robot in kitchen and robot sees alice within 3 for 2 or not alice in hall")
    assert until["valid"] and until["kind"] == "until"
    assert until["node"]["form"] == "any"
    first, second = until["node"]["nodes"]
    assert first["form"] == "hold" and first["seconds"] == 2.0
    assert [atom["form"] for atom in first["node"]["nodes"]] == ["membership", "sees"]
    assert second["form"] == "not" and second["node"]["zone"] == "hall"
    assert until["canonical"] == "robot in kitchen and robot sees alice within 3 for 2 or not alice in hall"
    condition = worlds.worlds_condition_parse("{op: before, p: robot in hall, q: robot in lab}")
    assert condition["valid"] and condition["kind"] == "condition" and condition["condition"]["q"] == "robot in lab"
    assert not worlds.worlds_condition_parse("robot dances")["valid"]
    assert not worlds.worlds_condition_parse("{op: before, p: robot in hall}")["valid"]


def test_condition_parse_reads_structured_conditions():
    parsed = worlds.worlds_condition_parse("{all: [{field: wardrobe.openness.door_left, ge: 0.8}, {some: wardrobe.openness, within: [0.2, 0.4]}], for: 2}")
    assert parsed["valid"] and parsed["kind"] == "until"
    assert parsed["node"]["form"] == "hold" and parsed["node"]["seconds"] == 2.0
    field, family = parsed["node"]["node"]["nodes"]
    assert field == {"form": "field", "entity": "wardrobe", "field": "openness.door_left", "op": "ge", "value": 0.8}
    assert family == {"form": "family", "entity": "wardrobe", "family": "openness", "quantifier": "some", "op": "within", "value": [0.2, 0.4]}
    assert parsed["canonical"] == "wardrobe openness.door_left >= 0.8 and some wardrobe openness within [0.2, 0.4] for 2 s"
    json.dumps(parsed)
    broken = worlds.worlds_condition_parse("{field: wardrobe.state, lt: open}")
    assert not broken["valid"] and "'lt' takes a number" in broken["reason"]


@pytest.fixture
def object_root(local_root):
    from arena_simulation_setup.tree import DynamicPaths

    previous = DynamicPaths.ARENA.path
    DynamicPaths.ARENA.path = local_root
    yield local_root
    DynamicPaths.ARENA.path = previous


def _cabinet(root) -> str:
    name = f"Kitchen/McpCabinet{uuid.uuid4().hex[:8]}"
    bundle = root / "Kitchen" / "Object" / name.split("/")[1]
    bundle.mkdir(parents=True)
    articulation = {
        "links": {"body": ["body"], "door": ["door"]},
        "joints": [{"name": "door", "type": "revolute", "parent": "body", "child": "door", "origin": [0.5, -0.3, 0.0], "axis": [0, 0, 1], "limit": [0.0, 1.57]}],
        "states": {"closed": {"door": 0.0}, "open": {"door": 1.5}},
    }
    (bundle / "annotation.yaml").write_text(yaml.safe_dump({"bounding_box": [[-0.5, 0.5], [-0.3, 0.3], [0.0, 1.0]], "articulation": articulation}))
    return name


def _cabinet_scenario(model: str, until: str | dict, condition: dict, timeline_set: dict) -> str:
    return yaml.safe_dump(
        {
            "format": "arena-scenario/2",
            "objects": [{"name": "cabinet", "model": model, "pose": [1.0, 1.0, 0.0]}],
            "robots": [{"start": "box", "phases": [{"goto": "nook", "until": until}]}],
            "conditions": [condition],
            "timeline": [{"at": 1.0, "set": [timeline_set]}],
        }
    )


def test_scenario_validate_checks_conditions_and_timeline_against_articulated_objects(object_root):
    worlds.worlds_write("mcp_scene", yaml.safe_dump(BOX))
    model = _cabinet(object_root)
    good = worlds.worlds_scenario_validate(
        _cabinet_scenario(model, {"field": "cabinet.openness.door", "ge": 0.8}, {"op": "eventually", "p": {"some": "cabinet.openness", "ge": 0.5}}, {"entity": "cabinet", "field": "openness.door", "value": 0.5}),
        world="mcp_scene",
    )
    assert good["valid"], good
    bad = worlds.worlds_scenario_validate(
        _cabinet_scenario(model, {"field": "cabinet.openness.drawer", "ge": 0.8}, {"op": "eventually", "p": {"some": "cabinet.setting", "eq": "off"}}, {"entity": "cabinet", "field": "pos.door", "value": 0.5}),
        world="mcp_scene",
    )
    assert not bad["valid"]
    assert "robots[0].phases[0].until: cabinet has no field 'openness.drawer'" in bad["reason"]
    assert "conditions[0].p: cabinet has no 'setting' fields" in bad["reason"]
    assert "timeline[0].set[0]: " in bad["reason"]
    other = worlds.worlds_scenario_validate(
        _cabinet_scenario(model, "robot in nook", {"op": "eventually", "p": {"field": "door.openness.left", "ge": 0.5}}, {"entity": "door", "field": "locked", "value": True}),
        world="mcp_scene",
    )
    assert other["valid"], other


def test_scenario_validate_checks_joints_when_triggers_and_world_objects(object_root):
    model = _cabinet(object_root)
    level = {**BOX["levels"][0], "objects": [{"name": "wardrobe", "model": model, "pose": [2.0, 2.0, 0.0], "joints": {"openness": {"lid": 1.0}}}]}
    worlds.worlds_write("mcp_wardrobe", yaml.safe_dump({**BOX, "levels": [level]}))
    scenario = {
        "format": "arena-scenario/2",
        "objects": [{"name": "cabinet", "model": model, "pose": [1.0, 1.0, 0.0], "joints": {"setting": {"door": "high"}}}],
        "robots": [{"start": "box", "phases": [{"goto": "nook", "until": {"field": "wardrobe.openness.door", "ge": 0.8}}]}],
        "timeline": [{"when": {"each": "wardrobe.pos", "ge": 0.5}, "set": [{"entity": "wardrobe", "field": "openness", "value": 1}]}],
    }
    bad = worlds.worlds_scenario_validate(yaml.safe_dump(scenario), world="mcp_wardrobe")
    assert not bad["valid"]
    assert "objects[0] 'cabinet' joints: " in bad["reason"]
    assert "world objects[0] 'wardrobe' joints: " in bad["reason"]
    assert "timeline[0].when: wardrobe has no 'pos' fields" in bad["reason"]
    assert "robots[0].phases[0].until" not in bad["reason"]
    scenario["objects"][0]["joints"] = 0.5
    scenario["timeline"][0]["when"] = {"each": "wardrobe.openness", "ge": 0.5}
    level["objects"][0]["joints"] = "open"
    worlds.worlds_write("mcp_wardrobe_open", yaml.safe_dump({**BOX, "levels": [level]}))
    good = worlds.worlds_scenario_validate(yaml.safe_dump(scenario), world="mcp_wardrobe_open")
    assert good["valid"], good


def test_world_resource_is_the_inspect_summary_without_images():
    summary = worlds.world_resource("sbend_corridor")
    head, _ = _split(worlds.worlds_inspect("sbend_corridor"))
    assert summary["levels"] == head["levels"] and "previews" not in summary
    json.dumps(summary)


def test_server_serves_the_worlds_domain_with_images(tmp_path):
    from mcp import ClientSession, StdioServerParameters, stdio_client

    async def session():
        params = StdioServerParameters(command=sys.executable, args=["-m", "arena_mcp", "serve", "worlds"], env={**os.environ, "ARENA_DATA_DIR": str(tmp_path)})
        async with stdio_client(params) as (read, write), ClientSession(read, write) as client:
            await client.initialize()
            tools = {tool.name for tool in (await client.list_tools()).tools}
            inspected = await client.call_tool("worlds_inspect", {"name": "sbend_corridor"})
            missing = await client.call_tool("worlds_inspect", {"name": "mcp_no_such_world_anywhere"})
            resource = await client.read_resource("arena://worlds/sbend_corridor")
            return tools, inspected, missing, resource

    tools, inspected, missing, resource = asyncio.run(session())
    assert {"worlds_inspect", "worlds_validate", "worlds_write", "worlds_nav_at_r", "worlds_condition_parse"} <= tools
    assert not {"worlds_list", "worlds_find"} & tools
    assert not inspected.is_error
    assert [block.type for block in inspected.content] == ["text", "image"]
    assert json.loads(inspected.content[0].text)["levels"][0]["id"] == "0"
    assert missing.is_error and "hint:" in missing.content[0].text
    assert json.loads(resource.contents[0].text)["world"] == "sbend_corridor"
