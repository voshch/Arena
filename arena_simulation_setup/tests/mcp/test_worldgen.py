from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("arena_mcp")

from arena_mcp.core.errors import ToolError
from arena_simulation_setup.mcp import worldgen
from arena_simulation_setup.tree.World import WorldView
from arena_simulation_setup.tree.World.Map import preview_resolution
from arena_simulation_setup.generative.world import WorldGenerator
from mcp.server.mcpserver import MCPServer

SKETCH = "\u250c\u2500\u252c\u2500\u2510\n\u2502 \u2502 \u2502\n\u2514\u2500\u2534\u2500\u2518"
BARN_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "barn"
SMALL_CONFIGS = {
    "empty": {"width": 6.0, "height": 4.0},
    "hallway": {},
    "barn": {},
    "sketch": {"sketch": SKETCH, "cell": 4.0},
    "letter": {"text": "AB"},
}


@pytest.fixture
def writable_root(tmp_path):
    previous = os.environ.get("ARENA_ASSETS_DIR_LOCAL")
    os.environ["ARENA_ASSETS_DIR_LOCAL"] = str(tmp_path)
    yield tmp_path
    if previous is None:
        del os.environ["ARENA_ASSETS_DIR_LOCAL"]
    else:
        os.environ["ARENA_ASSETS_DIR_LOCAL"] = previous


def _call(name, arguments):
    server = MCPServer("worldgen-test")
    worldgen.domain.install(server)
    return asyncio.run(server.call_tool(name, arguments))


def test_generators_lists_every_registered_generator_with_its_schema():
    listed = {entry["name"]: entry for entry in worldgen.worldgen_generators()}
    assert set(listed) == {kind.value for kind in WorldGenerator.available()}
    assert listed["barn"]["episode_binding"] == {"tm_robots": "scenario", "robots_params": {"file": "barn"}}
    assert listed["sketch"]["takes_sketch"] and not listed["empty"]["takes_sketch"]
    assert listed["sketch"]["binding_follows_pedestrians"] and not listed["barn"]["binding_follows_pedestrians"]
    assert "width" in listed["empty"]["config_schema"]["properties"]


@pytest.mark.parametrize("generator", sorted(SMALL_CONFIGS))
def test_preview_through_the_server_returns_diagnostics_and_a_map(generator):
    result = _call("worldgen_preview", {"generator": generator, "config": SMALL_CONFIGS[generator], "seed": 3})
    assert not result.is_error
    kinds = [block.type for block in result.content]
    assert kinds == ["text", "image"]
    report = json.loads(result.content[0].text)
    assert report["generator"] == generator
    assert report["diagnostics"]["components"] >= 1
    assert report["diagnostics"]["zones"] >= 1
    assert min(report["diagnostics"]["extent"]) > 0
    assert result.content[1].mime_type == "image/png"


def test_preview_reports_sketch_warnings_by_cell_and_the_normalized_sketch():
    report, _ = worldgen.worldgen_preview("sketch", sketch="+-+\n| |\n+-+")
    assert report["normalized"] == "\u253c\u2500\u253c\n\u2502 \u2502\n\u253c\u2500\u253c"
    assert report["grid"]["rows"] == 3 and report["grid"]["cols"] == 3
    one_sided, _ = worldgen.worldgen_preview("sketch", sketch="\u2500\u2502")
    assert any(note["row"] == 0 and note["col"] == 0 and "one side only" in note["text"] for note in one_sided["warnings"])


def test_preview_rejects_unknown_generators_fields_and_sketches_for_sketchless_generators():
    with pytest.raises(ToolError, match="no generator"):
        worldgen.worldgen_preview("nope")
    with pytest.raises(ToolError, match="no field"):
        worldgen.worldgen_preview("empty", config={"widht": 3.0})
    with pytest.raises(ToolError, match="takes no sketch"):
        worldgen.worldgen_preview("empty", sketch="\u253c")
    with pytest.raises(ToolError, match="invalid empty configuration"):
        worldgen.worldgen_preview("empty", config={"width": "wide"})


def test_barn_takes_a_challenge_index_and_source_through_the_server():
    from arena_simulation_setup.generative.world import WorldGeneratorType
    from arena_simulation_setup.generative.world.layout import diagnostics_of

    barn = next(entry for entry in worldgen.worldgen_generators() if entry["name"] == "barn")
    assert {"index", "source", "passage_width", "dead_ends"} <= set(barn["config_schema"]["properties"])
    config = {"index": 0, "source": str(BARN_FIXTURES)}
    result = _call("worldgen_preview", {"generator": "barn", "config": config, "seed": 3})
    assert not result.is_error, result.content[0].text
    report = json.loads(result.content[0].text)
    canonical = diagnostics_of(WorldGenerator(WorldGeneratorType.BARN, config).compute().all_levels[0])
    assert report["diagnostics"]["extent"] == [round(float(value), 3) for value in canonical.extent]
    assert report["diagnostics"]["extent"] != worldgen.worldgen_preview("barn", seed=3)[0]["diagnostics"]["extent"]
    assert report["episode_binding"] == {"tm_robots": "scenario", "robots_params": {"file": "barn"}}
    with pytest.raises(ToolError, match="invalid barn configuration"):
        worldgen.worldgen_preview("barn", config={"index": 300})


def test_preview_renders_at_the_shared_preview_resolution():
    report, _ = worldgen.worldgen_preview("empty", config={"width": 120.0, "height": 20.0})
    assert report["map"]["resolution"] == pytest.approx(preview_resolution(max(report["diagnostics"]["extent"])))
    assert report["map"]["resolution"] > 0.05


def test_preview_with_the_same_seed_is_deterministic():
    first, _ = worldgen.worldgen_preview("barn", seed=11)
    second, _ = worldgen.worldgen_preview("barn", seed=11)
    assert first["diagnostics"]["extent"] == second["diagnostics"]["extent"]
    assert first["diagnostics"]["zones"] == second["diagnostics"]["zones"]


@pytest.mark.parametrize("generator", sorted(SMALL_CONFIGS))
def test_save_writes_a_loadable_world_under_the_writable_root(writable_root, generator):
    saved = worldgen.worldgen_save(f"gen_{generator}", generator, config=SMALL_CONFIGS[generator], seed=5)
    path = Path(saved["path"])
    assert path == writable_root / "worlds" / f"gen_{generator}"
    world = WorldView(path).load(validate=True)
    assert len(world.levels) == 1
    assert (path / "world.yaml").is_file()
    assert saved["episode_binding"] == _binding(generator)


def _binding(generator):
    return next(entry["episode_binding"] for entry in worldgen.worldgen_generators() if entry["name"] == generator)


def test_save_of_barn_writes_its_scenario(writable_root):
    saved = worldgen.worldgen_save("gen_barn_scenario", "barn", seed=2)
    assert (Path(saved["path"]) / "scenarios" / "barn" / "scenario.yaml").is_file()


def test_save_refuses_an_existing_world_unless_overwrite_which_saves_in_place(writable_root):
    worldgen.worldgen_save("twice", "empty", config={"width": 5.0, "height": 5.0})
    with pytest.raises(ToolError, match="already exists"):
        worldgen.worldgen_save("twice", "empty")
    target = writable_root / "worlds" / "twice"
    (target / "scenarios" / "kept").mkdir(parents=True)
    (target / "scenarios" / "kept" / "scenario.yaml").write_text("format: arena-scenario/2\n")
    saved = worldgen.worldgen_save("twice", "empty", config={"width": 9.0, "height": 5.0}, overwrite=True)
    assert Path(saved["path"]) == target
    assert (target / "scenarios" / "kept" / "scenario.yaml").read_text() == "format: arena-scenario/2\n"
    assert saved["diagnostics"]["extent"][0] == pytest.approx(9.0)
    assert WorldView(target).load(validate=True).levels


@pytest.mark.parametrize("name", ["", ".", "..", ".hidden", "a/b", "../escape", "a\\b"])
def test_save_rejects_names_that_are_not_one_path_component(writable_root, name):
    with pytest.raises(ToolError, match="one path component"):
        worldgen.worldgen_save(name, "empty")
    assert not (writable_root / "worlds").exists() or not any((writable_root / "worlds").iterdir())


def test_alphabet_carries_glyphs_aliases_and_void_characters():
    table = worldgen.worldgen_alphabet()
    glyphs = {entry["glyph"]: entry["arms"] for entry in table["glyphs"]}
    assert glyphs["\u253c"] == {"N": "light", "E": "light", "S": "light", "W": "light"}
    assert glyphs["\u2588"] == dict.fromkeys(table["directions"], "full")
    aliases = {entry["glyph"]: entry for entry in table["aliases"]}
    assert aliases["+"]["canonical"] == "\u253c"
    assert aliases["#"]["canonical"] == "\u2588"
    assert table["void_chars"] == " ."
    assert table["weights"] == ["none", "light", "heavy", "double", "full"]


def test_normalize_rewrites_shorthand_and_reports_unknown_cells():
    out = worldgen.worldgen_normalize("!legend:\n!  R: {fill: true}\n+-R\n|?|")
    assert out["sketch"] == "!legend:\n!  R: {fill: true}\n\u253c\u2500R\n\u2502?\u2502"
    assert out["unknown"] == [{"row": 1, "col": 1, "char": "?"}]


QUEUE_PEER = r"""
import asyncio, json, threading
import rclpy
from rclpy.executors import SingleThreadedExecutor
from task_generator_msgs.srv import QueueEpisode
from arena_mcp.core import ROS
from arena_mcp.core.errors import ToolError
from arena_simulation_setup.mcp.worldgen import worldgen_queue

received = []


def serve(request, response):
    received.append(request)
    response.success = request.world != "refused_world"
    response.error_msg = "" if response.success else "not today"
    return response


async def main():
    own = await ROS.node()
    for _ in range(2):
        await asyncio.sleep(1.0)
        foreign = [name for name in (await ROS.graph())["nodes"] if name != own.get_fully_qualified_name()]
        if foreign:
            print(json.dumps({"busy": foreign}))
            return
    node = rclpy.create_node("task_generator_node", namespace="/worldgen_test/env_7")
    node.create_service(QueueEpisode, "/worldgen_test/env_7/task_generator_node/config/queue_episode", serve)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        for _ in range(50):
            if "/worldgen_test/env_7/task_generator_node" in (await ROS.graph())["nodes"]:
                break
            await asyncio.sleep(0.2)
        result = await worldgen_queue("gen_barn", env_id=7, binding={"tm_robots": "scenario", "robots_params": {"file": "barn"}, "obstacles_params": {"dynamic.n": [3, 3]}})
        errors = []
        for name, binding in (("refused_world", None), ("other", {"bogus": 1}), ("../up", None)):
            try:
                await worldgen_queue(name, env_id=7, binding=binding)
                errors.append(None)
            except ToolError as exc:
                errors.append(exc.message)
        request = received[0]
        print(json.dumps({
            "result": result,
            "errors": errors,
            "calls": len(received),
            "world": request.world,
            "action": request.action,
            "keep_modules": request.keep_modules,
            "tm_robots": request.tm_robots,
            "tm_obstacles": request.tm_obstacles,
            "robots_params": [[p.name, p.value.type, p.value.string_value] for p in request.robots_params],
            "obstacles_params": [[p.name, p.value.type, list(p.value.integer_array_value)] for p in request.obstacles_params],
        }))
    finally:
        executor.shutdown()
        node.destroy_node()
        await ROS.shutdown()


asyncio.run(main())
"""


def test_queue_sends_the_world_and_its_binding_to_the_env_queue_service(tmp_path):
    pytest.importorskip("rclpy")
    pytest.importorskip("task_generator_msgs")
    env = {**os.environ, "ROS_DOMAIN_ID": str(90 + os.getpid() % 10), "ROS_AUTOMATIC_DISCOVERY_RANGE": "LOCALHOST", "ARENA_DATA_DIR": str(tmp_path)}
    done = subprocess.run([sys.executable, "-c", QUEUE_PEER], capture_output=True, text=True, timeout=120, env=env, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    if "busy" in out:
        pytest.skip(f"ROS domain {env['ROS_DOMAIN_ID']} is in use by {out['busy']}")
    assert out["result"]["queued"] == "gen_barn"
    assert out["result"]["service"] == "/worldgen_test/env_7/task_generator_node/config/queue_episode"
    assert out["world"] == "gen_barn" and out["keep_modules"] and out["action"] == 0
    assert out["tm_robots"] == "scenario" and out["tm_obstacles"] == ""
    assert out["robots_params"] == [["file", 4, "barn"]]
    assert out["obstacles_params"] == [["dynamic.n", 7, [3, 3]]]
    refused, bogus, escape = out["errors"]
    assert "not today" in refused
    assert "bogus" in bogus
    assert "one path component" in escape
    assert out["calls"] == 2
