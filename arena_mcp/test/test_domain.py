from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import Field

from arena_mcp.core.domain import Domain, tool_spec


def test_tool_names_carry_the_domain_prefix():
    domain = Domain("demo", "Demo domain.")
    with pytest.raises(ValueError, match="demo_<verb>"):

        @domain.tool()
        def other_verb() -> None:
            """Does nothing."""


def test_tools_need_a_docstring():
    domain = Domain("demo", "Demo domain.")
    with pytest.raises(ValueError, match="docstring"):

        @domain.tool()
        def demo_verb() -> None:
            pass


def test_annotations_follow_the_flags():
    domain = Domain("demo", "Demo domain.")

    @domain.tool(read_only=True)
    def demo_read() -> int:
        """Reads."""
        return 1

    @domain.tool(destructive=True, open_world=True)
    def demo_push() -> int:
        """Publishes."""
        return 2

    read, push = domain.tools
    assert read.annotations.read_only_hint and read.annotations.idempotent_hint and not read.annotations.destructive_hint
    assert push.annotations.destructive_hint and push.annotations.open_world_hint and not push.annotations.read_only_hint
    assert read.summary == "Reads."
    assert tool_spec(demo_push) is push


def test_resources_live_under_the_domain_uri():
    domain = Domain("demo", "Demo domain.")
    with pytest.raises(ValueError, match="arena://demo/"):
        domain.resource("arena://other/thing")


def test_unavailable_names_missing_modules_and_check_reasons():
    assert Domain("demo", "Demo.", requires=("json", "no_such_module_xyz.sub")).unavailable() == "missing no_such_module_xyz.sub"
    assert Domain("demo", "Demo.", requires=("json",)).unavailable() is None
    assert Domain("demo", "Demo.", check=lambda: "airchitect is down").unavailable() == "airchitect is down"


def test_domain_names_are_lowercase_identifiers():
    with pytest.raises(ValueError):
        Domain("Bad-Name", "Bad.")


def test_media_tools_return_image_content_through_the_server():
    import asyncio

    from mcp.server.mcpserver import MCPServer

    from arena_mcp.core.media import array_png

    domain = Domain("demo", "Demo domain.")

    @domain.tool(read_only=True)
    def demo_snapshot() -> list:
        """A summary and a picture."""
        return [{"pixels": 4}, array_png([[0, 255], [255, 0]])]

    server = MCPServer(name="test")
    domain.install(server)
    result = asyncio.run(server.call_tool("demo_snapshot", {}))
    kinds = [block.type for block in result.content]
    assert "image" in kinds and "text" in kinds


def test_only_dict_results_get_a_structured_schema():
    domain = Domain("demo", "Demo domain.")

    @domain.tool()
    def demo_dict() -> dict[str, object]:
        """A mapping."""
        return {}

    @domain.tool()
    def demo_list() -> list[dict[str, object]]:
        """A list."""
        return []

    @domain.tool()
    def demo_any() -> object:
        """Anything."""
        return 1

    assert [spec.structured for spec in domain.tools] == [True, False, False]


def test_list_results_go_out_as_one_json_block_and_keep_their_parameters():
    import asyncio
    import json

    from mcp.server.mcpserver import MCPServer

    domain = Domain("demo", "Demo domain.")

    @domain.tool(read_only=True)
    async def demo_rows(limit: Annotated[int, Field(description="how many rows")] = 2) -> list[dict[str, object]]:
        """Rows."""
        return [{"row": index} for index in range(limit)]

    server = MCPServer(name="test")
    domain.install(server)
    tools = asyncio.run(server.list_tools())
    schema = next(tool for tool in tools if tool.name == "demo_rows").input_schema
    assert schema["properties"]["limit"]["description"] == "how many rows"
    result = asyncio.run(server.call_tool("demo_rows", {"limit": 3}))
    assert len(result.content) == 1 and json.loads(result.content[0].text) == [{"row": 0}, {"row": 1}, {"row": 2}]
