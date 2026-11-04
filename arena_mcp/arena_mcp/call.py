"""One tool call from a terminal, with the jobs it starts followed in the foreground."""

from __future__ import annotations

import asyncio
import base64
import difflib
import json
import logging
import mimetypes
import os
import signal
import socket
import sys
import time
from collections.abc import Iterable
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, TextIO

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import AudioContent, CallToolResult, ImageContent, InputRequiredResult, TextContent, Tool

from . import GROUP
from .core.jobs import JOBS, SUCCEEDED, TERMINAL, Job, data_dir
from .core.ros import ROS
from .server import build, discover

POLL_S = 0.2
PROGRESS_S = 1.0
INTERRUPTED = 130

USAGE = """\
Usage: arena mcp call TOOL [KEY=VALUE ...]

Run one MCP tool and print its result on stdout.
  arena mcp ls                    the domains
  arena mcp ls DOMAIN             a domain's tools
  arena mcp call TOOL --help      a tool's parameters

A string parameter takes VALUE as is, any other parses it as JSON (poses='[[1, 2, 0]]'), and null clears an optional one.
Image and audio results are saved under $ARENA_DATA_DIR/mcp/<hostname>/media, their paths printed.
A job the tool starts runs in the foreground: its log goes to stderr, its final status to stdout, and Ctrl-C cancels it.
Exit code 0 on success, 1 on a tool error or a job that did not succeed, 2 on bad usage, 130 after Ctrl-C."""


class UsageError(Exception):
    """A call that never reached the tool, its message says what exists."""


def _domain_of(tool: str) -> str | None:
    names = [point.name for point in entry_points(group=GROUP) if tool.startswith(f"{point.name}_")]
    return max(names, key=len, default=None)


def _unknown_tool(name: str) -> str:
    loaded = discover()
    domains = {entry.name: entry.domain for entry in loaded}
    named = domains.get(name)
    if named is not None:
        tools = "\n".join(f"  {spec.name}  {spec.summary}" for spec in named.tools)
        return f"'{name}' is a domain, its tools:\n{tools}"
    tools = [spec.name for entry in loaded if entry.domain is not None for spec in entry.domain.tools]
    close = difflib.get_close_matches(name, tools, n=5, cutoff=0.5)
    nearest = f", closest: {', '.join(close)}" if close else ""
    return f"no tool '{name}'{nearest}\nDomains: {', '.join(sorted(domains))}. 'arena mcp ls DOMAIN' lists a domain's tools."


async def _load(name: str) -> tuple[MCPServer, Tool]:
    domain = _domain_of(name)
    if domain is None:
        raise UsageError(_unknown_tool(name))
    server, loaded = build([domain])
    if not loaded[0].served:
        raise UsageError(f"domain {domain} is unavailable: {loaded[0].reason}")
    for tool in await server.list_tools():
        if tool.name == name:
            return server, tool
    raise UsageError(_unknown_tool(name))


def _kinds(schema: dict[str, Any]) -> set[str]:
    kinds: set[str] = set()
    for option in schema.get("anyOf", ()):
        kinds |= _kinds(option)
    kind = schema.get("type")
    kinds.update([kind] if isinstance(kind, str) else kind or ())
    return kinds


def _value(raw: str, schema: dict[str, Any]) -> object:
    kinds = _kinds(schema)
    if raw == "null" and "null" in kinds:
        return None
    if kinds and kinds <= {"string", "null"}:
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _type_text(schema: dict[str, Any]) -> str:
    if "anyOf" in schema:
        return " | ".join(_type_text(option) for option in schema["anyOf"])
    if "enum" in schema:
        return "|".join(str(value) for value in schema["enum"])
    if "$ref" in schema:
        return str(schema["$ref"]).rsplit("/", 1)[-1]
    kind = schema.get("type", "any")
    if kind == "array" and "items" in schema:
        return f"array of {_type_text(schema['items'])}"
    return str(kind)


def _parameter_lines(tool: Tool) -> list[str]:
    schema = tool.input_schema
    required = schema.get("required", ())
    properties = schema.get("properties", {})
    width = max((len(key) for key in properties), default=0)
    lines = []
    for key, field in properties.items():
        needs = "required" if key in required else f"default {json.dumps(field.get('default'))}"
        about = f"  {field['description']}" if "description" in field else ""
        lines.append(f"  {key:<{width}}  {_type_text(field)}, {needs}{about}")
    return lines


def _tool_help(tool: Tool) -> str:
    lines = _parameter_lines(tool)
    parameters = "\n".join(["Parameters:", *lines]) if lines else "No parameters."
    return f"Usage: arena mcp call {tool.name} {'[KEY=VALUE ...]' if lines else ''}".rstrip() + f"\n\n{tool.description}\n\n{parameters}"


def _arguments(tool: Tool, tokens: Iterable[str]) -> dict[str, object]:
    properties = tool.input_schema.get("properties", {})
    arguments: dict[str, object] = {}
    for token in tokens:
        key, equals, raw = token.partition("=")
        if not equals or not key:
            raise UsageError(f"'{token}' is not KEY=VALUE\n\n{_tool_help(tool)}")
        if key not in properties:
            raise UsageError(f"{tool.name} has no parameter '{key}'\n\n{_tool_help(tool)}")
        if key in arguments:
            raise UsageError(f"'{key}' is given twice")
        arguments[key] = _value(raw, properties[key])
    missing = [key for key in tool.input_schema.get("required", ()) if key not in arguments]
    if missing:
        raise UsageError(f"{tool.name} needs {', '.join(f'{key}=' for key in missing)}\n\n{_tool_help(tool)}")
    return arguments


def _save(tool: str, index: int, block: ImageContent | AudioContent) -> Path:
    folder = data_dir() / "mcp" / socket.gethostname() / "media"
    folder.mkdir(parents=True, exist_ok=True)
    extension = mimetypes.guess_extension(block.mime_type) or ".bin"
    path = folder / f"{tool}-{time.strftime('%m%d%H%M%S')}-{os.getpid()}-{index}{extension}"
    path.write_bytes(base64.b64decode(block.data))
    return path


def _render(tool: str, result: CallToolResult, stream: TextIO) -> None:
    if result.structured_content is not None:
        print(json.dumps(result.structured_content, indent=2, default=str), file=stream)
        return
    for index, block in enumerate(result.content):
        if isinstance(block, TextContent):
            print(block.text, file=stream)
        elif isinstance(block, ImageContent | AudioContent):
            print(_save(tool, index, block), file=stream)
        else:
            print(block.model_dump_json(indent=2), file=stream)


async def _cancel(jobs: Iterable[Job]) -> None:
    for job in jobs:
        try:
            await JOBS.cancel(job.id)
        except ToolError as exc:
            print(exc, file=sys.stderr)


def _stream_logs(jobs: Iterable[Job], offsets: dict[str, int]) -> None:
    for job in jobs:
        if not job.log_path.is_file():
            continue
        with job.log_path.open("rb") as log:
            log.seek(offsets[job.id])
            data = log.read()
        offsets[job.id] += len(data)
        if data:
            sys.stderr.write(data.decode(errors="replace"))
            sys.stderr.flush()


def _show_progress(jobs: Iterable[Job], shown: dict[str, dict[str, Any]]) -> None:
    for job in jobs:
        if job.progress and job.progress != shown.get(job.id):
            shown[job.id] = dict(job.progress)
            print(f"{job.id}: {json.dumps(job.progress, default=str)}", file=sys.stderr)


async def _follow(jobs: list[Job], stop: asyncio.Event) -> int:
    offsets = {job.id: 0 for job in jobs}
    shown: dict[str, dict[str, Any]] = {}
    shown_at = 0.0
    canceller: asyncio.Task[None] | None = None
    while True:
        if stop.is_set() and canceller is None:
            print("stopping, Ctrl-C again exits at once and may leave it running", file=sys.stderr)
            canceller = asyncio.create_task(_cancel(jobs))
        _stream_logs(jobs, offsets)
        if time.monotonic() - shown_at >= PROGRESS_S:
            _show_progress(jobs, shown)
            shown_at = time.monotonic()
        if all(job.state in TERMINAL for job in jobs) and (canceller is None or canceller.done()):
            break
        await asyncio.sleep(POLL_S)
    _stream_logs(jobs, offsets)
    for job in jobs:
        print(json.dumps(job.as_dict(), indent=2, default=str))
    if stop.is_set():
        return INTERRUPTED
    return 0 if all(job.state == SUCCEEDED for job in jobs) else 1


def _trap(stop: asyncio.Event) -> None:
    def interrupted() -> None:
        if stop.is_set():
            print("exiting at once", file=sys.stderr)
            os._exit(INTERRUPTED)
        stop.set()

    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(number, interrupted)


async def _invoke(server: MCPServer, tool: Tool, arguments: dict[str, object]) -> int:
    stop = asyncio.Event()
    _trap(stop)
    call = asyncio.create_task(server.call_tool(tool.name, arguments))
    stopped = asyncio.create_task(stop.wait())
    await asyncio.wait({call, stopped}, return_when=asyncio.FIRST_COMPLETED)
    if not call.done():
        call.cancel()
        await asyncio.wait({call})
    stopped.cancel()
    if call.cancelled():
        jobs = JOBS.started_here()
        return await _follow(jobs, stop) if jobs else INTERRUPTED
    error = call.exception()
    if error is not None:
        if not isinstance(error, ToolError):
            raise error
        print(error, file=sys.stderr)
        return 1
    result = call.result()
    if isinstance(result, InputRequiredResult):
        print(f"{tool.name} asks for more input, which a terminal call cannot answer", file=sys.stderr)
        return 1
    jobs = JOBS.started_here()
    if jobs:
        _render(tool.name, result, sys.stderr)
        print(f"following {', '.join(job.id for job in jobs)}, Ctrl-C cancels", file=sys.stderr)
        return await _follow(jobs, stop)
    _render(tool.name, result, sys.stderr if result.is_error else sys.stdout)
    return 1 if result.is_error else 0


async def _run(name: str, tokens: list[str]) -> int:
    server, tool = await _load(name)
    if {"-h", "--help"} & set(tokens):
        print(_tool_help(tool))
        return 0
    arguments = _arguments(tool, tokens)
    try:
        return await _invoke(server, tool, arguments)
    finally:
        await JOBS.shutdown()
        await ROS.shutdown()


def main(argv: list[str]) -> int:
    """`python3 -m arena_mcp call TOOL [KEY=VALUE ...]`, the exit code of the call."""
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    os.environ.setdefault("RCUTILS_LOGGING_USE_STDOUT", "0")
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    ROS.settle_discovery = True
    try:
        return asyncio.run(_run(argv[0], argv[1:]))
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return 2


def completion_values() -> tuple[dict[str, dict[str, str]], list[str]]:
    """Tool names and per tool `key=` candidates for the completion manifest, with the domain modules as deps."""
    server, loaded = build()
    values: dict[str, dict[str, str]] = {"mcp.tools": {}}
    for tool in asyncio.run(server.list_tools()):
        values["mcp.tools"][tool.name] = (tool.description or "").split("\n", 1)[0]
        values[f"mcp.args.{tool.name}"] = {f"{key}=": _type_text(field) for key, field in tool.input_schema.get("properties", {}).items()}
    modules = (sys.modules.get(entry.source.partition(":")[0]) for entry in loaded if entry.served)
    return values, [os.path.realpath(module.__file__) for module in modules if module is not None and module.__file__]
