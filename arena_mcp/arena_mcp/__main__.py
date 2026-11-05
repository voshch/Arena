from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from . import call
from .server import describe, discover, serve

USAGE = """\
python3 -m arena_mcp serve [domain ...]   serve the domains over stdio (all served domains by default)
python3 -m arena_mcp ls [domain ...]      list installed domains, whether each is served and why not
python3 -m arena_mcp ls --json [domain]   the same as JSON, with every tool
python3 -m arena_mcp call TOOL [KEY=VALUE ...]   run one tool, see `call --help`"""


def _ls(names: list[str], as_json: bool) -> int:
    found = discover(names or None)
    if as_json:
        print(json.dumps([describe(entry) for entry in found], indent=2))
        return 0
    width = max((len(entry.name) for entry in found), default=6)
    for entry in found:
        tools = len(entry.domain.tools) if entry.domain is not None else 0
        status = "served" if entry.served else f"unavailable: {entry.reason}"
        print(f"{entry.name:<{width}}  {tools:>3} tools  {status}")
        if names and entry.domain is not None:
            for spec in entry.domain.tools:
                print(f"    {spec.name}  {spec.summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["call"]:
        return call.main(argv[1:])
    parser = argparse.ArgumentParser(prog="arena_mcp", description="Arena MCP server", usage=USAGE)
    sub = parser.add_subparsers(dest="verb", metavar="serve|ls|call")
    serve_parser = sub.add_parser("serve", help="serve over stdio")
    serve_parser.add_argument("domains", nargs="*", help="domains to serve, all served ones when omitted")
    ls_parser = sub.add_parser("ls", help="list domains")
    ls_parser.add_argument("domains", nargs="*", help="domains to list with their tools")
    ls_parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub.add_parser("call", help="run one tool")
    args = parser.parse_args(argv)
    if args.verb is None:
        parser.print_help()
        return 2
    if args.verb == "ls":
        return _ls(args.domains, args.json)
    os.environ.setdefault("RCUTILS_LOGGING_USE_STDOUT", "0")
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    try:
        asyncio.run(serve(args.domains or None))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
