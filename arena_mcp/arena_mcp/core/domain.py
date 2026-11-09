from __future__ import annotations

import dataclasses
import functools
import importlib.util
import inspect
import json
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

F = TypeVar("F", bound=Callable[..., Any])


@dataclasses.dataclass(frozen=True)
class ToolSpec:
    fn: Callable[..., Any]
    name: str
    title: str | None
    annotations: ToolAnnotations

    @property
    def structured(self) -> bool:
        """Only dict results get a structured schema, every other result (lists, images, audio) goes out as plain content."""
        returns = self.fn.__annotations__.get("return")
        if returns is dict:
            return True
        return _is_dict_annotation(returns if isinstance(returns, str) else str(returns))

    @property
    def summary(self) -> str:
        return inspect.cleandoc(self.fn.__doc__ or "").split("\n", 1)[0]


_SPECS: dict[Callable[..., Any], ToolSpec] = {}


def tool_spec(fn: Callable[..., Any]) -> ToolSpec:
    """The registration of a function decorated with Domain.tool, KeyError when it is none."""
    return _SPECS[fn]


@dataclasses.dataclass(frozen=True)
class ResourceSpec:
    fn: Callable[..., Any]
    uri: str
    mime_type: str | None
    description: str | None


@dataclasses.dataclass(frozen=True)
class PromptSpec:
    fn: Callable[..., Any]
    name: str
    description: str | None


class Domain:
    """A named group of tools, resources and prompts one package contributes to the server."""

    def __init__(
        self,
        name: str,
        doc: str,
        *,
        requires: Iterable[str] = (),
        check: Callable[[], str | None] | None = None,
    ) -> None:
        if not name.isidentifier() or name != name.lower():
            raise ValueError(f"domain name {name!r} must be a lowercase identifier")
        self.name = name
        self.doc = inspect.cleandoc(doc)
        self.requires = tuple(requires)
        self._check = check
        self.tools: list[ToolSpec] = []
        self.resources: list[ResourceSpec] = []
        self.prompts: list[PromptSpec] = []

    def unavailable(self) -> str | None:
        """Why this domain cannot be served in this environment, None when it can."""
        missing = [module for module in self.requires if _missing(module)]
        if missing:
            return f"missing {', '.join(missing)}"
        return self._check() if self._check is not None else None

    def tool(
        self,
        *,
        read_only: bool = False,
        destructive: bool = False,
        idempotent: bool = False,
        open_world: bool = False,
        title: str | None = None,
    ) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            name = fn.__name__
            if not name.startswith(f"{self.name}_"):
                raise ValueError(f"tool {name!r} of domain {self.name!r} must be named {self.name}_<verb>")
            if not (fn.__doc__ or "").strip():
                raise ValueError(f"tool {name!r} needs a docstring, it is the description agents read")
            annotations = ToolAnnotations(
                title=title,
                read_only_hint=read_only,
                destructive_hint=destructive and not read_only,
                idempotent_hint=idempotent or read_only,
                open_world_hint=open_world,
            )
            spec = ToolSpec(fn=fn, name=name, title=title, annotations=annotations)
            self.tools.append(spec)
            _SPECS[fn] = spec
            return fn

        return decorator

    def resource(self, uri: str, *, mime_type: str | None = None, description: str | None = None) -> Callable[[F], F]:
        if not uri.startswith(f"arena://{self.name}/"):
            raise ValueError(f"resource {uri!r} of domain {self.name!r} must live under arena://{self.name}/")

        def decorator(fn: F) -> F:
            self.resources.append(ResourceSpec(fn=fn, uri=uri, mime_type=mime_type, description=description))
            return fn

        return decorator

    def prompt(self, *, description: str | None = None) -> Callable[[F], F]:
        def decorator(fn: F) -> F:
            name = fn.__name__
            if not name.startswith(f"{self.name}_"):
                raise ValueError(f"prompt {name!r} of domain {self.name!r} must be named {self.name}_<what>")
            self.prompts.append(PromptSpec(fn=fn, name=name, description=description))
            return fn

        return decorator

    def install(self, server: MCPServer) -> None:
        for spec in self.tools:
            fn = spec.fn if spec.structured else _packed(spec.fn)
            server.add_tool(fn, name=spec.name, title=spec.title, annotations=spec.annotations, structured_output=None if spec.structured else False)
        for resource in self.resources:
            server.resource(resource.uri, mime_type=resource.mime_type, description=resource.description)(resource.fn)
        for prompt in self.prompts:
            server.prompt(name=prompt.name, description=prompt.description)(prompt.fn)


def _pack(result: object) -> object:
    from mcp.server.mcpserver import Audio, Image

    if isinstance(result, list | tuple) and not any(isinstance(item, Image | Audio) for item in result):
        return json.dumps(result, indent=2, default=str)
    return result


def _packed(fn: Callable[..., object]) -> Callable[..., object]:
    """The tool with a list result sent as one JSON text block, mixed media lists left as content blocks."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def run_async(*args: object, **kwargs: object) -> object:
            return _pack(await fn(*args, **kwargs))

        return run_async

    @functools.wraps(fn)
    def run(*args: object, **kwargs: object) -> object:
        return _pack(fn(*args, **kwargs))

    return run


def _missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:
        return True


def _is_dict_annotation(text: str) -> bool:
    text = text.replace(" ", "")
    for prefix in ("dict", "typing.Dict"):
        if text == prefix:
            return True
        if text.startswith(f"{prefix}["):
            depth = 0
            for index in range(len(prefix), len(text)):
                depth += text[index] == "["
                depth -= text[index] == "]"
                if depth == 0:
                    return index == len(text) - 1
    return False
