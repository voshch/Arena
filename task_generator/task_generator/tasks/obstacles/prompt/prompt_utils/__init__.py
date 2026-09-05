import os
import typing

from .context import ARENA_FORMAT, BEHAVIOR_FORMAT, SYSTEM_INSTRUCTION

# Preview model names get retired (gemini-3-pro-preview died 2026-07); keep the
# default on a stable name and allow overriding without a rebuild.
REMOTE_LM = os.environ.get("ARENA_PROMPT_LLM", "gemini-3.5-flash")

# HuNav's prompt backend only.
LOCAL_LM = os.environ.get("ARENA_PROMPT_LOCAL_LM", "Qwen/Qwen3-0.6B")

# Names served lazily by __getattr__ below, mapped to the module they come from.
# arena_hunav_sim_bridge and chromadb are optional deps: importing them at module
# level would make `tm_obstacles:=prompt` unimportable wherever hunav is not
# installed, which is every arena-only workspace. Resolving them on first
# attribute access keeps the cost on the code path that actually needs them —
# `prompt.hunav`, itself only imported when human:=hunav registers the mode.
_LAZY: typing.Final[dict[str, str]] = {
    "BEHAVIOR_TREE_FORMAT": ".context_bt",
    "SPLIT_PROMPT_INSTRUCTION": ".context_bt",
    "BT_REF_DOC_PATH": "arena_hunav_sim_bridge",
    "CHROMA_DB_PATH": "arena_hunav_sim_bridge",
    "create_chroma_db": ".vector_db",
    "get_chroma_collection": ".vector_db",
    "get_relevant_bt_nodes": ".vector_db",
    "process_json_doc": ".vector_db",
}


def __getattr__(name: str) -> object:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        resolved = importlib.import_module(module, __package__ if module.startswith(".") else None)
    except ImportError as e:
        raise AttributeError(f"{name!r} needs the optional HuNav prompt dependencies ({module}). Install hunav_sim and arena_hunav_sim_bridge from arena.repos, or use tm_obstacles:=prompt under human:=arena, which does not require them.") from e
    value = getattr(resolved, name)
    globals()[name] = value
    return value


__all__ = [
    "ARENA_FORMAT",
    "BEHAVIOR_FORMAT",
    "BEHAVIOR_TREE_FORMAT",
    "BT_REF_DOC_PATH",
    "CHROMA_DB_PATH",
    "LOCAL_LM",
    "REMOTE_LM",
    "SPLIT_PROMPT_INSTRUCTION",
    "SYSTEM_INSTRUCTION",
    "create_chroma_db",
    "get_chroma_collection",
    "get_relevant_bt_nodes",
    "process_json_doc",
]
