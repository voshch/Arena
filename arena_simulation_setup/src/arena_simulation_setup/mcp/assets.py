from __future__ import annotations

import fnmatch
import json
import sys
import typing
from pathlib import Path
from typing import Annotated

from pydantic import Field

from arena_mcp.core import JOBS, Domain, Job, ToolError, cli


def _ready() -> str | None:
    try:
        cli.arena_dir()
    except ToolError as exc:
        return exc.message
    return None


domain = Domain(
    "assets",
    """Resolvable Arena data of every kind `arena asset` moves (assets_kinds lists them) through the same code the `arena asset` verbs run.
    find and ls show what each resolver answers, pull fetches into the local cache, push publishes to the kind's bucket after the preflight that refuses unresolvable references.
    import_sources lists the upstream datasets `arena asset import` converts from, import_items one dataset's importable items, and import converts chosen items into the imported root as a job.
    retarget_animation turns SMPL-X motion files into Animation clips as a job.
    articulation shows the links, joints, named states, detents and field families of an articulated object, the values scenario objects take as `joints:`.""",
    check=_ready,
)

Kind = Annotated[str, Field(description="asset kind, assets_kinds lists them")]
Bucket = Annotated[str | None, Field(description="bucket to use instead of the kind's first configured one")]


def _json(*args: str, check: bool = True) -> object:
    done = cli.run(*args, check=check, clip=None)
    try:
        return json.loads(done["stdout"])
    except json.JSONDecodeError as exc:
        tail = (done["stderr"] or done["stdout"]).strip().splitlines()[-10:]
        raise ToolError(f"{done['command']} exited {done['exit_code']} without JSON:\n" + "\n".join(tail), hint="assets_kinds lists the kinds") from exc


@domain.tool(read_only=True)
def assets_kinds() -> dict[str, object]:
    """The resolvable kinds `arena asset` moves, with a description each."""
    return typing.cast(dict[str, object], _json("asset", "ls", "--json"))


@domain.tool(read_only=True, open_world=True)
def assets_find(kind: Kind, name: Annotated[str, Field(description="asset name, e.g. domain/name for domain assets")]) -> dict[str, object]:
    """Every resolver's verdict for one asset (hit, miss, shadowed, stale) without downloading it, like `arena asset find`."""
    verdicts = typing.cast(list[dict[str, object]], _json("asset", "find", kind, name, "--json", check=False))
    return {"kind": kind, "name": name, "resolves": any(verdict["verdict"] in ("hit", "stale") for verdict in verdicts), "verdicts": verdicts}


@domain.tool(read_only=True, open_world=True)
def assets_ls(
    kind: Kind,
    network: Annotated[bool, Field(description="also list bucket-only entries and mark local copies that shadow them")] = False,
    pattern: Annotated[str | None, Field(description="glob over the names")] = None,
) -> dict[str, object]:
    """Every available asset of one kind, like `arena asset ls`."""
    listed = typing.cast(list[object], _json("asset", "ls", kind, "--json", *(("--network",) if network else ())))
    rows = [entry if isinstance(entry, dict) else {"name": entry} for entry in listed]
    if pattern is not None:
        rows = [row for row in rows if fnmatch.fnmatchcase(str(row["name"]), pattern)]
    return {"kind": kind, "count": len(rows), "assets": rows}


@domain.tool(open_world=True, idempotent=True)
def assets_pull(kind: Kind, name: Annotated[str, Field(description="asset name")], bucket: Bucket = None) -> dict[str, object]:
    """Download one asset from its bucket into the local cache and return the cache path, like `arena asset pull`."""
    done = cli.run("asset", "pull", kind, name, *(("--bucket", bucket) if bucket else ()), timeout=1800, check=False)
    lines = done["stdout"].strip().splitlines()
    if done["exit_code"] != 0 or not lines:
        reason = (done["stderr"] or done["stdout"]).strip().splitlines()[-5:]
        raise ToolError(f"{kind} {name} was not pulled:\n" + "\n".join(reason), hint="assets_find shows which resolvers know it")
    return {"kind": kind, "name": name, "path": lines[-1]}


@domain.tool(destructive=True, open_world=True)
def assets_push(
    kind: Kind,
    name: Annotated[str | None, Field(description="asset name, none publishes every local asset of the kind in one upload")],
    bucket: Bucket = None,
    force: Annotated[bool, Field(description="publish despite a failed preflight or a world that ships in the repo")] = False,
    only: Annotated[str | None, Field(description="publish only the files matching GLOB[,GLOB], the bucket's other files stay as they are")] = None,
) -> dict[str, object]:
    """Publish one asset to its kind's bucket after the preflight `arena asset push` runs, which refuses references nobody else could resolve, with GCS_ACCESS_TOKEN set."""
    done = cli.run("asset", "push", kind, *((name,) if name else ("--all",)), "--yes", *(("--bucket", bucket) if bucket else ()), *(("--force",) if force else ()), *(("--only", only) if only else ()), timeout=3600, check=False)
    if done["exit_code"] != 0:
        reason = (done["stderr"] + done["stdout"]).strip().splitlines()[-30:]
        raise ToolError(f"{kind} {name} was not published:\n" + "\n".join(reason), hint="bundle the missing references under the asset directory, or pass force=true")
    return {"kind": kind, "name": name, "published": done["stdout"].strip().splitlines()[-1:]}


@domain.tool(read_only=True, open_world=True)
def assets_articulation(name: Annotated[str, Field(description="object name as a scenario's objects reference it, [domain/]name")]) -> dict[str, object]:
    """Links, joints (type, parent, child, limits, speed, role, authored dynamics), link materials, named states, detents, field families (family to member joints) and swept box of an articulated object, or that the object is rigid."""
    from arena_simulation_setup.tree.assets.Object import ObjectIdentifier

    try:
        view = ObjectIdentifier.parse(name).resolve_sync()
    except FileNotFoundError as exc:
        raise ToolError(f"object {name!r} does not resolve: {exc}", hint="assets_find with kind object shows every resolver's verdict") from exc
    try:
        articulation = view.articulation
    except ValueError as exc:
        raise ToolError(f"object {name!r} has an invalid articulation: {exc}", hint="its annotation.yaml carries the articulation the build wrote") from exc
    if articulation is None:
        return {"object": name, "path": str(view.path), "articulated": False}
    return {
        "object": name,
        "path": str(view.path),
        "articulated": True,
        "root": articulation.root,
        "links": [{"name": link, "nodes": nodes, "bundle": (view.path / "links" / link).is_dir()} for link, nodes in articulation.links.items()],
        "joints": [{**joint.unstructure(), "speed": joint.speed} for joint in articulation.joints],
        "materials": {link: material.unstructure() for link, material in articulation.materials.items()},
        "states": articulation.states,
        "detents": articulation.detents,
        "families": {family: list(members) for family, members in articulation.families().items()},
        "initial": articulation.initial(),
        "swept": None if articulation.swept is None else [list(side) for side in articulation.swept],
    }


ImportKind = Annotated[str, Field(description="kind an import writes, assets_import_sources lists them with their sources")]
Source = Annotated[str, Field(description="upstream dataset of that kind, assets_import_sources lists them")]


@domain.tool(read_only=True)
def assets_import_sources(kind: Annotated[str | None, Field(description="only the sources of this kind")] = None) -> dict[str, object]:
    """The upstream datasets `arena asset import` converts from, with kind, license verdict, host and granularity."""
    sources = typing.cast(list[dict[str, object]], _json("asset", "import", *((kind,) if kind else ()), "--json"))
    return {"count": len(sources), "sources": sources}


@domain.tool(read_only=True, open_world=True)
def assets_import_items(
    kind: ImportKind,
    source: Source,
    pattern: Annotated[str | None, Field(description="glob over the item ids")] = None,
    limit: Annotated[int, Field(description="most item ids returned, count is the full match")] = 500,
) -> dict[str, object]:
    """The importable item ids of one dataset and the root they import into, the list cached like `arena asset import KIND SOURCE`."""
    listed = typing.cast(dict[str, object], _json("asset", "import", kind, source, "--json", check=False))
    items = [str(item) for item in typing.cast(list[object], listed["items"])]
    if pattern is not None:
        items = [item for item in items if fnmatch.fnmatchcase(item, pattern)]
    return {"kind": kind, "source": source, "target": listed["target"], "count": len(items), "items": items[:limit]}


def _imported(job: Job) -> dict[str, object]:
    lines = job.log_path.read_text().splitlines() if job.log_path.is_file() else []
    return {"paths": [line for line in lines if line.startswith("/") and Path(line).is_dir()]}


def _retargeted(job: Job) -> dict[str, object]:
    lines = job.log_path.read_text().splitlines() if job.log_path.is_file() else []
    return {"paths": [first for line in lines if (first := line.split(" ", 1)[0]).startswith("/") and Path(first).is_dir()]}


@domain.tool(open_world=True, idempotent=True)
async def assets_import(
    kind: ImportKind,
    source: Source,
    items: Annotated[list[str], Field(description="item ids of that source, assets_import_items lists them", min_length=1)],
) -> dict[str, object]:
    """Convert items of one dataset into the imported root the resolvers read, each with an ATTRIBUTION.yaml, as a job whose result lists the item paths."""
    if (kind, source) not in {(str(row["kind"]), str(row["name"])) for row in typing.cast(list[dict[str, object]], _json("asset", "import", "--json"))}:
        raise ToolError(f"no {kind} import source {source!r}", hint="assets_import_sources lists the sources by kind")
    job = await cli.job("import", f"import {kind} {source} {' '.join(items)}", "asset", "import", kind, source, *items, result=_imported)
    return job.as_dict()


@domain.tool(idempotent=True)
async def assets_retarget_animation(
    sources: Annotated[list[str], Field(description="SMPL-X .npz files or directories, absolute paths", min_length=1)],
    out: Annotated[str, Field(description="Animation directory to write the clip folders into, absolute path")],
    pair: Annotated[bool, Field(description="each source is a two-person interaction, written as <name>_a and <name>_b")] = False,
    loop: Annotated[bool, Field(description="mark the clips as looping")] = False,
) -> dict[str, object]:
    """Retarget SMPL-X motion onto Arena's 40-DOF human wire as a job whose result lists the clip directories."""
    if missing := [source for source in sources if not Path(source).exists()]:
        raise ToolError(f"no such source: {', '.join(missing)}", hint="sources are absolute .npz files or directories")
    flags = [*(["--pair"] if pair else []), *(["--loop"] if loop else [])]
    argv = [sys.executable, "-m", "arena_simulation_setup.adapters.arena_animation_retarget", *sources, "-o", out, *flags]
    job = await JOBS.start_process("retarget", f"retarget {' '.join(sources)}", argv, cwd=cli.ws_dir(), result=_retargeted)
    return job.as_dict()
