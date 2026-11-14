"""A generated scenario with its agent types, provenance and cameras, packed into the files of its scenario directory."""

from __future__ import annotations

import typing

import attrs
import yaml

GENERATED_MARKER = "# Generated - do not hand-edit; regenerate instead."


@attrs.frozen
class Rejected:
    reason: str


@attrs.frozen
class Generated:
    scenario: dict[str, object]
    types: dict[str, dict[str, object]]
    provenance: dict[str, object]
    cameras: dict[str, dict[str, object]] = attrs.Factory(dict)


def summary(reasons: typing.Sequence[str], limit: int = 3) -> str:
    """The first distinct rejection reasons, joined."""
    return "; ".join(list(dict.fromkeys(reasons))[:limit])


def _yaml(doc: object, marked: bool) -> bytes:
    text = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
    return ((GENERATED_MARKER + "\n" + text) if marked else text).encode("utf-8")


def pack(name: str, generated: Generated) -> dict[str, bytes]:
    """The directory of scenario `name`, by path relative to its world."""
    root = f"scenarios/{name}"
    files = {f"{root}/scenario.yaml": _yaml(generated.scenario, True), f"{root}/provenance.yaml": _yaml(generated.provenance, False)}
    files.update({f"{root}/{path}": _yaml(doc, True) for path, doc in generated.types.items()})
    files.update({f"{root}/cameras/{camera}.yaml": _yaml(spec, True) for camera, spec in generated.cameras.items()})
    return files


def binding(name: str, axes: typing.Sequence[str] = ("robots", "obstacles")) -> dict[str, typing.Any]:
    """Episode binding running scenario `name` on the task axes `axes`."""
    return {"modes": {axis: "scenario" for axis in axes}, "params": {axis: {"file": name} for axis in axes}}
