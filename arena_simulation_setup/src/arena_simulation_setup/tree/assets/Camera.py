"""Camera assets: one YAML file that sets up a single camera's shot or a whole rig, resolved scenario first, then world, then global."""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from functools import cached_property
from pathlib import Path

import yaml

from arena_simulation_setup import ASS_DIR, DOMAIN_DEFAULT
from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    DynamicPathResolver,
    DynamicPaths,
    NetResolver,
    PathView,
    SimplePathResolver,
)


class CameraView(PathView):
    """A resolved camera asset: `<name>.yaml` in an asset directory, or the file itself for a scenario camera."""

    @cached_property
    def spec(self) -> dict:
        path = self.path if self.path.is_file() else self.path / f'{self.path.name}.yaml'
        with open(path) as f:
            spec = yaml.safe_load(f) or {}
        if not isinstance(spec, dict):
            raise ValueError(f'camera asset {path}: expected a mapping, got {spec!r}')
        return spec


class CameraIdentifier(DomainAssetIdentifier[CameraView]):
    _asset_type = 'Camera'

    def load(self, path: Path, /, **kwargs: object) -> CameraView:
        del kwargs  # unused
        return CameraView(path)


class ScenarioCameraResolver(DynamicPathResolver[CameraIdentifier]):
    """The active scenario's `cameras/<name>.yaml` plain files, which have no domain."""

    def __init__(self) -> None:
        super().__init__(CameraIdentifier, DynamicPaths.SCENARIO, fn=lambda path: path / 'cameras')

    async def resolve(self, identifier: CameraIdentifier) -> Path | None:
        candidate = self.path / f'{identifier.name}.yaml'
        return candidate if identifier.domain == DOMAIN_DEFAULT and candidate.is_file() else None

    def listall(self, **kwargs: bool) -> Iterator[CameraIdentifier]:
        del kwargs  # unused
        for path in sorted(self.path.glob('*.yaml')) if self.path.is_dir() else ():
            yield CameraIdentifier(path.stem)

    async def listall_async(self, **kwargs: bool) -> list[CameraIdentifier]:
        return list(self.listall(**kwargs))


_SCENARIO = ScenarioCameraResolver()
_WORLD, _LINKED, _LOCAL, _IMPORTED = DynamicPaths.as_resolvers(CameraIdentifier)
_BUNDLED = SimplePathResolver(CameraIdentifier, ASS_DIR / 'assets')
_NETWORK = list(NetResolver.all(CameraIdentifier, formats=(), annotated=False, list_prefix=f'{DOMAIN_DEFAULT}/Camera'))
_TIERS = ((_SCENARIO,), (_WORLD, _LINKED), (_LOCAL, _IMPORTED, _BUNDLED))
_LISTED: dict[CameraIdentifier, bool] = {}

CameraIdentifier.use(_SCENARIO, _WORLD, _LINKED, _LOCAL, _IMPORTED, _BUNDLED, *_NETWORK)


async def local_spec(name: str) -> dict | None:
    """The spec of `name` from the local tiers, None when none holds it."""
    identifier = CameraIdentifier.parse(name)
    for resolver in itertools.chain.from_iterable(_TIERS):
        path = await resolver.resolve(identifier)
        if path is not None:
            return CameraView(path).spec
    return None


async def _on_network(identifier: CameraIdentifier) -> bool:
    """Whether a bucket lists `identifier`, asked once per process since an empty listing is not cached on disk."""
    if identifier not in _LISTED:
        _LISTED[identifier] = False
        for resolver in _NETWORK:
            if identifier in await resolver.listall_async(network=True):
                _LISTED[identifier] = True
                break
    return _LISTED[identifier]


async def find_spec(name: str) -> dict | None:
    """The spec of `name`, fetching it when only a bucket lists it, None when nothing holds it."""
    spec = await local_spec(name)
    if spec is not None:
        return spec
    identifier = CameraIdentifier.parse(name)
    if await _on_network(identifier):
        return (await identifier.resolve()).spec
    return None


async def candidates() -> list[CameraIdentifier]:
    """The sorted camera assets of the most specific local tier that holds any."""
    for tier in _TIERS:
        found = {identifier for resolver in tier for identifier in await resolver.listall_async()}
        if found:
            return sorted(found, key=lambda identifier: identifier.name)
    return []


async def pick_camera(request: str) -> tuple[CameraIdentifier | None, str | None]:
    """The camera asset a request selects and a warning for a fallback: exact name, else non-negative index, else index 0."""
    identifier = CameraIdentifier.parse(request)
    if await local_spec(request) is not None or (not request.isdecimal() and await _on_network(identifier)):
        return identifier, None
    found = await candidates()
    if not found:
        return None, f'no camera asset for {request!r}, no tier holds any'
    if request.isdecimal() and int(request) < len(found):
        return found[int(request)], None
    return found[0], f'no camera asset {request!r}, playing {found[0].name!r}'


def step_names(spec: Mapping) -> Iterator[str]:
    """Every step name of a spec's timeline and of each of its cameras' timelines."""
    timelines = [spec.get('timeline') or [], *((camera.get('timeline') or []) for camera in (spec.get('cameras') or {}).values())]
    for step in itertools.chain.from_iterable(timelines):
        yield from step


async def collect(root: str, spec: dict) -> dict[str, dict]:
    """A bundle of `root` and every camera asset it reaches, step names resolved through the local tiers only."""
    bundle = {root: spec}
    pending = list(step_names(spec))
    while pending:
        name = pending.pop()
        if name not in bundle:
            found = await local_spec(name)
            if found is not None:
                bundle[name] = found
                pending.extend(step_names(found))
    return bundle
