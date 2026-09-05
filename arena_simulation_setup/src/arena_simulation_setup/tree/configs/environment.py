import json
from pathlib import Path

import yaml
from arena_simulation_setup import ASS_DIR
from arena_simulation_setup.tree import Identifier, PathResolverBase
from typing_extensions import Self


class EnvironmentDescription(dict):
    # TODO
    ...


_SUFFIXES = ('.yaml', '.json')


def _strip_suffix(name: str) -> str:
    for suffix in _SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


class EnvironmentResolver(PathResolverBase):
    suffixes = _SUFFIXES

    @property
    def path(self) -> Path:
        return ASS_DIR / 'configs' / 'environment'


class EnvironmentIdentifier(Identifier[EnvironmentDescription]):
    """Name with or without suffix. Suffix-less names probe .yaml then .json."""

    @property
    def shortname(self) -> str:
        return _strip_suffix(self.name)

    @classmethod
    def from_relpath(cls, relpath: Path) -> Self:
        if relpath.name.endswith(_SUFFIXES):
            return cls(name=_strip_suffix(str(relpath)))
        raise FileNotFoundError(f"Invalid file {relpath} for environment identifier")

    def load(self, path: Path, /, **kwargs: object) -> EnvironmentDescription:
        del kwargs
        with open(path) as f:
            value = json.load(f) if path.suffix == '.json' else yaml.safe_load(f)
        if not isinstance(value, dict):
            raise ValueError(f"Environment file {path} must contain a mapping at the top level")
        return EnvironmentDescription(value)


EnvironmentIdentifier.use(EnvironmentResolver(EnvironmentIdentifier))
