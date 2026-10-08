from functools import cached_property
from pathlib import Path

import attrs
import yaml

from arena_simulation_setup import DOMAIN_DEFAULT
from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    DynamicPaths,
    NetResolver,
    PathView,
)


class SoundView(PathView):
    """View around a resolved Sound asset directory, a `<name>.yaml` manifest beside its wav files."""

    @cached_property
    def manifest(self) -> dict:
        with open(self.path / f'{self.path.name}.yaml') as f:
            return yaml.safe_load(f)


@attrs.define(eq=False, hash=False)
class SoundIdentifier(DomainAssetIdentifier[SoundView]):
    """Represents an identifier referencing a sound asset."""

    _asset_type = 'Sound'

    def load(self, path: Path, /, **kwargs: object) -> SoundView:
        del kwargs  # unused
        return SoundView(path)


SoundIdentifier.use(*DynamicPaths.as_resolvers(SoundIdentifier))
SoundIdentifier.use(*NetResolver.all(SoundIdentifier, formats=(), annotated=False, list_prefix=f'{DOMAIN_DEFAULT}/Sound'))
