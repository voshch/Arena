import typing
from collections.abc import Iterable
from functools import cached_property
from pathlib import Path

import attrs
import yaml

from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    DynamicPaths,
    NetResolver,
    PathView,
)
from arena_simulation_setup.utils.models import ModelWrapper
from arena_simulation_setup.utils.models.model_loader import (
    ModelProvider_OBJ,
    ModelProvider_SDF,
)


class HumanView(PathView):
    """View around a resolved Human asset directory.

    Mirrors `ObjectView` so the resolved-asset accessor is uniform across
    asset kinds, callers can always do `(await ident.resolve()).model`.
    """

    @cached_property
    def model(self) -> ModelWrapper:
        return ModelWrapper(
            self.path.name,
            {
                **ModelProvider_SDF.asdict(self.path, self.path.name),
                **ModelProvider_OBJ.asdict(self.path, self.path.name),
            },
        )

    @cached_property
    def tags(self) -> frozenset[str]:
        """ASA tags of annotation.yaml, empty when the bundle carries none."""
        path = self.path / 'annotation.yaml'
        if not path.is_file():
            return frozenset()
        return frozenset((yaml.safe_load(path.read_text()) or {}).get('tags') or ())


@attrs.define(eq=False, hash=False)
class HumanIdentifier(DomainAssetIdentifier[HumanView]):
    """Represents an identifier referencing a 3D model asset."""

    _asset_type = 'Human'

    def load(self, path: Path, /, **kwargs: object) -> HumanView:
        del kwargs  # unused
        return HumanView(path)

    @classmethod
    async def tagged(cls, tags: Iterable[str]) -> list[typing.Self]:
        """Humans on disk whose annotation carries every tag, ordered by name."""
        wanted = frozenset(tags)
        found = [ident for ident in await cls.listall_async() if wanted <= (await ident.resolve()).tags]
        return sorted(found, key=lambda ident: ident.shortname)


HumanIdentifier.use(*DynamicPaths.as_resolvers(HumanIdentifier))
HumanIdentifier.use(*NetResolver.all(HumanIdentifier, formats=()))
