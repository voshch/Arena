from functools import cached_property
from pathlib import Path

import attrs

from arena_simulation_setup.tree import (
    DomainAssetIdentifier,
    DynamicPaths,
    NetResolver,
    PathView,
)
from arena_simulation_setup.utils.models import ModelWrapper
from arena_simulation_setup.utils.models.model_loader import (
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
            },
        )


@attrs.define(eq=False, hash=False)
class HumanIdentifier(DomainAssetIdentifier[HumanView]):
    """Represents an identifier referencing a 3D model asset."""

    _asset_type = 'Human'

    def load(self, path: Path, /, **kwargs: object) -> HumanView:
        del kwargs  # unused
        return HumanView(path)


HumanIdentifier.use(*DynamicPaths.as_resolvers(HumanIdentifier))
HumanIdentifier.use(*NetResolver.all(HumanIdentifier, formats=()))
