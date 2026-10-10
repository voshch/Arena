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
    ModelProvider_USD,
)


class ObjectView(PathView):
    """View around a resolved Object asset directory.

    Exposes typed accessors over the asset contents (model + bounds), keeping
    asset-format details out of consumers. Mirrors the Map/ScenarioView/World
    pattern used elsewhere in the tree.
    """

    @cached_property
    def model(self) -> ModelWrapper:
        return ModelWrapper(
            self.path.name,
            {
                **ModelProvider_USD.asdict(self.path, self.path.name),
                **ModelProvider_SDF.asdict(self.path, self.path.name),
                **ModelProvider_OBJ.asdict(self.path, self.path.name),
            },
        )

    @cached_property
    def annotation(self) -> dict | None:
        """Parsed annotation.yaml contents, or None if absent."""
        ann_path = self.path / 'annotation.yaml'
        if not ann_path.exists():
            return None
        return yaml.safe_load(ann_path.read_text())

    @cached_property
    def bounds(self) -> list[tuple[float, float]] | None:
        """2D footprint corners in obstacle-local frame (pre-rotation), or None
        if annotation.yaml is absent or lacks a bounding_box."""
        ann = self.annotation
        if ann is None:
            return None
        bbox = ann.get('bounding_box')
        if bbox is None:
            return None
        (min_x, max_x), (min_y, max_y), _ = bbox
        return [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

    @cached_property
    def bbox(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        """3D bounding box as (size, center) in obstacle-local frame, or None if
        annotation.yaml is absent or lacks a bounding_box."""
        ann = self.annotation
        if ann is None:
            return None
        bbox = ann.get('bounding_box')
        if bbox is None:
            return None
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = bbox
        size = (max_x - min_x, max_y - min_y, max_z - min_z)
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2)
        return size, center


@attrs.define(eq=False, hash=False)
class ObjectIdentifier(DomainAssetIdentifier[ObjectView]):
    """Represents an identifier referencing a 3D model asset."""

    _asset_type = 'Object'

    def load(self, path: Path, /, **kwargs: object) -> ObjectView:
        del kwargs  # unused
        return ObjectView(path)


ObjectIdentifier.use(*DynamicPaths.as_resolvers(ObjectIdentifier))
ObjectIdentifier.use(*NetResolver.all(ObjectIdentifier))
