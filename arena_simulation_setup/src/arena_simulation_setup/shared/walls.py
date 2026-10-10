from __future__ import annotations

import typing

import attrs

from arena_simulation_setup.tree.assets.Material import Material, MaterialIdentifier
from arena_simulation_setup.utils.cattrs import Serializable
from arena_simulation_setup.utils.geometry import Position

if typing.TYPE_CHECKING:
    from arena_simulation_setup.tree.Wall import WallRealization


@attrs.define
class Wall(Serializable):
    start: Position = attrs.field(converter=Position.converter)
    end: Position = attrs.field(converter=Position.converter)
    kind: str = ''
    material: MaterialIdentifier | None = None
    top: float | None = None
    bottom: float | None = None

    async def assets(self) -> WallRealization:
        """
        Get sub-assets that make up the wall.
        """
        from arena_simulation_setup.tree.Wall import WallDescription, WallIdentifier

        try:
            if self.kind:
                _description = await WallIdentifier(self.kind).resolve()
            else:
                _description = WallDescription.simple(material=self.material if self.material else None)
            return _close_to(self.top, self.bottom, _description.realize(self.start, self.end))
        except Exception as e:
            import logging
            import traceback

            logging.error(f"Failed to load wall assets for wall from {self.start} to {self.end} of kind '{self.kind}' and material '{self.material}': {e}\n{traceback.format_exc()}")

            return _close_to(self.top, self.bottom, WallDescription.simple(material=Material.default('wall')).realize(self.start, self.end))

    def __iter__(self):
        yield self.start
        yield self.end

    def serialize(self) -> dict:
        ser = attrs.asdict(self)
        ser.pop('top', None)
        ser.pop('bottom', None)
        if self.kind or not self.material:
            ser.pop('material', None)
        if not self.kind:
            ser.pop('kind', None)
        return ser


def _close_to(top: float | None, bottom: float | None, realization: WallRealization) -> WallRealization:
    """Raise the highest segment of a wall stack to top, or make the wall one segment from bottom to top."""
    segments, obstacles = realization
    segments = list(segments)
    if top is None or not segments:
        return segments, obstacles
    highest = max(range(len(segments)), key=lambda index: segments[index].start.z + segments[index].height)
    segment = segments[highest]
    if bottom is not None:
        lintel = attrs.evolve(
            segment,
            start=Position(segment.start.x, segment.start.y, bottom),
            end=Position(segment.end.x, segment.end.y, bottom),
            height=top - bottom,
        )
        return [lintel], ()
    if segment.start.z + segment.height < top:
        segments[highest] = attrs.evolve(segment, height=top - segment.start.z)
    return segments, obstacles
