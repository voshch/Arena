"""Layout IR: the geometry every world-generator front end compiles down to."""

import attrs
import shapely

from arena_simulation_setup.tree.World import Level, World, Zone

from .utils import to_corners, to_walls, world_of

Point = tuple[float, float]


@attrs.define
class Segment:
    """A centreline edge carrying its own width."""

    a: Point
    b: Point
    width: float


@attrs.define
class Area:
    """A filled polygon, unioned with the buffered segments."""

    polygon: shapely.Polygon | shapely.MultiPolygon


@attrs.define
class Region:
    """A named area used to name and materialise zones, geometry comes from the union."""

    name: str
    polygon: shapely.Polygon | shapely.MultiPolygon
    description: str = ''
    material: str | None = None


@attrs.define
class Note:
    """Something the caller should know about a cell of the source, by zero-based position."""

    row: int
    col: int
    text: str


@attrs.define
class Diagnostics:
    components: int
    islands: int
    zones: int
    extent: tuple[float, float]


@attrs.define
class GridFrame:
    """Where a generator's cells sit in world metres, so an editor can draw on the rendered map."""

    origin: tuple[float, float]  # world metres at the bottom-left corner of cell (rows - 1, 0)
    pitch: float
    rows: int
    cols: int


@attrs.define
class Layout:
    segments: list[Segment] = attrs.field(factory=list)
    areas: list[Area] = attrs.field(factory=list)
    regions: list[Region] = attrs.field(factory=list)

    def geometry(self) -> shapely.geometry.base.BaseGeometry:
        """Free space: buffered centrelines unioned with the filled areas."""
        pieces: list[shapely.Polygon | shapely.MultiPolygon] = [area.polygon for area in self.areas]
        for segment in self.segments:
            line = shapely.LineString([segment.a, segment.b])
            pieces.append(line.buffer(segment.width / 2, cap_style='flat', join_style='mitre'))
        if not pieces:
            return shapely.Polygon()
        return shapely.union_all(pieces)


def _polygons(geometry: shapely.geometry.base.BaseGeometry) -> list[shapely.Polygon]:
    """Polygonal parts only: cutting a hole open can shed a line or a point alongside them."""
    if isinstance(geometry, shapely.Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, shapely.MultiPolygon | shapely.GeometryCollection):
        return [polygon for part in geometry.geoms for polygon in _polygons(part)]
    return []


def _simple(geometry: shapely.geometry.base.BaseGeometry) -> list[shapely.Polygon]:
    """Zone corners are one ring, so a part with holes is sliced into vertical bands at every hole
    edge. No band can contain a hole, and the cuts land on coordinates the geometry already has."""
    simple: list[shapely.Polygon] = []
    for part in _polygons(geometry):
        if not part.interiors:
            simple.append(part)
            continue
        minx, miny, maxx, maxy = part.bounds
        holes = [shapely.Polygon(ring).bounds for ring in part.interiors]
        # Two edges that differ only in the last bits of a float are the same edge. Slicing between
        # them yields a hairline band that is valid geometry and a meaningless zone.
        apart = (maxx - minx) * 1e-9
        edges: list[float] = []
        for edge in sorted((minx, maxx, *(bound[0] for bound in holes), *(bound[2] for bound in holes))):
            if not edges or edge - edges[-1] > apart:
                edges.append(edge)
        for left, right in zip(edges, edges[1:], strict=False):
            simple.extend(_polygons(part.intersection(shapely.box(left, miny, right, maxy))))
    return simple


def diagnostics_of(level: Level) -> Diagnostics:
    """Topology of a finished level. Reads the zones, so it holds for generators built by hand
    as well as for those compiled from a layout."""
    rings = [zone.shape() for zone in level.zones if len(zone.polygon) >= 3]
    parts = _polygons(shapely.union_all(rings)) if rings else []
    minx, miny, maxx, maxy = shapely.union_all(parts).bounds if parts else (0.0, 0.0, 0.0, 0.0)
    return Diagnostics(
        components=len(parts),
        islands=sum(len(part.interiors) for part in parts),
        zones=len(level.zones),
        extent=(maxx - minx, maxy - miny),
    )


def compile_layout(layout: Layout, ceiling: bool = False, grid: float | None = None) -> tuple[World, Diagnostics]:
    """Turn a layout into a one-level World plus what the caller should be told about it, regions cut on a *grid* in meters when given."""
    geometry = layout.geometry()
    parts = _polygons(geometry)

    boundary = shapely.MultiLineString([ring for part in parts for ring in [part.exterior, *part.interiors]])

    # A region yields whatever earlier regions left of it. Accumulating that as one growing
    # polygon costs a difference against the whole world per region. Subtracting the earlier
    # regions themselves is the same set, and lets the index rule out the ones nowhere near.
    claims = [region.polygon for region in layout.regions]
    neighbours = shapely.STRtree(claims) if claims else None

    roof: dict = {} if ceiling else {'ceiling': None}
    zones: list[Zone] = []
    descriptions: dict[str, str] = {}
    for order, region in enumerate(layout.regions):
        piece = region.polygon.intersection(geometry, grid_size=grid)
        assert neighbours is not None
        earlier = [taken for taken in neighbours.query(region.polygon) if taken < order]
        if earlier:
            piece = piece.difference(shapely.union_all([claims[taken] for taken in earlier]), grid_size=grid)
        for index, part in enumerate(_simple(piece)):
            suffix = '' if index == 0 else f'_{index}'
            fields: dict = {'name': f'{region.name}{suffix}', 'polygon': to_corners(part), **roof}
            if region.material is not None:
                fields['floor'] = region.material
            zones.append(Zone(**fields))
            descriptions[zones[-1].name] = region.description

    leftover = geometry.difference(shapely.union_all(claims), grid_size=grid) if claims else geometry
    for index, part in enumerate(_simple(leftover)):
        zones.append(Zone(name=f'area_{index}', polygon=to_corners(part), **roof))

    level = Level(zones=zones, walls=to_walls(boundary) if zones else [])
    return world_of(level, descriptions), diagnostics_of(level)
