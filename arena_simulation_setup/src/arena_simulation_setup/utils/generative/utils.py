from shapely import LineString, MultiLineString
from shapely.geometry import Point, Polygon

from arena_simulation_setup.shared import Position, Wall
from arena_simulation_setup.tree.assets.Material import MaterialIdentifier


def line_pairs(geom: MultiLineString | LineString | Polygon):
    """Create an iterator for all line segments of a geometry.

    Args:
        geom (MultiLineString | LineString | Polygon): The geometry to extract line segments from.

    Yields:
        tuple[Point, Point]: A tuple of start and end points for each line segment.
    """
    if geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield from line_pairs(geom.exterior)
        for interior_ring in geom.interiors:
            yield from line_pairs(interior_ring)
        return

    if isinstance(geom, LineString):
        geom = MultiLineString((geom,))

    for line in geom.geoms:
        coords = list(line.coords)
        if len(coords) >= 2:
            for start, end in zip(coords[:-1], coords[1:], strict=False):
                yield Point(start), Point(end)


def to_corners(geom: Polygon) -> list[Position]:
    """Convert the corners of a polygon to a list of Positions.

    Args:
        geom (Polygon): The polygon to convert.

    Returns:
        list[Position]: A list of Positions representing the corners of the polygon.
    """
    return [Position(x=pt[0], y=pt[1]) for pt in geom.exterior.coords]


def to_walls(geom: MultiLineString | LineString | Polygon, material: MaterialIdentifier | None = None) -> list[Wall]:
    """Convert a geometry to a list of Wall segments.

    Args:
        geom (MultiLineString | LineString | Polygon): The geometry to convert.
        material (MaterialIdentifier | None): Optional material applied to every wall segment.

    Returns:
        list[Wall]: A list of Wall segments representing the geometry.
    """
    return [Wall(start=Position(x=start.x, y=start.y), end=Position(x=end.x, y=end.y), material=material) for (start, end) in line_pairs(geom)]
