import io
import logging
import math
import typing
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageDraw
import shapely
import shapely.affinity
import yaml

from arena_simulation_setup.tree import PathView


class Map(PathView):
    @property
    def map_yaml(self) -> Path:
        return self.path / 'map.yaml'

    @property
    def map_png(self) -> Path:
        return self.path / 'map.png'

    @classmethod
    def _render_image(
        cls, rooms: shapely.MultiPolygon, doors: shapely.MultiPolygon, walls: shapely.MultiLineString, resolution: float = 0.01, padding: int = 5, *, static_objects: Iterable[tuple[str, shapely.Polygon]] = (), asset_color: str | None = "grey", asset_name_color: str | None = "blue"
    ) -> tuple[PIL.Image.Image, tuple[float, float]]:
        min_x, min_y, max_x, max_y = rooms.bounds
        if not doors.is_empty:
            dmin_x, dmin_y, dmax_x, dmax_y = doors.bounds
            min_x = min(min_x, dmin_x)
            min_y = min(min_y, dmin_y)
            max_x = max(max_x, dmax_x)
            max_y = max(max_y, dmax_y)

        width = max_x - min_x
        height = max_y - min_y

        img = PIL.Image.new(
            'RGB',
            (
                math.ceil(width / resolution) + 2 * padding,
                math.ceil(height / resolution) + 2 * padding,
            ),
            color='black',
        )

        scaling_factor = 1 / resolution

        # World metres to pixels, y flipped: the three steps compose into one matrix. A whole
        # collection goes through in one crossing into shapely, where per part the crossing costs
        # far more than the arithmetic.
        to_pixels = [
            scaling_factor,
            0.0,
            0.0,
            -scaling_factor,
            -scaling_factor * min_x,
            scaling_factor * (min_y + height),
        ]

        def to_pixel_parts(shape: shapely.Geometry) -> object:
            return shapely.get_parts(shapely.affinity.affine_transform(shape, to_pixels))

        def tf(shape: shapely.Geometry) -> shapely.Geometry:
            """Cleanup stays per part: snapping a whole collection at once lets neighbouring
            zones merge back into one ring, and a filled exterior would swallow the island."""
            shape = shapely.set_precision(shape, 0.01)
            shape = shapely.make_valid(shape)
            shape = shapely.remove_repeated_points(shape)
            return shape

        def as_int(coords: object) -> list[tuple[int, int]]:
            return [(int(math.trunc(x) + padding), int(math.trunc(y) + padding)) for (x, y, *_) in coords]

        draw = PIL.ImageDraw.Draw(img)
        for cutout in to_pixel_parts(rooms):
            poly = tf(cutout)
            draw.polygon(as_int(poly.exterior.coords), fill='white')

        for wall in to_pixel_parts(walls):
            line = tf(wall)
            draw.line(as_int(line.coords), fill='black', width=1)

        for cutout in to_pixel_parts(doors):
            poly = tf(cutout)
            draw.polygon(as_int(poly.exterior.coords), fill='white')

        if asset_color is not None:
            for name, obj in static_objects:
                logging.debug(f"Drawing asset '{name}' with geometry: {obj} in color {asset_color}")
                poly = tf(shapely.affinity.affine_transform(obj, to_pixels))
                if len(poly.exterior.coords) < 3:
                    logging.warning(f"Skipping asset '{name}' because it has insufficient geometry to draw ({len(poly.exterior.coords)} coordinates).")
                    continue
                draw.polygon(as_int(poly.exterior.coords), fill=asset_color)
                if asset_name_color is not None:
                    _min_x, _min_y, _max_x, _max_y = poly.bounds
                    logging.debug(f"Drawing name for asset '{name}' at ({int(_max_x)}, {int(_max_y)}) color {asset_name_color}")
                    draw.text((int(_max_x), int(_max_y)), name, fill=asset_name_color)

        return img, (min_x - padding * resolution, min_y - padding * resolution)

    @classmethod
    def generate_png(
        cls, rooms: shapely.MultiPolygon, doors: shapely.MultiPolygon, walls: shapely.MultiLineString, resolution: float = 0.01, padding: int = 5, *, static_objects: Iterable[tuple[str, shapely.Polygon]] = (), asset_color: str | None = "grey", asset_name_color: str | None = "blue"
    ) -> tuple[bytes, tuple[float, float]]:
        """
        Generate a PNG image of the map with the given elements.

        Args:
            rooms (shapely.MultiPolygon): MultiPolygon representing the rooms in the map.
            doors (shapely.MultiPolygon): MultiPolygon representing the doors in the map.
            walls (shapely.MultiLineString): MultiLineString representing the walls in the map.
            resolution (float): Size of each pixel in meters.
            padding (int): Number of pixels to pad around the map.
            show_obj_name (bool): Whether to display object names on the map.
            static_objects (Optional[List[Tuple[str, shapely.Polygon]]]): Optional list of (name, Polygon) tuples for static objects to draw.
            asset_color (str | None): Color used to fill static objects.
            asset_name_color (str | None): Color used for static object names.
        """
        img, origin = cls._render_image(
            rooms,
            doors,
            walls,
            resolution=resolution,
            padding=padding,
            static_objects=static_objects,
            asset_color=asset_color,
            asset_name_color=asset_name_color,
        )

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue(), origin

    @classmethod
    def rasterize(
        cls, rooms: shapely.MultiPolygon, doors: shapely.MultiPolygon, walls: shapely.MultiLineString, resolution: float = 0.01, padding: int = 5, *, static_objects: Iterable[tuple[str, shapely.Polygon]] = (), asset_color: str | None = "grey", asset_name_color: str | None = "blue"
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Like `generate_png` but returns a uint8 grayscale array (255=free, 0=occupied)."""
        img, origin = cls._render_image(
            rooms,
            doors,
            walls,
            resolution=resolution,
            padding=padding,
            static_objects=static_objects,
            asset_color=asset_color,
            asset_name_color=asset_name_color,
        )
        return np.asarray(img.convert('L'), dtype=np.uint8), origin

    @classmethod
    def generate_map_yaml(cls, resolution: float, filename: str, origin: tuple[float, float]) -> str:
        return typing.cast(
            str,
            yaml.safe_dump(
                {
                    'free_thresh': 0.1,
                    'image': filename,
                    'negate': 0,
                    'occupied_thresh': 0.9,
                    'origin': [*origin, 0],
                    'resolution': resolution,
                }
            ),
        )
