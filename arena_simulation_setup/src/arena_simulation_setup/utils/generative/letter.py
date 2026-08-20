"""Segment-display glyphs: an 18-segment cell per character, lit segments become corridors."""

import functools
import pathlib
import unicodedata

import pydantic
import shapely
import shapely.affinity

from . import LevelDescription, PedestrianConfiguration, WithPedestrians
from .layout import Area, Layout, Note, Segment, compile_layout

# Coverage the segment lattice cannot reach is painted from a font instead, the first one that
# has the character. DejaVu carries Latin, Greek, Cyrillic and symbols, Droid the CJK.
FONTS = (
    pathlib.Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    pathlib.Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'),
)
PAINT_HEIGHT = 128  # raster rows per character cell

Node = tuple[int, int]  # (row from the top, column), on a 3 x 5 lattice

SEGMENTS: dict[str, tuple[Node, Node]] = {
    'a1': ((0, 0), (0, 1)),
    'a2': ((0, 1), (0, 2)),
    'f': ((0, 0), (2, 0)),
    'i': ((0, 1), (2, 1)),
    'b': ((0, 2), (2, 2)),
    'g1': ((2, 0), (2, 1)),
    'g2': ((2, 1), (2, 2)),
    'e': ((2, 0), (4, 0)),
    'l': ((2, 1), (4, 1)),
    'c': ((2, 2), (4, 2)),
    'd1': ((4, 0), (4, 1)),
    'd2': ((4, 1), (4, 2)),
    'h': ((0, 0), (2, 1)),
    'k': ((0, 2), (2, 1)),
    'n': ((2, 1), (4, 0)),
    'm': ((2, 1), (4, 2)),
    'p': ((2, 0), (4, 1)),
    'q': ((2, 2), (4, 1)),
}

# what a font draws for a character it does not have, and what we draw when it cannot be painted
TOFU = 'a1 a2 f e b c d1 d2 h k n m'

# a permanently unassigned codepoint, so whatever the font draws for it is its missing-glyph box
NOTDEF = '﷐'


def _blank(character: str) -> bool:
    """Space of any width, and the formatting marks that are meant to be invisible."""
    return character.isspace() or unicodedata.category(character) in ('Zs', 'Zl', 'Zp', 'Cf')


@functools.lru_cache(maxsize=1024)
def _render(font_index: int, character: str) -> shapely.Polygon | shapely.MultiPolygon | None:
    """One font's ink for one character, in a unit box."""
    path = FONTS[font_index]
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(str(path), PAINT_HEIGHT) if path.exists() else ImageFont.load_default(PAINT_HEIGHT)
        image = Image.new('1', (PAINT_HEIGHT * 2, PAINT_HEIGHT * 2), 0)
        ImageDraw.Draw(image).text((image.width // 2, image.height // 2), character, fill=1, font=font, anchor='mm')
    except (ImportError, OSError, TypeError, ValueError):
        return None

    # One box per run of set pixels. The union carries holes, which the layout slits open later.
    runs: list[shapely.Polygon] = []
    pixels = image.load()
    for y in range(image.height):
        start = None
        for x in range(image.width + 1):
            on = x < image.width and pixels[x, y]
            if on and start is None:
                start = x
            elif not on and start is not None:
                runs.append(shapely.box(start, y, x, y + 1))
                start = None
    if not runs:
        return None

    ink = shapely.union_all(runs)
    minx, miny, maxx, maxy = ink.bounds
    if maxx - minx <= 0 or maxy - miny <= 0:
        return None
    # Stretched to the cell like a segment glyph, so painted and lit characters share a pitch.
    ink = shapely.affinity.translate(ink, -minx, -miny)
    ink = shapely.affinity.scale(ink, 1 / (maxx - minx), -1 / (maxy - miny), origin=(0.0, 0.0))
    return shapely.affinity.translate(ink, 0.0, 1.0).simplify(1.5 / PAINT_HEIGHT)


def paint(character: str) -> shapely.Polygon | shapely.MultiPolygon | None:
    """The character's ink as polygons in a unit box, or None when no font has a glyph for it.
    A font answers for a character it lacks with its own box, which is not coverage."""
    for font_index in range(len(FONTS)):
        ink = _render(font_index, character)
        notdef = _render(font_index, NOTDEF)
        if ink is not None and (notdef is None or not ink.equals(notdef)):
            return ink
    return None

GLYPHS: dict[str, str] = {
    ' ': '',
    '-': 'g1 g2',
    '|': 'i l',
    '+': 'g1 g2 i l',
    '/': 'k n',
    '\\': 'h m',
    '0': 'a1 a2 b c d1 d2 e f',
    '1': 'i l',
    '2': 'a1 a2 b g1 g2 e d1 d2',
    '3': 'a1 a2 b g2 c d1 d2',
    '4': 'f g1 g2 b c',
    '5': 'a1 a2 f g1 g2 c d1 d2',
    '6': 'a1 a2 f e g1 g2 c d1 d2',
    '7': 'a1 a2 b c',
    '8': 'a1 a2 b c d1 d2 e f g1 g2',
    '9': 'a1 a2 b f g1 g2 c d1 d2',
    'A': 'a1 a2 f e b c g1 g2',
    'B': 'a1 a2 i l g2 b c d1 d2',
    'C': 'a1 a2 f e d1 d2',
    'D': 'a1 a2 i l b c d1 d2',
    'E': 'a1 a2 f e g1 g2 d1 d2',
    'F': 'a1 a2 f e g1 g2',
    'G': 'a1 a2 f e d1 d2 c g2',
    'H': 'f e b c g1 g2',
    'I': 'a1 a2 i l d1 d2',
    'J': 'b c d1 d2 e',
    'K': 'f e g1 k m',
    'L': 'f e d1 d2',
    'M': 'f e b c h k',
    'N': 'f e b c h m',
    'O': 'a1 a2 b c d1 d2 e f',
    'P': 'a1 a2 b f e g1 g2',
    'Q': 'a1 a2 b c d1 d2 e f m',
    'R': 'a1 a2 b f e g1 g2 m',
    'S': 'a1 a2 f g1 g2 c d1 d2',
    'T': 'a1 a2 i l',
    'U': 'f e d1 d2 c b',
    'V': 'f p b q',
    'W': 'f e n m c b',
    'X': 'h k n m',
    'Y': 'h k l',
    'Z': 'a1 a2 k n d1 d2',
    'А': 'a1 a2 f e b c g1 g2',
    'Б': 'a1 a2 f e g1 g2 c d1 d2',
    'В': 'a1 a2 f e g1 g2 b c d1 d2',
    'Г': 'a1 a2 f e',
    'Д': 'a1 a2 i n m d1 d2',
    'Е': 'a1 a2 f e g1 g2 d1 d2',
    'Ж': 'h k i l n m',
    'З': 'a1 a2 b g1 g2 c d1 d2',
    'И': 'f e b c n k',
    'Й': 'f e b c n k a1 a2',
    'К': 'f e g1 k m',
    'Л': 'a2 b c i n',
    'М': 'f e b c h k',
    'Н': 'f e b c g1 g2',
    'О': 'a1 a2 b c d1 d2 e f',
    'П': 'a1 a2 f e b c',
    'Р': 'a1 a2 b f e g1 g2',
    'С': 'a1 a2 f e d1 d2',
    'Т': 'a1 a2 i l',
    'У': 'h k l',
    'Ф': 'a1 a2 b c d1 d2 e f i l',
    'Х': 'h k n m',
    'Ц': 'f e b c d1 d2',
    'Ч': 'f g1 g2 b c',
    'Ш': 'f e i l b c d1 d2',
    'Щ': 'f e i l b c d1 d2 g2',
    'Ъ': 'a1 i l g2 c d2',
    'Ы': 'f e g1 l d1 b c',
    'Ь': 'f e g1 g2 c d1 d2',
    'Э': 'a1 a2 b g2 c d1 d2',
    'Ю': 'f e g1 a2 b c d2 i l',
    'Я': 'a1 a2 f g1 g2 b c n',
}


class WorldGeneratorLetter(WithPedestrians):
    class Configuration(PedestrianConfiguration):
        text: str = pydantic.Field('A', json_schema_extra={'widget': 'text'})
        cell: float = 20.0
        ratio: float = 0.62
        stroke: float = 3.0

    config: Configuration

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        self.warnings = []
        self.diagnostics = None

    def _origin(self, row: int, col: int, rows: int) -> tuple[float, float]:
        """Bottom-left corner of the character cell at grid position (row, col), cells abut."""
        return col * self.config.cell * self.config.ratio, (rows - 1 - row) * self.config.cell

    def _point(self, origin: tuple[float, float], node: Node) -> tuple[float, float]:
        width = self.config.cell * self.config.ratio
        return origin[0] + node[1] * width / 2, origin[1] + (4 - node[0]) * self.config.cell / 4

    def compute(self) -> LevelDescription:
        # Composed form, and cased per character: 'ß'.upper() is two characters, which would put
        # every warning after it on the wrong column.
        lines = unicodedata.normalize('NFC', self.config.text).splitlines() or ['']
        layout = Layout()
        drawn: dict[tuple[int, int], list[tuple[float, float]]] = {}

        for row, line in enumerate(lines):
            for col, character in enumerate(line):
                origin = self._origin(row, col, len(lines))
                lit = GLYPHS.get(character, GLYPHS.get(character.upper()))
                if lit is None and _blank(character):
                    lit = ''
                if lit is None:
                    self.warnings.append(Note(row=row, col=col, text=f'no segment glyph for {character!r}'))
                    painted = paint(character)
                    if painted is None:
                        lit = TOFU
                    else:
                        width = self.config.cell * self.config.ratio
                        placed = shapely.affinity.scale(painted, width, self.config.cell, origin=(0.0, 0.0))
                        layout.areas.append(Area(polygon=shapely.affinity.translate(placed, *origin)))
                        continue
                nodes: set[tuple[float, float]] = set()
                for name in lit.split():
                    start, end = SEGMENTS[name]
                    a, b = self._point(origin, start), self._point(origin, end)
                    layout.segments.append(Segment(a=a, b=b, width=self.config.stroke))
                    nodes.update((a, b))
                drawn[row, col] = sorted(nodes)

        if not layout.segments and not layout.areas:
            raise ValueError(f'{self.config.text!r} draws nothing')

        joints = {point for nodes in drawn.values() for point in nodes}

        # flat caps notch where segments meet
        half = self.config.stroke / 2
        layout.areas.extend(Area(polygon=shapely.box(x - half, y - half, x + half, y + half)) for x, y in joints)

        level, diagnostics = compile_layout(layout)
        self.diagnostics = diagnostics
        return level
