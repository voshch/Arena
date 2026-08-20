"""Grid sketches: box-drawing glyphs declare their own edges, compiled through the layout IR."""

import typing

import attrs
import pydantic
import shapely
import yaml

from . import LevelDescription, PedestrianConfiguration, WithPedestrians, alphabet
from .layout import Area, GridFrame, Layout, Note, Region, compile_layout

DEFAULT_SKETCH = '┌─┐\n│ │\n└─┘'

Key = tuple[int, int]


Weight = typing.Literal['light', 'heavy', 'double', 'full']

WEIGHTS = {'light': alphabet.LIGHT, 'heavy': alphabet.HEAVY, 'double': alphabet.DOUBLE, 'full': alphabet.FULL}


class Symbol(pydantic.BaseModel):
    """What a character means beyond the alphabet, including arms no glyph can spell."""

    width: float | None = None
    weight: Weight | None = None
    fill: bool = False
    material: str | None = None
    zone: str | None = None
    sockets: list[str] | dict[str, Weight] | None = None

    @pydantic.field_validator('sockets')
    @classmethod
    def _known_directions(cls, sockets: 'list[str] | dict[str, Weight] | None') -> 'list[str] | dict[str, Weight] | None':
        unknown = [socket for socket in sockets or () if socket not in alphabet.DIRECTIONS]
        if unknown:
            raise ValueError(f'unknown socket {unknown[0]!r}, expected one of {" ".join(alphabet.DIRECTIONS)}')
        return sockets

    def socket_weights(self, default: int) -> dict[str, int]:
        if isinstance(self.sockets, dict):
            return {socket: WEIGHTS[weight] for socket, weight in self.sockets.items()}
        return dict.fromkeys(self.sockets or (), default)


class Directives(pydantic.BaseModel):
    cell: float | None = None
    light: float | None = None
    heavy: float | None = None
    double: float | None = None
    legend: dict[str, Symbol] = pydantic.Field(default_factory=dict)


@attrs.define
class Cell:
    row: int
    col: int
    arms: alphabet.Arms
    symbol: Symbol | None = None
    declares: bool = True


def split_sketch(text: str) -> tuple[Directives, list[str]]:
    """Separate the `!`-prefixed directive block from the grid rows, padded to a rectangle."""
    directives: list[str] = []
    rows: list[str] = []
    for line in text.splitlines():
        if line.startswith('!'):
            directives.append(line[1:])
        else:
            rows.append(line)
    parsed = yaml.safe_load('\n'.join(directives)) if directives else None
    parsed = {key: value for key, value in (parsed or {}).items() if value is not None}
    # Blank rows are kept, as blank columns always were: an editor addressing cell (row, col)
    # needs the grid it drew, and a row carrying nothing carries nothing either way.
    width = max((len(row) for row in rows), default=0)
    return Directives.model_validate(parsed), [row.ljust(width) for row in rows]


def normalize(text: str) -> str:
    """Rewrite a sketch into canonical glyphs, leaving directives and legend symbols alone."""
    directives, _ = split_sketch(text)
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith('!'):
            lines.append(line)
            continue
        chars = []
        for char in line:
            canonical = None if char in directives.legend else alphabet.normalize(char)
            chars.append(char if canonical is None else canonical)
        lines.append(''.join(chars))
    return '\n'.join(lines)


class WorldGeneratorSketch(WithPedestrians):
    class Configuration(PedestrianConfiguration):
        sketch: str = pydantic.Field(DEFAULT_SKETCH, json_schema_extra={"widget": "sketch"})
        cell: float = 8.0
        light: float = 1.5
        heavy: float = 3.0
        double: float = 6.0

    config: Configuration
    _frame: GridFrame | None

    def configure(self, configuration: dict):
        self.config = self.Configuration.model_validate(configuration)
        self.warnings = []
        self.diagnostics = None
        self._frame = None

    def normalize(self, source: str) -> str:
        return normalize(source)

    def frame(self) -> GridFrame | None:
        return self._frame

    def _metres(self, weight: int) -> float:
        return {
            alphabet.LIGHT: self.config.light,
            alphabet.HEAVY: self.config.heavy,
            alphabet.DOUBLE: self.config.double,
            alphabet.FULL: self.config.cell,
        }[weight]

    def _width(self, cell: Cell, weight: int) -> float:
        if cell.symbol is not None and cell.symbol.width is not None:
            return cell.symbol.width
        return self._metres(weight)

    def _cells(self, directives: Directives, rows: list[str]) -> dict[Key, Cell]:
        cells: dict[Key, Cell] = {}
        for row, line in enumerate(rows):
            if '\t' in line:
                raise ValueError(f'row {row + 1} contains a tab, which breaks the column grid')
            for col, char in enumerate(line):
                symbol = directives.legend.get(char)
                if symbol is not None:
                    default = alphabet.FULL if symbol.fill else WEIGHTS.get(symbol.weight or '', alphabet.LIGHT)
                    weights = symbol.socket_weights(default)
                    cells[row, col] = Cell(
                        row=row,
                        col=col,
                        arms=alphabet.arms(weights),
                        symbol=symbol,
                        declares=bool(weights),
                    )
                    continue
                arms = alphabet.arms_of(char)
                if arms is None:
                    raise ValueError(f'row {row + 1} column {col + 1}: {char!r} is not a corridor glyph and has no legend entry')
                if arms != alphabet.VOID:
                    cells[row, col] = Cell(row=row, col=col, arms=arms)
        return cells

    def _links(self, cells: dict[Key, Cell]) -> dict[tuple[Key, str], float]:
        """Link width per cell and direction. An arm stops at the boundary it shares, but a
        glyph with nothing to reach draws its whole cell rather than collapsing to a dot."""
        links: dict[tuple[Key, str], float] = {}
        for key, cell in cells.items():
            arms = [(index, direction) for index, direction in enumerate(alphabet.DIRECTIONS) if cell.arms[index] != alphabet.NONE]
            alone = all((cell.row + alphabet.OFFSETS[d][0], cell.col + alphabet.OFFSETS[d][1]) not in cells for _, d in arms)
            for index, direction in arms:
                width = self._width(cell, cell.arms[index])
                offset = alphabet.OFFSETS[direction]
                other_key = (cell.row + offset[0], cell.col + offset[1])
                other = cells.get(other_key)
                if other is None:
                    if alone:
                        links[key, direction] = width
                    continue
                back = other.arms[alphabet.DIRECTIONS.index(alphabet.OPPOSITE[direction])]
                if back == alphabet.NONE:
                    if other.declares:
                        self.warnings.append(
                            Note(row=cell.row, col=cell.col, text=f'{direction} link is declared on one side only')
                        )
                else:
                    width = max(width, self._width(other, back))
                links[other_key, alphabet.OPPOSITE[direction]] = width
                links[key, direction] = width
        return links

    def _centre(self, cell: Cell, rows: int, left: int) -> tuple[float, float]:
        size = self.config.cell
        return ((cell.col - left) * size + size / 2, (rows - 1 - cell.row) * size + size / 2)

    def _quad(self, centre: tuple[float, float], direction: str, width: float) -> shapely.Polygon:
        """Half-edge from a cell centre to the boundary it shares with its neighbour."""
        offset = alphabet.OFFSETS[direction]
        step = self.config.cell / 2
        delta = (offset[1] * step, -offset[0] * step)
        length = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
        unit = (delta[0] / length, delta[1] / length)
        perp = (-unit[1] * width / 2, unit[0] * width / 2)
        far = (centre[0] + delta[0], centre[1] + delta[1])
        return shapely.Polygon(
            [
                (centre[0] + perp[0], centre[1] + perp[1]),
                (far[0] + perp[0], far[1] + perp[1]),
                (far[0] - perp[0], far[1] - perp[1]),
                (centre[0] - perp[0], centre[1] - perp[1]),
            ]
        )

    def _footprint(self, cell: Cell, centre: tuple[float, float], widths: dict[str, float]) -> shapely.Polygon:
        """The cell's own ink: a box across its orthogonal arms, a diamond across its diagonal
        ones, so nothing juts out of a corridor at an angle to it."""
        own = 0.0
        if cell.symbol is not None and cell.symbol.fill:
            own = self.config.cell
        elif cell.symbol is not None and not cell.declares:
            # a symbol with no sockets is a room, and its width is how big the room is
            own = cell.symbol.width or 0.0
        vertical = max([own, *(widths[d] for d in ('N', 'S') if d in widths)])
        horizontal = max([own, *(widths[d] for d in ('E', 'W') if d in widths)])
        diagonal = max([0.0, *(widths[d] for d in ('NE', 'SE', 'SW', 'NW') if d in widths)])
        if vertical == 0.0 and horizontal == 0.0 and diagonal == 0.0:
            vertical = max(widths.values(), default=self._width(cell, alphabet.LIGHT))

        parts: list[shapely.Polygon] = []
        if vertical > 0.0 or horizontal > 0.0:
            vertical = vertical or horizontal
            horizontal = horizontal or vertical
            parts.append(
                shapely.box(centre[0] - vertical / 2, centre[1] - horizontal / 2, centre[0] + vertical / 2, centre[1] + horizontal / 2)
            )
        if diagonal > 0.0:
            reach = diagonal / 2**0.5  # the largest diamond that stays inside a band of this width
            parts.append(
                shapely.Polygon(
                    [
                        (centre[0] + reach, centre[1]),
                        (centre[0], centre[1] + reach),
                        (centre[0] - reach, centre[1]),
                        (centre[0], centre[1] - reach),
                    ]
                )
            )
        return shapely.union_all(parts)

    def _runs(
        self,
        cells: dict[Key, Cell],
        links: dict[tuple[Key, str], float],
        centres: dict[Key, tuple[float, float]],
        footprints: dict[Key, shapely.Polygon],
        forward: str,
        start: int,
    ) -> list[Region]:
        """Rectangular regions along maximal straight runs of equal width, so floors stay rectangles."""
        back = alphabet.OPPOSITE[forward]
        offset = alphabet.OFFSETS[forward]
        regions: list[Region] = []
        for key in cells:
            width = links.get((key, forward))
            if width is None:
                continue
            previous = links.get((key, back))
            if previous is not None and abs(previous - width) < 1e-9:
                continue
            span = [key]
            current = key
            while True:
                nxt = (current[0] + offset[0], current[1] + offset[1])
                if nxt not in cells or abs(links.get((current, forward), -1.0) - width) > 1e-9:
                    break
                span.append(nxt)
                current = nxt
            boxes = [footprints[member] for member in span]
            boxes.extend(self._quad(centres[member], forward, width) for member in span if (member, forward) in links)
            regions.append(Region(name=f'corridor_{start + len(regions)}', polygon=shapely.union_all(boxes).envelope))
        return regions

    def compute(self) -> LevelDescription:
        directives, rows = split_sketch(self.config.sketch)
        if directives.cell is not None:
            self.config.cell = directives.cell
        if directives.light is not None:
            self.config.light = directives.light
        if directives.heavy is not None:
            self.config.heavy = directives.heavy
        if directives.double is not None:
            self.config.double = directives.double

        self.warnings = []
        cells = self._cells(directives, rows)
        if not cells:
            # A blank grid is a state every edit path can reach, so it still has to make a
            # world: one full cell at the origin, where the caret starts.
            rows = rows or ['']
            anchor: Key = (len(rows) - 1, 0)
            cells = {anchor: Cell(row=anchor[0], col=anchor[1], arms=alphabet.VOID, symbol=Symbol(fill=True), declares=False)}
            self.warnings.append(Note(row=anchor[0], col=anchor[1], text='sketch is empty, drew one full cell'))
        # The drawing sits on the cells it occupies, so moving the caret off it cannot move the
        # world. The frame shifts by the rows and columns left over instead, which keeps cell
        # (rows - 1, 0) on its own origin whatever blank space surrounds the ink.
        drawn = max(cell.row for cell in cells.values()) + 1
        left = min(cell.col for cell in cells.values())
        self._frame = GridFrame(
            origin=(-left * self.config.cell, -(len(rows) - drawn) * self.config.cell),
            pitch=self.config.cell,
            rows=len(rows),
            cols=max(1, *(len(row) for row in rows)),
        )

        links = self._links(cells)
        centres = {key: self._centre(cell, drawn, left) for key, cell in cells.items()}
        # A full arm's box already reaches every corner of its cell, so a diagonal quad or diamond
        # of that width could only jut past it. Full cells meet diagonally through their neighbours.
        widths = {
            key: {
                direction: links[key, direction]
                for index, direction in enumerate(alphabet.DIRECTIONS)
                if (key, direction) in links and not (index % 2 == 1 and cell.arms[index] == alphabet.FULL)
            }
            for key, cell in cells.items()
        }
        footprints = {key: self._footprint(cells[key], centres[key], widths[key]) for key in cells}

        areas = [Area(polygon=polygon) for polygon in footprints.values()]
        for key, cell_widths in widths.items():
            for direction, width in cell_widths.items():
                areas.append(Area(polygon=self._quad(centres[key], direction, width)))

        regions = self._runs(cells, links, centres, footprints, 'E', 1)
        regions += self._runs(cells, links, centres, footprints, 'S', len(regions) + 1)
        named: dict[str, list[shapely.Polygon]] = {}
        materials: dict[str, str | None] = {}
        for key, cell in cells.items():
            if cell.symbol is not None and cell.symbol.zone is not None:
                named.setdefault(cell.symbol.zone, []).append(footprints[key])
                materials[cell.symbol.zone] = cell.symbol.material
        for name, polygons in named.items():
            regions.insert(0, Region(name=name, polygon=shapely.union_all(polygons), material=materials[name]))

        level, diagnostics = compile_layout(Layout(areas=areas, regions=regions))
        self.diagnostics = diagnostics
        return level
