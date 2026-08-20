"""Box-drawing glyphs as socket declarations: one arm weight per direction, looked up either way."""

import unicodedata

DIRECTIONS = ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW')

OFFSETS: dict[str, tuple[int, int]] = {
    'N': (-1, 0),
    'NE': (-1, 1),
    'E': (0, 1),
    'SE': (1, 1),
    'S': (1, 0),
    'SW': (1, -1),
    'W': (0, -1),
    'NW': (-1, -1),
}

OPPOSITE = {'N': 'S', 'NE': 'SW', 'E': 'W', 'SE': 'NW', 'S': 'N', 'SW': 'NE', 'W': 'E', 'NW': 'SE'}

NONE = 0
LIGHT = 1
HEAVY = 2
DOUBLE = 3
FULL = 4

Arms = tuple[int, ...]

VOID: Arms = (NONE,) * 8
FILLED: Arms = (FULL,) * 8

_WEIGHT_WORDS = {'LIGHT': LIGHT, 'SINGLE': LIGHT, 'HEAVY': HEAVY, 'DOUBLE': DOUBLE}
_DIRECTION_WORDS = {
    'UP': ('N',),
    'DOWN': ('S',),
    'LEFT': ('W',),
    'RIGHT': ('E',),
    'VERTICAL': ('N', 'S'),
    'HORIZONTAL': ('E', 'W'),
}

_DIAGONAL_GLYPHS = {
    '╱': ('NE', 'SW'),
    '╲': ('NW', 'SE'),
    '╳': ('NE', 'SW', 'NW', 'SE'),
}

FULL_BLOCK = '█'

ASCII_ALIASES: dict[str, tuple[str, ...]] = {
    '-': ('E', 'W'),
    '|': ('N', 'S'),
    '+': ('N', 'E', 'S', 'W'),
    '/': ('NE', 'SW'),
    '\\': ('NW', 'SE'),
    'X': ('NE', 'SW', 'NW', 'SE'),
    'x': ('NE', 'SW', 'NW', 'SE'),
}

FILL_ALIASES = ('#', '%')

VOID_CHARS = (' ', '.')


def arms(weights: dict[str, int]) -> Arms:
    """Build an arm vector from a direction to weight mapping."""
    return tuple(weights.get(direction, NONE) for direction in DIRECTIONS)


def _parse_name(name: str) -> Arms | None:
    """Read a Unicode box-drawing name as arm weights, or None when the name is not one."""
    weights: dict[str, int] = {}
    weight = NONE
    for clause in name.split(' AND '):
        words = clause.split()
        directions: list[str] = []
        for word in words:
            if word in _DIRECTION_WORDS:
                directions.extend(_DIRECTION_WORDS[word])
            elif word in _WEIGHT_WORDS:
                weight = _WEIGHT_WORDS[word]
            else:
                return None
        if not directions:
            return None
        for direction in directions:
            weights[direction] = weight
    if not weights or weight == NONE:
        return None
    return arms(weights)


def _build() -> dict[str, Arms]:
    table: dict[str, Arms] = {}
    for code in range(0x2500, 0x2580):
        glyph = chr(code)
        if glyph in _DIAGONAL_GLYPHS:
            table[glyph] = arms(dict.fromkeys(_DIAGONAL_GLYPHS[glyph], LIGHT))
            continue
        name = unicodedata.name(glyph, '')
        if not name.startswith('BOX DRAWINGS ') or 'ARC' in name or 'DASH' in name:
            continue
        parsed = _parse_name(name.removeprefix('BOX DRAWINGS '))
        if parsed is not None:
            table[glyph] = parsed
    table[FULL_BLOCK] = FILLED
    return table


GLYPHS: dict[str, Arms] = _build()

_BY_ARMS: dict[Arms, str] = {}
for _glyph, _arms in GLYPHS.items():
    _BY_ARMS.setdefault(_arms, _glyph)


def arms_of(glyph: str) -> Arms | None:
    """Arm weights a glyph declares, or None when it carries no geometry."""
    if glyph in VOID_CHARS:
        return VOID
    if glyph in GLYPHS:
        return GLYPHS[glyph]
    if glyph in FILL_ALIASES:
        return FILLED
    if glyph in ASCII_ALIASES:
        return arms(dict.fromkeys(ASCII_ALIASES[glyph], LIGHT))
    return None


def glyph_for(vector: Arms) -> str | None:
    """Canonical glyph for an arm vector, or None when the alphabet has no such glyph."""
    if vector == VOID:
        return ' '
    return _BY_ARMS.get(vector)


def normalize(glyph: str) -> str | None:
    """Canonical form of a single cell, or None when it is not a geometry character."""
    vector = arms_of(glyph)
    if vector is None:
        return None
    return glyph_for(vector)


def entries() -> list[tuple[str, Arms]]:
    """The full table, for publishing to clients that must not hardcode it."""
    return sorted(GLYPHS.items())
