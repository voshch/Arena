import pytest

from arena_simulation_setup.utils.generative import alphabet


def test_light_glyphs_declare_their_own_edges():
    assert alphabet.arms_of('─') == alphabet.arms({'E': alphabet.LIGHT, 'W': alphabet.LIGHT})
    assert alphabet.arms_of('│') == alphabet.arms({'N': alphabet.LIGHT, 'S': alphabet.LIGHT})
    assert alphabet.arms_of('┌') == alphabet.arms({'S': alphabet.LIGHT, 'E': alphabet.LIGHT})
    assert alphabet.arms_of('┼') == alphabet.arms(dict.fromkeys(('N', 'E', 'S', 'W'), alphabet.LIGHT))


def test_weight_families():
    assert alphabet.arms_of('━') == alphabet.arms({'E': alphabet.HEAVY, 'W': alphabet.HEAVY})
    assert alphabet.arms_of('═') == alphabet.arms({'E': alphabet.DOUBLE, 'W': alphabet.DOUBLE})
    assert alphabet.arms_of('█') == alphabet.FILLED


def test_mixed_junctions_carry_per_arm_weight():
    assert alphabet.arms_of('╪') == alphabet.arms(
        {'N': alphabet.LIGHT, 'S': alphabet.LIGHT, 'E': alphabet.DOUBLE, 'W': alphabet.DOUBLE}
    )
    assert alphabet.arms_of('╛') == alphabet.arms({'N': alphabet.LIGHT, 'W': alphabet.DOUBLE})
    assert alphabet.arms_of('┿') == alphabet.arms(
        {'N': alphabet.LIGHT, 'S': alphabet.LIGHT, 'E': alphabet.HEAVY, 'W': alphabet.HEAVY}
    )


def test_diagonals_and_stubs():
    assert alphabet.arms_of('╱') == alphabet.arms({'NE': alphabet.LIGHT, 'SW': alphabet.LIGHT})
    assert alphabet.arms_of('╲') == alphabet.arms({'NW': alphabet.LIGHT, 'SE': alphabet.LIGHT})
    assert alphabet.arms_of('╳') == alphabet.arms(dict.fromkeys(('NE', 'SW', 'NW', 'SE'), alphabet.LIGHT))
    assert alphabet.arms_of('╴') == alphabet.arms({'W': alphabet.LIGHT})


def test_round_trip_is_identity_for_every_glyph():
    for glyph, vector in alphabet.entries():
        assert alphabet.glyph_for(vector) == glyph, glyph


def test_arm_vectors_are_unique():
    seen: dict[tuple[int, ...], str] = {}
    for glyph, vector in alphabet.entries():
        assert vector not in seen, f'{glyph} collides with {seen.get(vector)}'
        seen[vector] = glyph


def test_ascii_normalizes_to_canonical_glyphs():
    assert alphabet.normalize('-') == '─'
    assert alphabet.normalize('|') == '│'
    assert alphabet.normalize('+') == '┼'
    assert alphabet.normalize('/') == '╱'
    assert alphabet.normalize('\\') == '╲'
    assert alphabet.normalize('X') == '╳'
    assert alphabet.normalize('#') == '█'
    assert alphabet.normalize(' ') == ' '


def test_normalization_is_idempotent():
    for glyph in list(alphabet.GLYPHS) + list(alphabet.ASCII_ALIASES) + list(alphabet.FILL_ALIASES):
        once = alphabet.normalize(glyph)
        assert once is not None
        assert alphabet.normalize(once) == once


def test_unknown_characters_are_rejected():
    assert alphabet.arms_of('W') is None
    assert alphabet.arms_of('@') is None


@pytest.mark.parametrize(
    'vector',
    [
        alphabet.arms(dict.fromkeys(alphabet.DIRECTIONS, alphabet.LIGHT)),
        alphabet.arms({'NE': alphabet.HEAVY, 'SW': alphabet.HEAVY}),
    ],
)
def test_unsupported_combinations_have_no_glyph(vector):
    assert alphabet.glyph_for(vector) is None
