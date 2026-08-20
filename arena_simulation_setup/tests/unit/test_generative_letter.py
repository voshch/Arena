"""Letter worlds: an 18-segment cell per character, lit segments become corridors."""

import random
import unicodedata

import pytest
import shapely

from arena_simulation_setup.utils.generative.letter import GLYPHS, SEGMENTS, WorldGeneratorLetter, paint


def build(text: str, **config) -> tuple[WorldGeneratorLetter, shapely.Polygon | shapely.MultiPolygon]:
    generator = WorldGeneratorLetter({'text': text, **config}, random.Random(0))
    level = generator.compute()
    free = shapely.union_all(
        [shapely.Polygon([(corner.x, corner.y) for corner in zone.corners]) for zone in level.zones]
    )
    return generator, free


def test_every_glyph_lights_known_segments():
    for character, lit in GLYPHS.items():
        assert set(lit.split()) <= set(SEGMENTS), character


def test_the_counter_of_an_o_is_not_free_space():
    generator, free = build('O')
    assert generator.diagnostics.islands == 1
    assert not free.contains(shapely.Point(free.centroid.x, free.centroid.y))


def test_glyphs_sit_on_a_monospace_grid():
    _, one = build('T', cell=20.0, ratio=0.6)
    _, two = build('TT', cell=20.0, ratio=0.6)
    assert two.bounds[2] - one.bounds[2] == pytest.approx(20.0 * 0.6)


def test_rows_stack_downwards():
    _, one = build('T', cell=20.0)
    _, two = build('T\nT', cell=20.0)
    assert (two.bounds[3] - two.bounds[1]) - (one.bounds[3] - one.bounds[1]) == pytest.approx(20.0)


def test_neighbouring_glyphs_touch():
    for text in ('AB', 'A\nB', 'AB\nCD', 'ARRIVEDERCI'):
        generator, _ = build(text)
        assert generator.diagnostics.components == 1, text


def test_a_space_separates():
    generator, _ = build('A A')
    assert generator.diagnostics.components == 2


def test_v_stands_on_one_foot():
    _, free = build('V', cell=20.0)
    minx, miny, maxx, maxy = free.bounds
    foot = free.intersection(shapely.box(minx, miny, maxx, miny + 0.5))
    assert len(shapely.get_parts(foot)) == 1
    assert foot.centroid.x == pytest.approx((minx + maxx) / 2, abs=0.3)


def test_cyrillic_is_distinct_from_its_latin_lookalikes():
    assert GLYPHS['Ф'] != GLYPHS['O']
    assert GLYPHS['Ш'] != GLYPHS['Ц']
    drawn = {}
    for character in ('Ф', 'Ш', 'Ж', 'Д'):
        generator, free = build(character)
        assert generator.warnings == [], character
        drawn[character] = free
    for character, free in drawn.items():
        others = [other for name, other in drawn.items() if name != character]
        assert all(free.symmetric_difference(other).area > 0.0 for other in others), character


def test_lowercase_lights_the_same_segments():
    _, lower = build('ab')
    _, upper = build('AB')
    assert lower.symmetric_difference(upper).area == pytest.approx(0.0)


def test_stroke_sets_the_corridor_width():
    _, narrow = build('I', stroke=2.0)
    _, wide = build('I', stroke=4.0)
    assert wide.area > narrow.area


def test_unknown_characters_are_reported_by_position():
    generator, _ = build('A\nA§')
    assert [(note.row, note.col, note.text) for note in generator.warnings] == [
        (1, 1, "no segment glyph for '§'")
    ]


def test_text_that_draws_nothing_is_rejected():
    with pytest.raises(ValueError, match='draws nothing'):
        build('  ')


def test_pedestrian_count_binds_random_dynamic_obstacles():
    generator, _ = build('A', pedestrians=4)
    assert generator.params() == {'tm_obstacles': 'random', 'obstacles_params': {'dynamic.n': [4, 4]}}


def test_no_pedestrian_count_leaves_the_episode_alone():
    generator, _ = build('A')
    assert generator.params() == {}


def test_text_is_not_drawn_on_a_cell_grid():
    generator, _ = build('A')
    assert generator.frame() is None


def test_an_unknown_character_still_draws_something():
    """A font draws tofu for a character it lacks. A hole in the world reads as a mistake."""
    generator, free = build("IT'S")
    assert [note.col for note in generator.warnings] == [2]
    assert generator.diagnostics.components == 1
    assert free.area > build('ITS')[1].area


def test_any_character_the_font_has_is_painted():
    """Coverage past the lattice comes from painting the glyph, not from giving up on it."""
    for text in ('字', '=', 'é'):
        generator, free = build(text)
        assert [note.text for note in generator.warnings] == [f'no segment glyph for {text!r}']
        assert free.area > 0.0, text


def test_painting_keeps_what_sits_above_the_letter():
    """A diacritic is its own piece of ink, so it survives only if the glyph is not clipped to
    the raster. Painted ink is normalised to a unit box either way, so bounds cannot say this."""
    assert paint('E') is not None, 'no font to paint with'
    for bare, marked in (('E', 'É'), ('o', 'ö'), ('a', 'ä')):
        assert len(shapely.get_parts(paint(marked))) > len(shapely.get_parts(paint(bare))), marked


def test_a_counter_survives_painting():
    """Letters painted from a font enclose obstacles the same way lit segments do."""
    assert sum(len(part.interiors) for part in shapely.get_parts(paint('8'))) == 2


def test_a_font_without_the_character_is_not_coverage():
    """A font answers for what it lacks with its own box. Accepting it would put a rectangle in
    the world and call it a glyph, so these fall through to the segment lattice instead."""
    for character in ('\t', '\x1b', '﷐', '\U0001f1fa'):
        assert paint(character) is None, character


def test_blank_characters_of_any_width_stay_blank():
    """A pasted non-breaking or zero-width space is not an obstacle."""
    for text in ('A\xa0A', 'A​A', 'A　A'):
        generator, _ = build(text)
        assert generator.warnings == [], text
        assert generator.diagnostics.components == 2, text


def test_case_folding_never_moves_a_column():
    """'ß'.upper() is two characters, so folding the whole line would report the wrong column."""
    generator, _ = build('ßx§')
    assert [(note.row, note.col) for note in generator.warnings] == [(0, 0), (0, 2)]


def test_the_same_text_composed_either_way_draws_the_same_world():
    """Whether a client sends one codepoint or a letter plus a combining accent is not the
    user's choice, so it must not decide what gets built."""
    _, composed = build(unicodedata.normalize('NFC', '\u00e9'))
    _, decomposed = build(unicodedata.normalize('NFD', '\u00e9'))
    assert composed.equals(decomposed)
