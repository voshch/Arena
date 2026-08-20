import random

import pytest
import shapely

from arena_simulation_setup.utils.generative.sketch import WorldGeneratorSketch, normalize, split_sketch


def build(sketch: str, **config) -> WorldGeneratorSketch:
    generator = WorldGeneratorSketch({'sketch': sketch, **config}, random.Random(0))
    generator.compute()
    return generator


def free_space(generator: WorldGeneratorSketch, level) -> shapely.Polygon | shapely.MultiPolygon:
    return shapely.union_all([shapely.Polygon([(corner.x, corner.y) for corner in zone.corners]) for zone in level.zones])


def test_directives_are_split_from_the_grid():
    directives, rows = split_sketch('!cell: 4.0\n!light: 1.0\n───\n')
    assert directives.cell == 4.0
    assert directives.light == 1.0
    assert rows == ['───']


def test_dashes_in_the_grid_are_not_a_separator():
    _, rows = split_sketch('───\n───')
    assert rows == ['───', '───']


def test_tabs_are_rejected():
    with pytest.raises(ValueError, match='tab'):
        build('─\t─')


def test_unknown_characters_are_rejected_with_a_position():
    with pytest.raises(ValueError, match='row 1 column 2'):
        build('─@─')


def test_letters_stay_reserved():
    with pytest.raises(ValueError, match='no legend entry'):
        build('W')


def test_straight_corridor_geometry():
    generator = build('───', cell=8.0, light=1.5)
    level = generator.compute()
    assert generator.diagnostics.components == 1
    assert free_space(generator, level).bounds == pytest.approx((3.25, 3.25, 20.75, 4.75))


def test_ring_is_one_component_with_an_island():
    generator = build('┌─┐\n│ │\n└─┘')
    generator.compute()
    assert generator.diagnostics.components == 1
    assert generator.diagnostics.islands == 1


def test_detached_cells_are_reported():
    generator = build('─── ───')
    generator.compute()
    assert generator.diagnostics.components == 2


def test_mixed_junction_box_takes_each_axis_from_its_arms():
    generator = build('  │\n ═╪═\n  │', cell=8.0, light=1.5, double=6.0)
    level = generator.compute()
    free = free_space(generator, level)
    assert free.intersection(shapely.box(0.0, 16.0, 40.0, 24.0)).bounds == pytest.approx((11.25, 16.0, 12.75, 20.75))
    assert free.intersection(shapely.box(0.0, 0.0, 2.0, 40.0)).bounds == pytest.approx((1.0, 9.0, 2.0, 15.0))


def test_one_sided_links_warn_but_still_connect():
    generator = build('─│')
    generator.compute()
    assert generator.diagnostics.components == 1
    assert any('one side only' in warning.text for warning in generator.warnings)


def test_legend_symbols_carry_width_and_zone():
    sketch = '!legend:\n!  R: {width: 5.0, zone: ward}\n─R─'
    generator = build(sketch, cell=8.0, light=1.5)
    level = generator.compute()
    assert 'ward' in [zone.name for zone in level.zones]
    assert free_space(generator, level).bounds == pytest.approx((3.25, 1.5, 20.75, 6.5))


def test_fill_symbols_merge_into_rooms():
    sketch = '!legend:\n!  R: {fill: true}\nRRR\nRRR'
    generator = build(sketch, cell=4.0)
    level = generator.compute()
    assert generator.diagnostics.components == 1
    assert free_space(generator, level).area == pytest.approx(12.0 * 8.0)


def test_a_trailing_blank_row_does_not_move_the_drawing():
    """Parking the caret below the ink pads the grid, and that must not shift the world."""
    plain = free_space(build('──'), build('──').compute())
    padded = free_space(build('──\n  '), build('──\n  ').compute())
    assert plain.equals(padded)
    assert build('──\n  ').frame().rows == 2


def test_zones_are_rectangles_and_tile_the_free_space():
    generator = build('┌─┐\n│ │\n└─┘')
    level = generator.compute()
    covered = free_space(generator, level)
    for zone in level.zones:
        polygon = shapely.Polygon([(corner.x, corner.y) for corner in zone.corners])
        assert polygon.area == pytest.approx(polygon.envelope.area)
    assert covered.symmetric_difference(shapely.union_all([shapely.Polygon([(c.x, c.y) for c in z.corners]) for z in level.zones])).area == 0.0


def test_normalize_maps_ascii_to_canonical_glyphs():
    assert normalize('-|+/\\X') == '─│┼╱╲╳'


def test_normalize_is_position_preserving():
    text = '!cell: 4.0\n-- -\n |'
    result = normalize(text)
    assert [len(line) for line in result.splitlines()] == [len(line) for line in text.splitlines()]
    assert result.splitlines()[0] == '!cell: 4.0'


def test_normalize_leaves_legend_symbols_alone():
    text = '!legend:\n!  x: {fill: true}\nx-x'
    assert normalize(text).splitlines()[-1] == 'x─x'


def test_pedestrian_count_binds_random_dynamic_obstacles():
    assert build('──', pedestrians=7).params() == {
        'tm_obstacles': 'random',
        'obstacles_params': {'dynamic.n': [7, 7]},
    }
    assert build('──').params() == {}


def test_a_legend_symbol_can_socket_all_eight_directions():
    generator = build('!legend:\n!  O: {sockets: [N, NE, E, SE, S, SW, W, NW], width: 3.0}\n╲│╱\n─O─\n╱│╲', cell=8.0, light=3.0)
    assert generator.diagnostics.components == 1
    assert generator.warnings == []


def test_a_directive_without_a_value_is_ignored():
    assert build('!cell:\n──').diagnostics.components == 1


def ink(sketch: str, **config) -> shapely.Polygon | shapely.MultiPolygon:
    generator = WorldGeneratorSketch({'sketch': sketch, 'cell': 8.0, 'light': 1.5, **config}, random.Random(0))
    return free_space(generator, generator.compute())


def test_a_glyph_with_nothing_to_reach_draws_itself_in_full():
    """A lone glyph is the line it looks like, edge to edge, not a dot at its centre."""
    assert ink('─').bounds == pytest.approx((0.0, 3.25, 8.0, 4.75))
    assert ink('│').bounds == pytest.approx((3.25, 0.0, 4.75, 8.0))
    assert ink('╱').area == pytest.approx(8.0 * 2**0.5 * 1.5)


def test_a_diagonal_run_is_one_clean_band():
    """Nothing juts out sideways where diagonal cells meet: the run is exactly its own band."""
    for sketch in (' ╱\n╱ ', '   ╱\n  ╱ \n ╱  \n╱   ', '╲  \n ╲ \n  ╲'):
        band = ink(sketch)
        assert band.area == pytest.approx(band.minimum_rotated_rectangle.area), sketch


def test_a_diagonal_turn_needs_the_crossing_glyph():
    """No glyph carries two adjacent diagonals, so a peak turns through ╳ or not at all."""
    assert build(' ╱╲ \n╱  ╲').diagnostics.components == 2
    assert build(' ╳ \n╱ ╲').diagnostics.components == 1
    assert build('╲ ╱\n ╳ ').diagnostics.components == 1


def test_the_link_rule_the_canvas_mirrors():
    """`SketchCanvas::inkPath` paints the same three rules so a stroke shows before the render
    lands. Change any of them and the canvas drifts from what a generate would write."""
    # a glyph with nothing to reach spans its whole cell, edge to edge
    assert ink('─').bounds[0::2] == pytest.approx((0.0, 8.0))
    assert ink('─').area == pytest.approx(8.0 * 1.5)
    # in a run, the arm pointing off the drawing is dropped and the ink stops at the footprint
    assert ink('──').bounds[0::2] == pytest.approx((3.25, 12.75))
    # both sides of a boundary take the wider of the two, whichever side it is on
    assert ink('━─').area == pytest.approx(ink('─━').area)
    assert ink('━─').bounds == ink('─━').bounds


def test_a_legend_symbol_spells_sockets_the_alphabet_cannot():
    """Unicode has no glyph for a diagonal turn, so the legend carries one instead."""
    generator = build('!legend:\n!  a: {sockets: [SW, SE]}\n  a  \n ╱ ╲ \n╱   ╲')
    assert generator.diagnostics.components == 1
    assert generator.warnings == []


def test_a_legend_symbol_takes_its_width_from_a_weight():
    narrow = ink('!legend:\n!  a: {sockets: [E, W], weight: light}\n─a─')
    heavy = ink('!legend:\n!  a: {sockets: [E, W], weight: heavy}\n━a━', heavy=3.0)
    full = ink('!legend:\n!  a: {sockets: [E, W], weight: full}\n─a─')
    assert heavy.area > narrow.area
    assert full.intersection(shapely.box(8.0, 0.0, 16.0, 8.0)).area == pytest.approx(8.0 * 8.0)


def test_a_socketed_symbol_does_not_jut_out_of_its_own_diagonal():
    band = ink('!legend:\n!  a: {sockets: [NE, SW], weight: light}\n a\na ')
    assert band.area == pytest.approx(band.minimum_rotated_rectangle.area)


def test_unknown_sockets_are_rejected():
    with pytest.raises(ValueError, match='unknown socket'):
        build('!legend:\n!  a: {sockets: [NORTH]}\na')


def test_the_minted_legend_line_is_the_one_the_editor_writes():
    """Pins the format sketch_edit.cpp emits for an arm vector the alphabet cannot spell."""
    generator = build('!legend:\n!  a: {sockets: [NE, SW], weight: heavy}\n a\na ', cell=8.0, heavy=3.0)
    assert generator.diagnostics.components == 1
    assert generator.warnings == []


def test_the_grid_frame_locates_cells_in_world_metres():
    """Cell (rows - 1, 0) sits at the origin, so an editor can place a cursor on the rendered map."""
    frame = build('┌─┐\n│ │\n└─┘', cell=8.0).frame()
    assert (frame.origin, frame.pitch, frame.rows, frame.cols) == ((0.0, 0.0), 8.0, 3, 3)


def test_the_frame_counts_every_row_the_editor_holds():
    """Blank rows survive like blank columns, so a caret off the ink still lands inside the frame."""
    assert build('──\n  ', cell=8.0).frame().rows == 2
    assert build('  \n──', cell=8.0).frame().rows == 2
    assert build('──  ', cell=8.0).frame().cols == 4


def test_a_leading_blank_row_does_not_move_the_drawing():
    assert ink('──').equals(ink('  \n──'))


def test_blank_space_on_any_side_leaves_the_drawing_where_it_is():
    """The caret can walk off the ink in all four directions and the grid pads to follow it.
    Whichever side that padding lands on, the world it describes must not move."""
    for padded in ('  \n──', '──\n  ', ' ──', '──  ', ' ──\n   '):
        assert ink('──').equals(ink(padded)), padded


def test_the_frame_carries_the_blank_space_the_drawing_does_not():
    """Padding shifts the frame, never the geometry, so cell (rows - 1, 0) stays on its origin."""
    assert build(' ──', cell=8.0).frame().origin == (-8.0, 0.0)
    assert build('──\n  ', cell=8.0).frame().origin == (0.0, -8.0)
    assert build('  \n──', cell=8.0).frame().origin == (0.0, 0.0)


def test_a_symbol_can_carry_one_weight_per_socket():
    """The alphabet's 68 mixed-weight glyphs are the only other way to say this."""
    generator = build('!legend:\n!  a: {sockets: {W: heavy, E: light}}\n━a─', cell=8.0, light=1.5, heavy=3.0)
    assert generator.diagnostics.components == 1
    assert generator.warnings == []

    free = ink('!legend:\n!  a: {sockets: {W: heavy, E: light}}\n━a─', heavy=3.0)
    assert free.intersection(shapely.box(0.0, 0.0, 8.0, 16.0)).bounds[3] - free.bounds[1] == pytest.approx(3.0)
    assert free.intersection(shapely.box(16.0, 0.0, 24.0, 16.0)).bounds[3] - free.bounds[1] == pytest.approx(2.25)


def test_a_socket_mapping_rejects_unknown_directions():
    with pytest.raises(ValueError, match='unknown socket'):
        build('!legend:\n!  a: {sockets: {NORTH: light}}\na')


def test_the_minted_mixed_weight_line_is_the_one_the_editor_writes():
    """Pins the mapping form sketch_edit.cpp emits when a cell's arms do not share a weight."""
    generator = build('!legend:\n!  a: {sockets: {NE: heavy, SW: light}}\n a\na ', cell=8.0, heavy=3.0)
    assert generator.diagnostics.components == 1
    assert generator.warnings == []


def test_a_blank_sketch_still_makes_a_world():
    """Every edit path can reach a blank grid, so it draws one full cell at the origin instead of failing."""
    for sketch in ('', '   ', '!light: 2\n', '  \n  '):
        generator = WorldGeneratorSketch({'sketch': sketch, 'cell': 8.0}, random.Random(0))
        level = generator.compute()
        assert free_space(generator, level).bounds == pytest.approx((0.0, 0.0, 8.0, 8.0))
        assert generator.frame().rows >= 1 and generator.frame().cols >= 1
        assert [note.text for note in generator.warnings] == ['sketch is empty, drew one full cell']


def test_full_blocks_stay_inside_their_cells():
    """A full arm fills to the cell edge and no further, so blocks make rectangles, not stars."""
    assert ink('█').bounds == pytest.approx((0.0, 0.0, 8.0, 8.0))
    assert ink('██\n██').bounds == pytest.approx((0.0, 0.0, 16.0, 16.0))
    assert ink('██\n██').area == pytest.approx(16.0 * 16.0)
