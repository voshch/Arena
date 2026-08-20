# world_generator_msgs

rosidl interfaces for the world generator: one service that previews or saves a world, plus the box-drawing alphabet the sketch editor draws with.

The generator itself lives in [`arena_simulation_setup/utils/generative`](../../../arena_simulation_setup/src/arena_simulation_setup/utils/generative/README.md); the rviz front end is the World Generator panel in [`task_generator_gui`](../../task_generator_gui/).

`Alphabet` publishes the glyph table, the ASCII shorthand with its arms, and the characters that mean an empty cell, so an editor hardcodes none of them. `.` is one of the void characters: it clears a cell and is printed, so a client testing for whitespace would treat an erased cell as drawn.

The `GenerateWorld` response carries the map's world frame (`map_origin`, `map_resolution`) and, for a generator drawn on cells, the grid's (`grid_origin`, `grid_pitch`, `grid_size`). Together they place cell `(row, col)` on the returned image, which is what lets the panel draw a cursor on the preview. A zero `grid_pitch` means the generator has no grid.

## Services (`srv/`)

| File | Purpose |
|---|---|
| `GenerateWorld.srv` | Preview or save one world. `preview_only` renders the map without writing it, so a preview and a save cannot drift apart. Request: generator name, optional sketch, a JSON `config` overlay of that generator's parameters, render `resolution` (0 = fit `PREVIEW_PIXEL_BUDGET`). Response: `png`, the `normalized` source, diagnostics (`components`, `islands`, `zones`, `extent`, `compile_ms`), `warnings`, and the `episode_binding` JSON the panel applies when queueing the world. `include_alphabet` returns the glyph table in the same round trip instead of subscribing. |

## Messages (`msg/`)

| File | Purpose |
|---|---|
| `Alphabet.msg` | Every glyph the sketch grammar accepts, latched on `<node>/alphabet`. `aliases` carries each plain-ASCII stand-in with the arms it normalizes to, so a client can canonicalise pasted text without a round trip, and `void_chars` says which characters mean an empty cell. Editors read this instead of hardcoding a table, so the grammar has one definition. |
| `AlphabetEntry.msg` | One glyph and its `arms[8]`: edge weight per direction in N NE E SE S SW W NW order, 0 absent and 1..4 light, heavy, double, full. Reversible, so an editor can look up a glyph by the arms it wants. |
| `SketchWarning.msg` | Something the caller should know about one cell of the source, by zero-based `row` and `col`. |
