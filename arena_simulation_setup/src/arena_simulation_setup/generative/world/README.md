# Generative worlds

World generators build a one-level format 2 `World` from parameters. Every front end compiles to the same geometry IR, so a new one only says where its walls go.

The ROS node is [`world_generator_ros.py`](world_generator_ros.py), its interfaces are [`world_generator_msgs`](../../../../../utils/msgs/world_generator_msgs/README.md), and its front end is the World Generator panel in [`task_generator_gui`](../../../../../utils/task_generator_gui/).

## The registry

| Generator | Input | Output |
|---|---|---|
| `empty` | width, height | a bare walled box |
| `hallway` | corridor and room sizes | a spine with rooms either side |
| `barn` | box size, passage width, straightness, or `index` | a BARN-style winding passage, or with `index` (0..299) the canonical BARN challenge world of that number, read from `source` (a directory of `world_<index>.world` files) or fetched from GitHub into `ARENA_ASSETS_DIR/.import/barn/worlds`. Each cylinder is an octagon of walls whose outer faces lie on the 0.075 m radius, in the challenge frame shifted by +4.5 m in x. Either way the episode is pinned to `tm_robots: scenario` |
| `sketch` | a grid of box-drawing glyphs | corridors and rooms exactly as drawn |
| `letter` | text | one 18-segment character cell per glyph |
| `corridor` | extent, corridor width and its spread, branches, bends, wiggle and diagonal | a tree of corridors grown from the center with right-angle elbows, or with `bends` winding through kinks at 45 and 90 degrees (`diagonal` off: right angles only), branches under three corridor widths staying elbows |
| `open` | extent, obstacle count, size and its spread, `max_tries` | a walled box with square obstacles at random positions, merged where they overlap, redrawn until the free space is one piece. Zones are vertical bands around the obstacles |
| `concourse` | spine shape, branches, elevators, pillars, colonnades, crowd flows, robot mission | a procedural concourse corridor or bend with side branches, doorway rooms and elevators, featuring semantic dock sockets (`dock_start`, `dock_end`, `dock_branch_<idx>_<side>`, `dock_elevator_<idx>`) for crowd sinks and sources. Binds robots and obstacles to its generated scenario |
| `egobench` | one EgoBench sample: world type and its configuration as JSON, world seed, encounter, pedestrian kind (`fixed` or `reactive`), lateral offset, speeds, pedestrian profile | the `corridor` or `open` world with that sample's one-pedestrian encounter scenario, failing when the encounter does not fit. Pinned to `scenario` on both task axes. See the [EgoBench generator](../egobench/README.md) |
| `rooms` | rows, cols, room side min and max, wall, door width, `max_depth`, `deep`, `loops` | a grid of rooms with one `entrance` room in row 0, joined by doors along a random spanning tree grown from it, every room within `max_depth` doors (0 = unbounded); `deep` grows long chains, `loops` adds doors that close cycles. Each room is a zone, each door connects two in the KG. See the [USAR generator](../usar/README.md) |
| `sidewalk` | layout (`straight`, `curve`, `corner`, `intersection`, `crosswalk`, `narrow_passage`, `plaza`, `block`), route `length`, `blocks`, band widths (`sidewalk_width`, `road_width`, `curb_width`, `verge_width`, `frontage_width`), `obstacles`, `scenario` (`empty`, `default` or a behavior: `obstructing`, `conversing`, `queueing`, `frontal`, `lateral`, `overtaking`, `crossing`, `gesturing`), `pedestrians` | street bands around the axes: `road`, `curb`, `verge`, `sidewalk`, `frontage` and `crosswalk` zones (suffixed `_<i>` when a kind has several pieces), buildings behind the frontage as walls, the curb and road free space. The `default` scenario carries the robot from the sidewalk centerline start to its end (`block`: over 100 m through the grid), `obstacles` street furniture in verge and frontage, and pedestrians: `default` flows `pedestrians` adults (4 when unset), a behavior places its group of agent type `sidewalk_<behavior>` at fixed arc and lateral offsets from the start plus `pedestrians` background adults. Pinned to `scenario` on both task axes |

Registration in [`__init__.py`](__init__.py) is lazy, so importing the package does not import every generator. An implementation subclasses `WorldGeneratorImpl` with:

- a nested `Configuration` extending `BaseConfiguration`, or `PedestrianConfiguration` to inherit the pedestrian count,
- `configure(dict)` and `compute() -> World`, which sets `self.diagnostics` and appends to `self.warnings`,
- optionally `files()` for extra artifacts written into the world directory by `save_generated`, `params()` for the episode binding, and `normalize(str)` for a canonical form of a textual input.

`Configuration` fields become ROS parameters under `algorithm.<generator>.<field>` through [`schema.py`](schema.py). Only `bool`, `int`, `float`, `str` and 2-tuples of numbers are supported (structured list/tuple fields are skipped for the ROS parameter server), an `int | None` field needs a `ge` bound and is declared as an integer whose value one below it stands for None (the node maps it back with `config_from_params`, so `-1` is the procedural `barn`), `ge`/`le` become the parameter's `Range`, an enum field becomes `Choices`, a field's pydantic `description` is the parameter description, so every field carries one sentence with its unit and sentinels, and `json_schema_extra={"widget": "sketch"}` asks the panel for the grid editor.

## Layout IR

[`layout.py`](layout.py) is what every front end targets. Free space is declared, walls are derived.

| Piece | Meaning |
|---|---|
| `Segment` | a centreline edge carrying its own width, buffered with flat caps and mitred joins |
| `Area` | a filled polygon, unioned with the buffered segments |
| `Region` | a named area that names and materialises a zone, geometry comes from the union |

Doors have no IR form. Every generator on the IR draws corridors, where an opening is just more free space, so nothing has needed one. [`hallway.py`](hallway.py) cuts its own and is not on the IR.

`compile_layout` unions everything, takes the rings of the result as walls, clips regions to non-overlapping zones, covers the rest as `area_N`, and returns `Diagnostics` beside the level.

A `Zone` is a single ring and cannot express a hole, so a courtyard would render as free space under a wall outline. A part with holes is sliced into vertical bands at every hole edge instead: no band can contain a hole, and the cuts land on coordinates the geometry already has, which a hairline slit does not.

## Sketch grammar

A sketch is a grid of box-drawing glyphs. Each glyph declares the edges leaving its cell, and an edge becomes geometry only when the cell across it declares the same edge back, which is what keeps a drawn shape the size it looks: a run ends at the centre of its last cell, capped by its own width. A glyph with no neighbour to reach at all is the exception, and draws itself across its whole cell, so a lone `─` or `╱` is the line it looks like instead of a dot.

```
!cell: 8.0
!heavy: 3.0
!legend:
!  R: {fill: true, zone: hall}
┌─┬─┐
│ ┃ │
└─┴─┘
```

- **Glyphs.** [`alphabet.py`](alphabet.py) derives the table by parsing Unicode character names, so it cannot drift from the standard. A glyph is 8 arm weights in `N NE E SE S SW W NW` order: none, light, heavy, double, and `█` full. The mapping is reversible, which is how an editor picks the glyph for the arms it wants.
- **Shorthand.** `- | + / \ X x` stand in for the light glyphs, `#` and `%` for a full block, `.` and space for void. `normalize()` rewrites a sketch into canonical glyphs.
- **Diagonals.** Unicode carries `╱`, `╲` and `╳` only, so a cell can hold a diagonal that runs through it but never one that turns in it. A peak or a valley goes through `╳`, or through a minted symbol when you want no spare arms.
- **Beyond the alphabet.** Unicode spells all 15 orthogonal arm combinations at light and heavy, 11 at double, none at full, and only three diagonals in total, so most of the grammar has no character. The legend covers the rest: there is one rule, the alphabet's glyph when it has one and a minted legend entry when it does not. Minted symbols are deduplicated by arm vector, and dropped once nothing uses them. Only lines in the exact form the editor writes are swept, so a hand-written entry survives whether it is used or not.
- **Pasting.** The editor loads pasted text as a sketch rather than replaying it as keystrokes, canonicalising shorthand on the way in and refusing the paste outright if a character is neither a glyph nor declared by the pasted legend. A paste replaces the whole sketch, legend included.
- **Trimming.** Blank rows and columns are cut back to the drawing plus the caret after every edit, in the grid and in the legend alike. Parsing keeps whatever rows and columns it is given, so cell `(row, col)` means the same thing to the editor and to the geometry. A blank row carries nothing either way, and a leading one does not move the drawing.
- **Widths.** Weights map to metres via the `light`, `heavy` and `double` parameters, and a full block spans the whole `cell`. Every cell emits its own footprint plus a quad to each boundary it shares, and both sides of a boundary take the wider of the two, so a width change lands at the cell centre rather than on the seam. The footprint is a box across the cell's orthogonal arms and a diamond across its diagonal ones, so nothing juts out of a corridor running at an angle to the grid.
- **Directives.** Lines starting with `!` are a YAML block read before the grid, continuations included, since an indented line without one is grid content: `cell`, `light`, `heavy`, `double`, and a `legend`. A legend `Symbol` takes `width`, `weight`, `fill`, `material`, `zone` and `sockets`, and declares no edges unless `sockets` says so, so a room symbol grows no stubs. `weight` names a width in the same terms a glyph does, and `width` overrides it in metres. Without sockets the width is the room's extent instead. `sockets` is a list when one weight covers them all, or a mapping when it does not: `{sockets: {W: heavy, E: light}}` says what only the alphabet's 68 mixed-weight glyphs could otherwise say. Cells sharing a `zone` name become one region.

## Letter grammar

[`letter.py`](letter.py) puts each character on a 3 x 5 node lattice and lights a subset of 18 segments, the way a segment display does. Every character occupies the same cell, so the result is monospace by construction rather than by rounding a font metric.

Segments carry the usual display names (`a1 a2 f i b g1 g2 e l c d1 d2`, plus diagonals `h k n m`) and two the standard set lacks, `p` and `q`, which run from the middle corners to the foot. Without those a V is indistinguishable from a Y, since every standard diagonal meets at the centre node. A glyph is one line:

```python
'Ж': 'h k i l n m'          # the diagonals plus the centre stem
'Ш': 'f e i l b c d1 d2'    # three verticals on a bottom bar
```

A character with no entry is painted from a font instead, stretched to the same cell so the pitch holds, which is how coverage reaches the rest of Unicode, and it is still reported by position. `FONTS` is tried in order, DejaVu for Latin, Greek, Cyrillic and symbols and Droid for CJK, and a font that answers with its own missing-glyph box counts as not having the character. Painting fills the cell in both axes, so a character whose ink is narrow, a comma or an apostrophe, arrives as a filled cell rather than a mark hanging in space: a world has to stay connected to be worth walking. Tofu, the box-with-a-cross, is what remains when no font has the character at all. Adding one is a single entry in `GLYPHS`, and a test asserts no glyph names a segment that does not exist. Digits, A-Z and А-Я ship. `cell` is the character height, `ratio` its width and `stroke` the corridor width. Cells abut, so neighbouring glyphs connect where their ink actually meets rather than through any synthetic link. `1` and `|` carry ink only on the centre column and therefore stand alone, and `Ы` is two strokes as the letter is.

## ROS surface

`world_generator` declares `generator` (a world generator name), `seed` (negative = nondeterministic), `world` (name of the world it writes) and every generator's configuration, then serves:

- `<node>/generate` (`GenerateWorld`) for preview and save. Both run the same path, so what the panel shows is what a save writes. Resolution 0 fits the map to `PREVIEW_PIXEL_BUDGET`. The response carries the map's world frame and, from `frame()`, the generator's cell grid, so an editor can place a cursor on the returned image.
- `<node>/generate_world` (`Trigger`), the older save-only entry that reads everything from parameters.
- `<node>/alphabet` (`Alphabet`, latched) so editors never hardcode the glyph table.

`params()` is the episode binding applied when the panel queues a generated world: `modes` (axis to mode names) plus `params` (axis to leaf name to value, the leaves of that axis's mode, so a binding never spells a param namespace). `barn` pins robots to its scenario. `sketch` and `letter` bind the obstacles axis to `random` with `dynamic.n` set to the pedestrian count, and `-1` returns an empty binding and leaves the episode alone.

## MCP domain

The `worldgen` domain of the Arena MCP server ([`mcp/worldgen.py`](../../mcp/worldgen.py)) runs the registry in process, so previews and saves need no world_generator node. `python3 -m arena_mcp ls worldgen` lists its tools.

- `worldgen_generators` reads every registered generator from `WorldGenerator.available()`, with its `config_model` as JSON schema and the episode binding of its defaults.
- `worldgen_preview` takes a generator, a configuration, an optional sketch and a seed (default 0, negative for a nondeterministic run), and answers what `<node>/generate` answers: diagnostics, warnings by sketch row and column, the normalized sketch, the episode binding and the rendered map at the node's default preview resolution (`preview_resolution` in `tree/World/Map.py`).
- `worldgen_save` writes through `WorldIdentifier(name).resolve_write_sync()` and `save_generated`, as the node does, so a generated world lands in the writable world root. It refuses an existing world unless `overwrite` is set, then saves over it in place, keeping its other files.
- `worldgen_alphabet` and `worldgen_normalize` serve the table the `alphabet` topic publishes and the canonical form of a sketch.
- `worldgen_queue` stages a saved world on an env through `config/queue_episode` with the binding `worldgen_save` returned, as the panel does. It prefixes each axis's leaves with the namespace of that axis's mode from the env's `query/task_modes` (`namespaces`, keyed `<axis>/<mode>`). The next episode reset loads the world.
