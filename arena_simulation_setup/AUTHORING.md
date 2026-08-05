# Authoring a new world

End-to-end guide for creating a world that Arena can load. Covers the manual
path; see [scripts/generate_world](scripts/generate_world)
for the AI-assisted generation pipeline.

## Prerequisites

- Arena workspace built (`colcon build`).
- `arena_simulation_setup` on `$AMENT_PREFIX_PATH`.
- `touch_world` script available: `ros2 run arena_simulation_setup touch_world`.

## 1. Create the world directory

```bash
export WORLD=my_world
mkdir -p \
    $ARENA_DIR/arena_simulation_setup/worlds/$WORLD/map \
    $ARENA_DIR/arena_simulation_setup/worlds/$WORLD/scenarios/default \
    $ARENA_DIR/arena_simulation_setup/worlds/$WORLD/assets
```

`$ARENA_DIR` is the workspace source root. If unset, `touch_world` uses the
current working directory.

## 2. Author `world.yaml`

Create `worlds/$WORLD/world.yaml`. The minimum valid file is one zone with
corners and at least one wall:

```yaml
zones:
- name: main_room
  description: Main room
  material:
  - Concrete_Smooth
  - {}
  corners:
  - {x: 0.0, y: 0.0, z: 0.0}
  - {x: 10.0, y: 0.0, z: 0.0}
  - {x: 10.0, y: 10.0, z: 0.0}
  - {x: 0.0, y: 10.0, z: 0.0}
  walls:
  - start: {x: 0.0, y: 0.0, z: 0.0}
    end:   {x: 10.0, y: 0.0, z: 0.0}
    material: {name: Plaster_Wall, domain: Common, _modifiers: null}
  - start: {x: 10.0, y: 0.0, z: 0.0}
    end:   {x: 10.0, y: 10.0, z: 0.0}
    material: {name: Plaster_Wall, domain: Common, _modifiers: null}
  - start: {x: 10.0, y: 10.0, z: 0.0}
    end:   {x: 0.0, y: 10.0, z: 0.0}
    material: {name: Plaster_Wall, domain: Common, _modifiers: null}
  - start: {x: 0.0, y: 10.0, z: 0.0}
    end:   {x: 0.0, y: 0.0, z: 0.0}
    material: {name: Plaster_Wall, domain: Common, _modifiers: null}
  doors: []
  elevators: []
  entities:
    static: []
    dynamic: []
```

Key points:

- `corners` is an ordered polygon; its interior is the navigable floor.
- Each `walls:` entry needs at least `start`, `end`, and either `material` or
  `kind`. Use `kind: <name>` to reference a `WallIdentifier` asset; omit
  `kind` for an inline material.
- `material` on a zone accepts the same forms as `MaterialIdentifier`: a plain
  string `Concrete_Smooth`, or a two-element list `[name, modifiers_dict]`.
- Multiple zones share a single flat `zones` list. Zone boundaries are defined
  only by their `corners` polygon and their `walls` list, with no explicit
  parent-child relationship.
- An empty material opts a surface out entirely: `material: ''` on a zone
  spawns no floor, `ceiling_material: ''` no ceiling, `material: ''` on a
  wall entry drops that wall, and `wall_material: ''` on every zone of a
  level suppresses the occupancy-detected collision walls. The short keys
  `mat`, `ceiling_mat`, `wall_mat` are accepted as aliases.

### Ceilings

Add a ceiling to any zone with these optional keys:

```yaml
  ceiling: true                # true by default, set false to leave the zone open
  ceiling_height: 3.0          # top height in metres, omit to derive from wall stack
  ceiling_cast_shadows: false  # false by default, true to let the ceiling occlude light
  ceiling_material:            # MaterialIdentifier, defaults to Concrete_Smooth
  - Concrete_Smooth
  - {}
```

With `ceiling_height` absent, the height is the tallest wall top in the zone
(`max(segment.start.z + segment.height)` over the zone walls), falling back to
`2.0` m when the zone has no walls.

Ceilings are opaque from below and transparent from above. They are visual-only
(no collision). With `ceiling_cast_shadows` false the ceiling does not occlude
the sun, so interiors stay lit without global illumination.

Full field reference: [worlds/README.md](worlds/README.md).

## 3. Generate the map

```bash
touch_world my_world
```

This reads `worlds/my_world/world.yaml`, renders a PNG via
`WorldDescription.render()`, and writes:
- `map/map.png`
- `map/map.yaml`

To regenerate after editing `world.yaml`:

```bash
touch_world my_world
```

To regenerate with visible static obstacle footprints:

```bash
touch_world my_world \
    --assets red \
    --assets-label blue \
    --assets-bbox "((-.5, .5), (-.5, .5))"
```

To regenerate all files (including `world.yaml` itself, e.g. after schema
changes):

```bash
touch_world my_world --all
```

`touch_world` accepts `--resolution <float>` (metres per pixel, default `0.05`).

## 4. Add a scenario

Create `worlds/$WORLD/scenarios/default/scenario.yaml`:

```yaml
static: []
dynamic: []
robots:
  - start: [1.0, 1.0, 0.0]
    goal:  [8.0, 8.0, 0.0]
```

An empty scenario (robot start/goal only) is valid. Add static obstacles by
listing `Obstacle` entries under `static:`, dynamic pedestrians under `dynamic:`.

For HuNav pedestrians include `behavior_tree:` pointing at either a shared
library file (`BTRegularNav.xml`) or a path-relative file
(`./hunav_1_behavior_tree.xml`):

```yaml
dynamic:
- name: agent_1
  model: female_adult_medical_01
  behavior_tree: BTRegularNav.xml
  pose: [2.0, 5.0, 0.0]
  velocity: 1.2
  desired_velocity: 1.2
  waypoint:
  - [2.0, 5.0, 0.0]
  - [8.0, 5.0, 0.0]
```

See [worlds/README.md](worlds/README.md) for the full
scenario schema.

## 5. Add world-local assets (optional)

Assets placed in `worlds/$WORLD/assets/` are resolved before network-fetched
assets. To ship a custom wall style:

```bash
mkdir -p worlds/$WORLD/assets/Common/Wall/my_style
# create worlds/$WORLD/assets/Common/Wall/my_style/my_style.yaml
```

Then reference it in `world.yaml` walls as `kind: my_style`.

Wall preset YAML schema: [configs/walls/README.md](configs/walls/README.md).

## 6. Generate with the AI pipeline (alternative)

[scripts/generate_world](scripts/generate_world) posts a
natural-language prompt to a generation server and extracts the resulting zip
into a world directory:

```bash
generate_world "A hospital floor with 4 patient rooms and a central hallway" \
    --endpoint http://localhost:5501 \
    --outdir my_world
```

`--outdir` is resolved relative to `<arena_simulation_setup share>/worlds/` if
it is not an absolute path. The server must be running separately; this script
is a thin client.

After generation, run `touch_world my_world` to ensure `map/` is current.

## 7. Validate

```python
from arena_simulation_setup.tree.World.World import WorldIdentifier

world = WorldIdentifier('my_world').resolve_sync()
desc = world.load()
print(f"{len(desc.zones)} zone(s), "
      f"{sum(1 for _ in desc.all_walls)} wall segments")
```

This will raise `FileNotFoundError` if `world.yaml` is missing, and `ValueError`
or `cattrs` errors if the YAML is malformed.

List all scenarios:

```python
for scenario_id in world.scenario.listall():
    print(scenario_id.name)
```

Load a scenario:

```python
scenario = world.scenario('default').resolve_sync()
print(scenario.load())
```
