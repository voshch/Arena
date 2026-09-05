# Wall presets

Files under `configs/walls/` are `WallDescription` YAML presets, a curated
library of named wall styles. They share the same schema as wall assets under
`assets/<domain>/Wall/<name>/<name>.yaml`, and can serve as starting points
when authoring world-local wall assets.

## `WallDescription` schema

A wall description has a single key `main`, whose value is a list of
sub-wall directives processed in order. Each directive places wall segments
and/or obstacles along the line from `start` to `end`.

Sub-wall types (see
[tree/Wall.py](../../src/arena_simulation_setup/tree/Wall.py)):

| Type | Discriminator | Fields | Effect |
|---|---|---|---|
| `PlaceWallSegmentAsset` | has `material` key | `material`, `height`, `width`, `name`, `x/y/z` offset | Places a single wall segment |
| `PlaceObstacleAsset` | has `model` key | `model`, `at` (position along wall, default `50%`), `orientation`, `name`, `x/y/z` offset | Places a single obstacle |
| `TilingAsset` | has `tile` key | `tile` (list of sub-walls), `every` (spacing [m]), `width` (half-width [m]) | Repeats `tile` contents every N meters |
| `FillAsset` | has `fill` key | `fill` (list of sub-walls), `start`, `end` (positions along wall) | Applies `fill` over a slice of the wall |

`at`, `start`, `end` accept:
- absolute metres: `1.5`
- negative absolute (from end): `-0.5`
- percentage: `"50%"`

### Example: layered wall with tiled paintings

```yaml
main:
  - material: Metal_Cast      # PlaceWallSegmentAsset; height 2 m
    height: 2
  - model: residential_painting_1   # PlaceObstacleAsset at 50%
    at: 50%
    orientation: 1.5708       # radians; rotate to face inward
    y: 0.1                    # shift 0.1 m away from the wall plane
    z: 1                      # height offset [m]
```

### Example: stacked material layers with tiled assets

```yaml
main:
  - material: Bamboo_Planks   # 1 m wainscoting
    height: 1
    z: 0
  - material: Mortar_White    # 0.75 m mid-band
    height: 0.75
    z: 1
  - material: Metal_Cast      # 0.25 m cornice
    height: 0.25
    z: 1.75
  - tile:                     # TilingAsset: repeat every 5 m
      - model: Hospital/Towel_Dispenser
        orientation: 1.5708
        y: 0.1
        z: 1
    every: 5
    width: 3
```

## How world.yaml references a wall preset

In `world.yaml`, a `walls:` list entry can carry a `kind:` field naming a
`WallIdentifier`. At simulation time,
[shared/walls.py:30](../../src/arena_simulation_setup/shared/walls.py#L30)
calls `WallIdentifier(kind).resolve()`, which searches
`DynamicPaths.WORLD / 'assets'` then `DynamicPaths.ARENA` then network
providers. The `configs/walls/` directory in this package is **not** on that
path directly, to expose a preset, copy or symlink it into
`worlds/<name>/assets/Common/Wall/<preset_name>/<preset_name>.yaml` (world-local)
or into `$ARENA_ASSETS_DIR_LOCAL/Common/Wall/<preset_name>/<preset_name>.yaml`
(shared local).

## Shipped presets

| File | Description |
|---|---|
| `hospital.yaml` | Three-layer wall: bamboo wainscoting, mortar mid-band, metal cornice |
| `hospital_2.yaml` – `hospital_5.yaml` | Hospital wall variants |
| `hospital_with_painting.yaml` | Hospital base with tiled painting assets |
| `painting.yaml` | Tiled paintings on a layered material wall |
| `invisible.yaml` | Material `invisible`; collision without visual geometry |
