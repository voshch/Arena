# Sound kinds and manifests

`configs/sounds/kinds.yaml` is the kinds table of the sound catalog
([tree/assets/sound_catalog.py](../../src/arena_simulation_setup/tree/assets/sound_catalog.py)).
A kind groups sound assets by what emits them and how a listener treats them.
The catalog is shared by the sounds task module, the acoustics backends and
the robot hearing stack.

## Kinds table

| Kind | Agent | Stem | Default asset | Notes |
|---|---|---|---|---|
| `footstep` | pedestrian | `pedestrian` | `footstep` | detected, marker |
| `speech` | pedestrian | `pedestrian` | `greeting` | detected, marker |
| `music` | environment | `ambient` | `radio_loop` | |
| `alarm` | environment | `ambient` | `alarm_loop` | |
| `motor` | robot | `motor` | `motor` | marker, `jackal_drivetrain` variant for Jackals |
| `onset` | | `ambient` | | class-agnostic detection of the srp front-end |

Row keys:

| Key | Default | Meaning |
|---|---|---|
| `stem` | required | `pedestrian`, `ambient` or `motor`, the mix group the kind's sources land in |
| `agent` | none | `pedestrian`, `robot`, `environment` or `external`, who may emit the kind |
| `height_m` | `0.0` | emitter height above the floor at the agent |
| `detect` | `false` | hearing front-ends report the kind |
| `marker` | `false` | the kind gets a visualization marker |
| `color` | `[0.8, 0.8, 0.8]` | marker color, RGB in 0..1 |
| `default_asset` | empty | asset used when a source names only the kind |

Unknown keys are rejected.

## Sound manifests

Sounds are assets of the `Sound` kind, resolved like every other asset:
world-local `worlds/<w>/assets/Common/Sound/<name>/`, then
`$ARENA_ASSETS_DIR_LOCAL/Common/Sound/<name>/`, then the network providers.
Each asset directory holds `<name>.yaml` beside its wav files:

```yaml
version: 2
kind: music
desc: Looping radio music.
tags: [music, radio, loop, environment]
level_db: 62.0              # SPL at reference_distance_m
reference_distance_m: 1.0
normalize_dbfs: -12.5
loop: true
model: wav_loop
variants:
  - {id: radio_loop_01, file: radio_loop.wav}
```

`desc` and `tags` describe the asset for the asset database, which indexes
sounds from these keys. Variant ids are unique within their asset. Variants
may carry `match` (for example `{floor: [oak]}` on footsteps),
`default: true`, `tags`, a per-variant `model` and `params`. The producer picks
the variant deterministically from the source seed
(`SoundAsset.select`, `selection_seed`).

A manifest may add kinds through a `kinds:` mapping with the row keys above.
`SoundLibrary` reads these fragments from every manifest in the world and the
local asset tree when it starts and on every world switch, so the kinds table
is complete before any asset loads. An asset only a network provider holds
adds its kinds when it first loads.

The default assets `footstep`, `greeting`, `motor`, `radio_loop` and
`alarm_loop` live in the asset bucket. Inspect and move sounds with the asset
CLI:

```bash
arena asset ls sound
arena asset find sound footstep
arena asset pull sound <name>
arena asset push sound <name>
```

Looping files should match in waveform and level at both ends so the join
does not click.
