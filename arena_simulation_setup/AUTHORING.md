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

`WorldIdentifier` resolves a name against, in order: `ARENA_WORLD_PATH` roots,
then this package's `worlds/`, then a network bucket named by `WORLD_BUCKETS`,
then a local write-only fallback. Once authored, publish a world with
`arena asset push world <name>`, which refuses to publish if any asset it
references would not resolve for someone else. `arena asset find world <name>`
shows which source a name currently resolves against. See
[tree/README.md](src/arena_simulation_setup/tree/README.md).

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
  level suppresses the occupancy-detected pedestrian collision walls (robot
  collision tracking reads the map regardless). The short keys
  `mat`, `ceiling_mat`, `wall_mat` are accepted as aliases.
- `entities: {static: [...], dynamic: [...]}` on a zone places obstacles and
  pedestrians that exist in every scenario run against this world. Same
  `Obstacle`/`DynamicObstacle` schema as a scenario's `static:`/`dynamic:`
  lists, see [4. Add a scenario](#4-add-a-scenario).

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

Ceilings are opaque from below and transparent from above. Isaac hides them
unless the world declares lights, `sim.isaac.viewport.ceilings:=on|off` forces
them. The walls of a zone with a ceiling reach it, and a door lower than the
ceiling gets a visual-only lintel above it. They are visual-only
(no collision). With `ceiling_cast_shadows` false the ceiling does not occlude
the sun, so interiors stay lit without global illumination.

### Doors and elevators

A `doors:` entry configures its geometry, kind, timing, and material. Full
field reference: [Door fields](worlds/README.md#door-fields).

An `elevators:` entry configures its cabin, destinations, and call
behavior. Full field reference:
[Elevator fields](worlds/README.md#elevator-fields).

A world with more than one storey follows the layout in
[worlds/README.md, Directory layout](worlds/README.md#directory-layout).

An elevator's `destination` names its counterpart as `<level_id>.<name>`,
e.g. `destination: "2.2_elevator"` targets the elevator named `2_elevator`
on level `2`. Several destinations are comma-separated. Elevator names must
be unique across every level, loading raises if two levels declare the same
name or a destination points at a level/elevator that does not exist.

### Microphones

A level's optional `microphones:` list places audio listeners for the
acoustics pipeline. Full field reference:
[microphones](worlds/README.md#microphones).

### Semantic annotations

Door and elevator state is intrinsic: every spawned door/elevator publishes
its full vocabulary (`state`/`progress`/`open`/... for a door; `arriving_eta`,
`occupants`, `cabin_door`, `cabin_door_progress`, ... for an elevator) with no
annotation needed. `semantics:` on a `doors:`/`elevators:` entry is only for
attaching a *scripted* kind (`gate`, `pressure_plate`) to that entry. Zones
accept an optional `semantics:` list of literal state/predicate primitives:

```yaml
zones:
- name: lobby
  corners: [...]
  semantics:
  - {state: max_speed, value: 1.5}
  - {predicate: quiet, value: true}
```

An `Elevator` entry configures its fire-recall regime as a first-class field,
not a semantics annotation:

```yaml
elevators:
- name: 1_elevator
  position: {x: 5.0, y: 0.0, z: 0.0}
  destination: "2.2_elevator"
  recall_on: alarm
```

Full field reference: [worlds/README.md](worlds/README.md).

### M2 kinds, params, and timelines

M2 adds scriptable/derived kinds on top of the intrinsic `door`/`elevator`
vocabularies and `zone`. Each kind takes its config under a `params:` dict on
the primitive or preset item, which round-trips opaquely (omitted on
serialize when empty):

| kind | attaches to | preset expansion | `params` keys |
| --- | --- | --- | --- |
| `signal` | a standalone `signals:` entry (zone-level) | `state`, `phase_remaining`, `stop` | `phases` (list of `{name, duration}`), `stop_phases`, `regime` |
| `schedule` | a standalone `schedules:` entry (zone-level) | `state`, `active`, `window_remaining` | `windows` (list of `{start, end, value}`), `default`, `regime` |
| `gate` | a `doors:` entry | `locked`, `blocked` | `authorized` (sim_paths/robot names), `unlock_on`, `locked` (initial value) |
| `pressure_plate` | a `doors:`/`elevators:` entry (own `position`) | `pressed` | `position` (`[x, y]`), `radius`, `drives`, `latch`, `press_on`, `regime` |
| `occupancy_cap` | a `zones:` entry | `occupancy`, `cap`, `over_cap` | `cap` |
| `sound` | a standalone `sounds:` entry (zone- or scenario-level) | `sounding`, `volume_db` | `sound_on`, `regime`, `sounding`/`volume_db` (initial values) |
| `light` | every `lights:` entry and every zone `ceiling_lights:` rig, attached from the light's own fields | `lit`, `level`, `dead_fraction` (rigs only) | `light_on` |

`signal`, `schedule`, and `sound` have no wall/door geometry of their own, so
a zone carries them as sibling lists to `doors:`/`elevators:`. A `sound`
still needs a placement: exactly one of `position` (`[x, y]`), `entity_ref`
(name of a static entity in the same world, whose yaw frame `offset` is
applied in) or `frame` (a TF frame the sound rides, `offset` local to it, so
`frame: jackal/base_link` is a speaker bolted to that robot). Other `Sound`
fields: `asset_id` (required, the audio catalog entry), `loop` (default
`true`), `reference_distance_m` (default `1.0`, must be positive, sets the
falloff reference distance), and `level` (the level id a bare `position`
belongs to, required in a multi-level world).

```yaml
zones:
- name: lobby
  corners: [...]
  schedules:
  - name: fire_alarm
    semantics:
    - {preset: schedule, params: {windows: [], regime: alarm}}
  signals:
  - name: crosswalk_light
    semantics:
    - {preset: signal, params: {phases: [{name: go, duration: 20.0}, {name: stop, duration: 10.0}]}}
  sounds:
  - name: hall_siren
    asset_id: alarm_loop
    position: [4.0, 2.3]
    semantics:
    - {preset: sound, params: {sound_on: alarm, volume_db: 88.0}}
```

A preset `params:` key that names one of the preset's own primitives
(`volume_db`, `sounding`, `locked`, ...) becomes that primitive's initial
`value` instead of a shared param, so `{preset: sound, params: {sounding:
true, volume_db: 62.0}}` is a radio that plays from the start at 62 dB. A
top-level `value:` on a preset item that expands to more than one primitive
is rejected at load: there is no single primitive it could mean.

A scenario may carry its own `sounds:` list with the same schema. Those are
episode-scoped: attached at reset, gone at the next one, and their
`entity_ref` may also name one of the scenario's own `static:` obstacles.

#### Lights

A light is one fixture, an ambient light, or a zone's ceiling rig. Each is a
`light` semantic entity with `lit` (on/off), `level` (dimming in 0..1) and,
for rigs, `dead_fraction` (share of dead fixtures), all writable through
`semantics/set` and the scenario timeline and recorded with every other kind.
Worlds that declare no light render as before. A world that declares any
light gets no default lighting at all, so what it declares is all there is
and a blackout is dark. Lights that objects carry (below) do not count as
declared, so a lamp adds its light on top of the default lighting.

| Where | Fixtures | Placement |
| --- | --- | --- |
| level `lights:` | `dome`, `sun` | none, `sun` takes `direction` |
| zone `lights:` | `panel`, `tube`, `downlight`, `spot`, `bulb` | exactly one of `position`, `entity_ref` or `frame`, plus `offset`, as for a `sound` |
| zone `ceiling_lights:` | `panel`, `tube`, `downlight`, `spot`, `bulb` | a grid at `spacing` metres over the zone polygon, just below the ceiling |
| object `annotation.yaml` `lights:` | `panel`, `tube`, `downlight`, `spot`, `bulb` | `offset` in the object frame, moved, turned and scaled with every placement |

`Light` fields: `name`, `fixture`, `lumens` (local fixtures, per fixture for a
rig) or `lux` (`dome`, `sun`), `cct_K` (default 6500), `cast_shadows` (default
false), `direction` (all but `bulb` and `dome`, default straight down),
`cone_deg` (`spot` only, default 40), and one
of `light_on` (a regime, `!` negates) or `lit` (initial value, default true),
plus `level` (default 1.0). `ceiling_lights:` takes `fixture` (default
`panel`), `spacing` (default 2.4), `lumens`, `cct_K`, `cast_shadows`,
`light_on`/`lit`, `level` and `dead_fraction`, needs a ceiling, and its rig is
named after the zone. The same fixtures of a rig die at the same
`dead_fraction` in every run, and raising it only adds dead ones. A frame
light follows its TF frame, so `frame: jackal/base_link` is a headlight.

A world without lights can still run lit: `world.lighting:=auto` gives every
zone that has a ceiling and no `lights:` or `ceiling_lights:` of its own a rig
of 7200 lm panels at 2.4 m spacing, without touching the world file. The
default `world.lighting:=authored` renders exactly what the world declares.

```yaml
lights:
- {name: ambient, fixture: dome, lux: 5}
zones:
- name: central_hallway
  ceiling_lights: {fixture: panel, spacing: 2.4, lumens: 3600, light_on: "!blackout"}
  lights:
  - {name: exit_strip, position: {x: 10, y: 34, z: 2.2}, fixture: tube, lumens: 300, cct_K: 6500, light_on: blackout}
  schedules:
  - name: power
    semantics:
    - {preset: schedule, params: {windows: [{start: 60, end: 90, value: out}], regime: blackout}}
```

An object's `annotation.yaml` can carry `lights:` entries with `name` (default
`light`), `fixture` (default `bulb`), `offset`, `direction`, `lumens`, `cct_K`,
`cone_deg` and `glow`. `arena_assets` writes them when it builds an object whose
source model contains lights (hand-written entries win, and the built models
carry no light themselves). `glow` names the material that glows with the
light, a lamp shade for instance: both simulators make it emit the light's
color scaled by its level, and it is dark while the light is off. Every placement of the object, in a zone's
`static:` list, a scenario's `static:` list or by an obstacle mode, then gets
the light `<entity>_<name>` for as long as the placement exists, and a
`light:` key on the placement sets its `lit`, `light_on` or `level`:

```yaml
static:
- {name: desk_lamp, model: Residential/Desk_Lamp, pose: {position: [2, 1, 0.75]}, light: {light_on: "!daylight"}}
```

A scenario dims the hall with `{entity: central_hallway, field: level, value:
"0.3..1.0"}` in its timeline. Isaac renders every fixture as an area light, and
shades asset materials that only emit a texture by that texture, so they
darken with the lights.
Gazebo renders a rig as one shadowless point light per 2 by 2 block of
fixtures, plus one unattenuated fill at the room center for the light the
room surfaces reflect (40 % reflectance), since its renderer traces no
indirect light.
Dome and sun light the whole stage, so the first env's world sets them for
every env of a runtime, and an env whose world declares other ones logs a
warning naming both.

#### Pedestrian stimuli

A sound that propagation marks audible for a pedestrian reaches humansim as a
stimulus named after the catalog `category` of its `asset_id` (`alarm_loop`
has `category: alarm`, so its stimulus is `alarm`). An agent type opts in by
declaring a need of that name and a `transitions:` entry on it, and humansim
sets that need to 100 once the agent's sampled `reaction_time` has elapsed,
so the transition fires. Agent types without such a need ignore the sound.
See
[worlds/hospital_1/scenarios/fire_alarm_evacuation](worlds/hospital_1/scenarios/fire_alarm_evacuation/scenario.yaml)
for the reference: the `evacuee` agent type idles until `alarm` rises, then
walks to the static `exit` object.

`gate` and `pressure_plate` reuse an existing `doors:`/`elevators:` entry as
their attachment point, `occupancy_cap` reuses a `zones:` entry, all via the
same `semantics:` list. A `gate` should spawn unlocked (`locked: false` in its
params): the world is shared by every scenario, so a gate that defaulted to
locked would block scenarios that never touch its regime. A scenario that
wants the door locked at episode start says so explicitly in its `timeline:`
(see below), leaving the world itself inert:

```yaml
doors:
- name: door_edge_1_1
  start: {x: 4.0, y: 1.55, z: 0.0}
  end:   {x: 4.0, y: 2.45, z: 0.0}
  semantics:
  - {preset: gate, params: {authorized: [], unlock_on: alarm, locked: false}}
  - {preset: pressure_plate, params: {position: [4.0, 2.0], press_on: alarm, drives: door_edge_1_1}}
```

A `regime` (or its per-kind alias `unlock_on`/`press_on`) names a boolean
asserted by a scripted kind's driving predicate. Other kinds (gate,
pressure_plate) consult that name without a direct wire between the two
entities. A `sound` consumes a regime the same way, via its `sound_on` alias,
and a light via `light_on`. Every follower field accepts a leading `!` for the
negation, so `light_on: "!blackout"` is lit unless `blackout` is asserted.
An elevator's `recall_on` field is the same regime-consult
mechanism, just wired as a first-class `Elevator` field instead of a
`semantics:` alias, since recall is mechanism configuration rather than
published state. See the fire-alarm worked example below.

### Scenario timelines

`scenario.yaml` accepts an optional `timeline:` list. Each entry fires a `set`
action list of `{entity, field, value}` writes against exactly one trigger:

| trigger | meaning |
| --- | --- |
| `at: <seconds>` | fire once at episode second `t` |
| `every: <seconds>` | fire at `offset` (default: one period in), then each period, optional `until` |
| `when: {entity, field, is}` | fire on the false-to-true edge of a semantic value |

```yaml
timeline:
- at: 0.0
  set:
  - {entity: door_edge_1_1, field: locked, value: "true"}
- at: 12.0
  set:
  - {entity: fire_alarm, field: active, value: "true"}
```

The world's `door_edge_1_1` spawns unlocked, and this scenario locks it at
`t=0` so the corridor starts sealed. At `t=12s` the `fire_alarm` schedule's `active` predicate goes true and
asserts regime `alarm`: a gate with `unlock_on: alarm` reports `locked=false`,
a pressure plate with `press_on: alarm` reports `pressed=true` and holds its
`drives` door open, and an elevator with `recall_on: alarm` refuses calls and
holds its cabin door open, all as pure regime consults with no per-effect
timeline entry. See
[worlds/three_storied_residential/scenarios/fire_alarm/scenario.yaml](worlds/three_storied_residential/scenarios/fire_alarm/scenario.yaml)
for the reference timeline.

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
    phases:
      - {goto: [8.0, 8.0, 0.0]}
```

An empty scenario (robot start and phases only) is valid. Phases run in order
and are `{goto: <pose or zone/door/elevator/pedestrian name>}`, `{gesture: <name>}`,
`{reach: <named target | random | pose>}`, or a hold phase with neither (the robot
stays where it is). Every phase takes `until: "<atom>"` (completes once the atom
holds, after arrival for a goto), `hold_time` (park seconds on a goto), `signal`
(the robot ends the goto itself by sending that signal, such as `arrived`, and
is judged against the tolerance at that moment), `conditions:` (clauses judged
over this phase only), `text:` (the instruction handed to language-conditioned
planners, `instruction:` is accepted as the same key) and `on_failure`. A robot entry may also carry
`conditions:` judged over its whole phase list. `goal:` is deprecated.

```yaml
robots:
  - start: [1.0, 1.0, 0.0]
    phases:
      - {goto: kitchen}
      - {until: "not visitor_1 in kitchen"}
      - {goto: sofa, hold_time: 2, conditions: [{op: never, p: "robot in hallway"}]}
```

Add static obstacles by
listing `Obstacle` entries under `static:`, dynamic pedestrians under `dynamic:`.
A static entry's flat top-level keys `type`, `capacity`, `satisfies`,
`interaction_radius` and `formation` are forwarded to humansim as its
WorldObject, so a `go_to` step or an interaction `target:` can name the
entry by `type`.

For HumanSim (arena_humansim) pedestrians include an `agent:` block. The
`agent_type` is either a built-in type (`adult`, `elder`) or a path-relative
agent-type YAML file (`./doctor.yaml`) defining scripted behavior
(sequences, interactions):

```yaml
dynamic:
- name: agent_1
  model: female_adult_medical_01
  agent: {agent_type: adult, desired_velocity: 1.2}
  pose: [2.0, 5.0, 0.0]   # yaw in radians
  waypoints:
  - [2.0, 5.0, 0.0]
  - [8.0, 5.0, 0.0]
```

A scenario can also carry `regions:` (named pedestrian source/sink spawners
for arena_humansim) and `conditions:` (episode success/failure clauses). Its
`sounds:` and `timeline:` keys are the ones covered above. See
[worlds/README.md](worlds/README.md) for the full key list of every
top-level key.

## 5. Add world-local assets (optional)

Assets placed under `worlds/$WORLD/assets/<domain>/<Type>/<name>/` are
resolved before shared-local (`$ARENA_ASSETS_DIR_LOCAL`) and network-fetched
assets. `<domain>` defaults to `Common` if you have no reason to scope the
asset further. Five asset kinds can be shipped this way:

| Type | Directory contents | Referenced as |
|---|---|---|
| `Object` | 3D model files (SDF and/or USD), optional `annotation.yaml` with a `bounding_box` | `model:` on a static obstacle entry |
| `Human` | 3D model files (SDF) | `model:` on a HumanSim pedestrian entry |
| `Material` | `<name>.mdl` plus its texture files | `material:` on a zone or wall entry |
| `Wall` | `<name>.yaml`, a `WallDescription` (see [configs/walls/README.md](configs/walls/README.md)) | `kind:` on a `walls:` entry |
| `Sound` | `<name>.yaml` (a sound manifest, see [configs/sounds/README.md](configs/sounds/README.md)) plus its wav files | `asset_id:` on a `sounds:` entry |

To ship a custom wall style:

```bash
mkdir -p worlds/$WORLD/assets/Common/Wall/my_style
# create worlds/$WORLD/assets/Common/Wall/my_style/my_style.yaml
```

Then reference it in `world.yaml` walls as `kind: my_style`. The other four
kinds follow the same `assets/<domain>/<Type>/<name>/` layout. Objects, humans
and materials carry model or material files instead of a preset YAML, a sound
carries its `<name>.yaml` manifest beside the wav files.

A wall style can also carry a captured look: a 3D Gaussian splat that Isaac
renders in front of the wall. The
[splat wall example](configs/walls/README.md#example-splat-wall) shows the wall
kind, the [arena_assets README](../arena_assets/README.md) how to build the asset.
A captured object (`arena-assets splat --object`) is placed like any other
static object, and it is visual only: Isaac renders it and its sensors see it,
but robots drive through it.

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

### Reachability (NAV@r)

`nav_at_r` reports how much of the floor a robot of footprint radius `r` can
reach, the NAV@r metric of the IndoorBench paper:

```bash
ros2 run arena_simulation_setup nav_at_r my_world
ros2 run arena_simulation_setup nav_at_r my_world --robot jackal --walls-only
ros2 run arena_simulation_setup nav_at_r /path/to/world --radius 0.3 --start 2.0,3.5 --json
```

`--image map.png` paints the result onto the map: the dominant component in
green, every stranded component in another color, free cells with clearance
below `r` in light grey, entities in brown and walls in black.

Each level is rasterized at 0.05 m per cell with the map rasterizer, walls
plus the footprint of every static entity (its `bbox`, else the
`bounding_box` of its model's `annotation.yaml`, entities whose box starts
above 2 m are skipped). A cell is traversable when its clearance is at least
`r` (default 0.267 m, the Jackal radius), and traversable cells group into
8-connected components. With `m_i` the share of traversable cells in component
`i` and `w_i` its share of spawn cells (clearance of at least 0.45 m), NAV@r is
`sum_i w_i * m_i`: 1.0 when every traversable cell is reachable from every
spawn, lower the more floor is stranded in other components.

| Column | Meaning |
|---|---|
| `NAV@r` | the expectation above, `n/a` when no cell admits a spawn |
| `components` | number of traversable components |
| `largest` | share of traversable cells in the largest (dominant) component |
| `start` | with `--start X,Y` (level frame, meters): share of the component holding the start, snapped to the nearest traversable cell |
| `no bbox` | static entities without a resolvable bounding box, out of all static entities. They are not rasterized |
| `cut-off zones` | zones of at least 2 m2 with no cell in the dominant component although they are traversable on walls only |

`--walls-only` drops the static entities. A zone that stays cut off there has a
layout fault (a missing, misplaced or too narrow door), one that is cut off
only in the furnished run is blocked by furniture. `--robot` takes a robot name
from the installed `arena_robots` package or a robot directory and reads
`radius` from its `caps/mobile.yaml`. A world name resolves like at runtime,
and `name[0,1]` restricts the levels.

The same numbers are available in Python:

```python
from arena_simulation_setup.metrics.nav_at_r import nav_at_r, resolve_world

view, levels = resolve_world('my_world')
for level_id, result in nav_at_r(view, radius=0.267, levels=levels).items():
    print(level_id, result.value, result.components, result.cut_off_zones)
```
