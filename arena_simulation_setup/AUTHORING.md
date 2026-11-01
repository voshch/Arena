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
    $ARENA_DIR/arena_simulation_setup/worlds/$WORLD/scenarios/default \
    $ARENA_DIR/arena_simulation_setup/worlds/$WORLD/assets
```

`$ARENA_DIR` is the workspace source root. If unset, `touch_world` uses the
current working directory.

`WorldIdentifier` resolves a name against, in order: `ARENA_WORLD_PATH` roots,
then the writable local root `$ARENA_ASSETS_DIR_LOCAL/worlds`, then this
package's `worlds/`, then a network bucket named by `WORLD_BUCKETS`. Worlds
saved through `resolve_write_sync()` (the world generator, the `worlds` MCP
tools) land in the writable local root, outside the repo and the install
space. Once authored, publish a world with
`arena asset push world <name>`, which refuses to publish if any asset it
references would not resolve for someone else. `arena asset find world <name>`
shows which source a name currently resolves against. See
[tree/README.md](src/arena_simulation_setup/tree/README.md).

## 2. Author `world.yaml`

A world is one file, `worlds/$WORLD/world.yaml`, holding every level. The
minimum valid file is one level with one walled zone:

```yaml
format: arena-world/2
levels:
- id: '0'
  zones:
  - name: main_room
    polygon: [[0, 0], [10, 0], [10, 10], [0, 10]]
    walls: outline
```

The full shape, every key optional except `format`, `levels`, a level's `id`
and a zone's `name`:

```yaml
format: arena-world/2
layout: spread                           # or stacked: levels share one floor plan at their elevations
defaults:                                # inherited by every level and item that does not set its own
  floor: Concrete_Smooth                 # material, or false for no floor
  ceiling: {height: 2.7, material: Concrete_Smooth}   # or false for open zones
  wall: {material: Plaster_Wall}         # or {kind: patient_room_walls}
  door: {kind: sliding, material: Aluminum_Anodized}
schedules: [...]                         # world scope, see M2 kinds below
signals: [...]
levels:
- id: '0'
  elevation: 0.0                         # floor height, a storey of 3 m per level before it by default
  pedestrian_walls: true                 # false drops the occupancy-detected pedestrian walls
  zones:
  - name: central_hallway
    polygon: [[8, 0], [12, 0], [12, 34.2], [8, 34.2]]
    floor: Marble
    ceiling: {height: 3.0, cast_shadows: false, lights: {fixture: panel, spacing: 2.4, lumens: 3600}}
    walls: outline                       # or {outline: true, open_to: [lobby], material: ..., kind: ...}
    semantics: [{state: max_speed, value: 1.5}]
  walls:
  - {path: [[0, 10], [0, 20], [8, 20]], kind: patient_room_walls}
  - {path: [[20, 0], [24, 0], [24, 4], [20, 4]], closed: true}
  doors:
  - {name: main_entrance, span: [[9.25, 0], [10.75, 0]]}
  elevators:
  - {name: lift_0, position: [5, 0], destinations: [lift_1]}
  stairs:
  - {name: stairs_0, kind: stairs, mouth: [16, 2], size: [5.2, 1.4], to: '1', exit: [16, 2]}
  escalators:
  - {name: up_0, kind: belt_escalator, mouth: [16, 6], size: [14.0, 1.0], to: '1', exit: [16, 6]}
  objects:
  - {name: desk_1, model: Common/Desk, pose: [2.0, 1.0, 1.5708]}
  humans: []
  lights:
  - {name: exit_strip, fixture: tube, lumens: 300, pose: {x: 10, y: 34, z: 2.2}}
  sounds:
  - {name: hall_siren, asset_id: alarm_loop, position: [4.0, 2.3]}
  microphones:
  - {zone: central_hallway, placement: ceiling, position: [10, 5, 3.0]}
```

Key points:

- A zone is a named region: its `polygon` (an open ring, the interior is
  navigable floor), its `floor` and `ceiling`, its runtime `semantics`, and
  optionally walls along its outline. It holds nothing else. Walls, doors,
  elevators, stairs, escalators, objects, humans, lights, sounds and
  microphones are lists of the level, and which zone a thing belongs to is a semantic fact (see
  [KG](#kg)).
- Every name (zone, door, elevator, stair or escalator half, object, human,
  light, sound, schedule, signal) is unique across the whole world, so a reference never needs a
  level. Loading fails on a repeated name, on a key the format does not know
  (the error names the YAML path, e.g. `levels[0].doors[3]: unknown keys
  ['heigth']`) and on a reference to an unknown name.
- Positions are `[x, y]`, `[x, y, z]` or `{x, y[, z]}`. Poses take the forms
  of `Pose.parse` in [utils/geometry.py](src/arena_simulation_setup/utils/geometry.py):
  `[x, y]`, `[x, y, yaw]` (radians), `{x, y[, z][, roll][, pitch][, yaw]}`,
  `[x, y, z, roll, pitch, yaw]`, `[x, y, z, w, x, y, z]` or a zone name.
- A material is `Name` (domain `Common`), `Domain/Name` or `[Name, {modifiers}]`.
- `defaults` apply to every zone, wall, outline and door that does not set
  the key itself. Without `defaults`, a zone has a `Concrete_Smooth` floor and
  ceiling, and walls and doors take the package defaults.
- `walls:` entries are polylines, `closed: true` joins the last point back to
  the first, `kind: <name>` references a `WallIdentifier` asset and
  `material:` an inline material, never both. `walls: outline` on a zone draws
  a wall along every polygon edge, `open_to:` leaves open the edges it shares
  with the named zones. Repeated walls are merged, and a door cuts every wall
  it lies on, so draw walls straight through doorways.
- `objects:` and `humans:` take the same schema as a scenario's lists (see
  [4. Add a scenario](#4-add-a-scenario)) and exist in every scenario run
  against the world.

### Linked levels

A level can come from another world instead of being written out:
`- {from: <world>[:<level id>], id: <level id>}`. `<world>` is a world name,
resolved like any world, or a path relative to this world's directory
(`./floors/2`), so a tall building keeps each floor in its own small format 2
world under a subfolder:

```yaml
format: arena-world/2
levels:
- from: ./floors/ground            # floors/ground/world.yaml, a one-level world
- from: ./floors/first
- {id: '2', from: 'office_1:0'}     # level 0 of the office_1 world, as level 2 here
```

`id` defaults to the level's id in its source, and the level id may be left
out when the source has a single level. A linked level is taken as its
source defines it, defaults included. Its only other key is `prefix:`, put
before every name of the level and every reference to one, so the same floor
can repeat (`- {id: '3', from: ./floors/typical, prefix: f3_}`) and two worlds
sharing a zone name can sit side by side. Names stay
unique across the composed world, elevator destinations that point outside
it and staircase halves whose partner half is outside it lose that link with
a warning, and a link back into a world already being
read is an error. Assets local to a linked world (`<world>/assets/`) resolve
after this world's own. Saving the world writes the links, not the levels
they name. `arena asset push world` refuses a world whose `./` or `../` link
leaves its folder, or that links a world only this machine has.

A linked level cannot set the destinations of its elevators, so the
destinations that leave the composed world are dropped and the linked levels
whose elevators are then left without one ride between each other: every
linked level with exactly one such cabin joins a ring in level order (two
levels ride to each other, three as 0 to 1, 1 to 2, 2 to 0). A level with
several such cabins stays out of the ring with a warning naming them.

Any world name also takes a comma list, `world:=hospital_1,office_1`, which
stacks the listed worlds as the levels of one world without writing a file:
each level of each world links in order with ids from `0`, a level filter
picks levels (`world:=hospital_2[0],three_storied_residential[1]`), and every
name gets the world name as prefix (`hospital_1_`, a second listing of the
same world `hospital_1_2_`). The composed world is written to
`$ARENA_ASSETS_DIR/_composed/<list>/world.yaml` and loads like any linked
world, the elevator ring included. It has no scenarios, the worlds' own
scenarios name the unprefixed entities.

### Ceilings

`ceiling:` on a zone is `false` for an open zone or a mapping of `height`
(top height in metres, derived when absent), `material`, `cast_shadows`
(default false) and `lights` (the ceiling rig, see [Lights](#lights)). With no
`height`, the ceiling sits on the tallest wall standing on or within 0.3 m of
the zone polygon, at least 2.0 m.

Ceilings are opaque from below and transparent from above. Isaac hides them
unless the world declares lights, `sim.isaac.viewport.ceilings:=on|off` forces
them. The walls a zone with a ceiling borders or holds reach it, and a door
lower than the ceiling gets a visual-only lintel above it. They are
visual-only (no collision). With `cast_shadows` false the ceiling does not
occlude the sun, so interiors stay lit without global illumination.

### Doors and elevators

A `doors:` entry sits on the level: `name`, `span: [[x, y], [x, y]]`, plus
its kind, timing and material. Full field reference:
[Door fields](worlds/README.md#door-fields).

An `elevators:` entry is one cabin on its level: `name`, `position`, `size`,
`door_side` (inferred from the nearest wall when absent) and `destinations`,
the names of the cabins a ride goes to, on any level, so a shaft is one cabin
per level linking the others and two cabins on one level make a teleporting
pair. Full field reference: [Elevator fields](worlds/README.md#elevator-fields).

A cabin with several `destinations` goes to the one in its writable semantic
field `target` (a cabin name, e.g. set by a robot task or a scenario
timeline). With `target` empty it cycles round-robin over the list, one step
per departure. A written `target` stays until rewritten, writing `""` returns
to round-robin. A single-destination elevator publishes `target: ""`.

Elevator `capacity: <int>` (default `null` = unlimited) caps the cabin: while
the agents inside (robots plus pedestrians) are at or above it, outside calls
are ignored, so the door closes after `hold_time` and the cabin departs with
whoever is inside. Departures from several cabins to the same destination queue
up and arrive in order. A rider whose teleport fails (refused by the sim, or
not seen to move within 10 s sim time) is released: the source cabin unseals
and retries the ride, the destination stops holding its door for them.

Elevator `call_mode: level | edge` (default `level`) sets how agents outside
the cabin call it. `level`: any agent within `activation_distance` of the hall
door calls every tick, so the door stays open (and a pending departure is
cancelled) for as long as anyone waits there. `edge`: a call only (re)opens the
door when an agent newly enters the call radius, or when the door is shut and
no departure is pending (nobody new inside). Agents merely standing in the
lobby no longer hold the door: it closes after `hold_time` (the slab safety
reopen still applies) and the cabin departs. Use `edge` for crowded lobbies.

A door's `activation_distance` takes a scalar or a pair `[a, b]` (the larger
value is the trigger radius).

Door `kind` is one of `sliding` (default, slab slides along the span),
`sliding_top` (slab lifts), `teleport` (slab vanishes) and `hinged`: the slab
rotates about the first span point by up to 90 deg over `transition_time`.
`swing_direction: 1` (default) opens towards the left normal of the span,
`swing_direction: -1` towards the right. Like sliding doors, a hinged slab
never moves into an agent: an opening step that would hit someone in the swept
arc holds, a close with anyone in the remaining arc is refused and the door
re-opens (`blocked_close`).

```yaml
doors:
- name: office_door
  span: [[2.0, 0.0], [2.9, 0.0]]
  kind: hinged
  swing_direction: -1
elevators:
- name: hub
  position: [5.0, 0.0]
  destinations: [elev_a, elev_b]
  capacity: 4
  call_mode: edge
```

Pedestrians see mechanisms as walls where they cannot pass: cabin back and
side walls always, and a door's segment while it cannot open for them (a
gate effectively locked with no authorized agent near, or
`activation_distance <= 0` with no pressed plate driving it).

### Stairs and escalators

Stairs is the umbrella for both kinds of staircase: fixed stairs under
`stairs:` and moving ones under `escalators:`. Full field reference:
[Stairs fields](worlds/README.md#stairs-fields).

#### Placing a staircase

A `stairs:` or `escalators:` entry sits on the lower of the two levels it
joins and names one whole staircase:

| key | meaning |
| --- | --- |
| `name` | unique across the world |
| `mouth` | `[x, y]` middle of the open lower end, in the level frame |
| `mouth_side` | side it is entered from (`+x`, `-x`, `+y`, `-y`, default `-x`), the staircase running away from it |
| `size` | `[length, width]` of the whole staircase in metres, length along the walking direction |
| `to` | id of the level it reaches, above its own |
| `exit` | spread worlds only: `[x, y]` middle of the open upper end on the `to` level, in that level's frame |
| `kind` | optional staircase kind to start from, see below |

The rise is the difference of the two levels' `elevation`, a level without
one sitting a storey (3 m) above the level before it. An escalator with
`direction: down` still sits on the lower level with its mouth at the bottom,
`direction` only sets which way it carries.

The world's `layout` says how its levels relate. In a `stacked` world every
level shares one floor plan and the staircase runs straight up from its
mouth. In a `spread` world (the default) every level keeps a floor plan of
its own, and the entry also takes `exit`.

`size` sets the incline, nothing else does, and nothing rejects a steep one:

| staircase | incline | typical | `size` length for a 3 m rise |
| --- | --- | --- | --- |
| stairs | `atan(rise / length)` | 30 to 35 degrees | 5.2 m (30 degrees) |
| step escalator | `atan(rise / (length - 2 * landing_length))` | 30 degrees | 6.8 m (30 degrees with 0.8 m landings) |
| belt escalator | `atan(rise / (length - 2 * landing_length))` | up to 12 degrees | 15.7 m (12 degrees with 0.8 m landings) |

`width` is the distance between the center lines of the two side walls,
which are 5 cm thick. Real stairs run 1.0 to 2.5 m wide and escalators 0.8 to
1.2 m. An escalator's moving treads leave 6 cm clear on each side, and its
brushes reach 5 cm in from each side wall. Keep about half a metre of clear
floor before each escalator mouth: the pit there is one tread deep along the
axis (about 0.3 m).

#### What loading builds

Loading splits a staircase at its midpoint into two halves, `<name>_bottom`
on its level and `<name>_top` on the `to` level. `<name>` names the whole
staircase, `<name>_bottom` and `<name>_top` each half, as places for
`robot in` atoms and zone references in scenarios, and the KG places each
half in the zone holding its center with `connects_floor` to the other. A
linked level prefixes all three names alike. Left and right always mean as
seen walking up from the mouth, loading mirrors them for the top half.

Stairs are solid steps, the bottom half rising from the floor and the top
half a well sinking from the upper floor to meet it. Floors and ceilings open
around a staircase by themselves:

- the upper floor over the top half, and in a stacked world also over the
  end of the bottom half where less than `headroom` (default 2.1 m) is left
  under it, with guard walls in the style of the sides around that overhang
  on the upper level,
- the lower floor under an escalator, whose treads or planks run level with
  the floor over a landing at each mouth (`landing_length`, default 0.8 m), and
  a short pit before each escalator mouth,
- in a stacked world, the lower level's ceilings over the overhang and the
  top half.

Zone ceiling heights are measured from the level's own floor. The floor of
the level above is a separate slab and does not stand in for a ceiling.

An escalator's treads or planks come out from under a landing plate flush
with the floor at each mouth, 1 mm low until they clear it, then run level
over the landing and rise into place. A belt's planks tilt along the ramp and
bend over its ends plank by plank. The plate is a comb: treads pass through
it, robots and pedestrians stand on it.

#### Dressing

`material` covers every part of a staircase (default `Marble`), and
`surfaces` sets one part apart:

| surface | parts | default |
| --- | --- | --- |
| `steps` | stair treads, escalator steps, belt planks | `material` |
| `risers` | stair risers, setting it makes each step a 2 cm tread plate on a riser body | `material` |
| `landings` | an escalator's landing plates | `material` |
| `fixture` | an escalator's base, the side linings under the walking surface, the seam wall | `material` |
| `brushes` | the skirt brushes along both inner sides of an escalator, just above its treads | `Carpet_Charcoal` |

A surface takes any material name (`arena asset ls material` lists them, for
example `Aluminum_Anodized`, `Carpet_Charcoal` (near black), `Concrete_Smooth`,
`Marble`, `Oak_Planks`, `Paint_Gloss` and `Paint_Matte` (white),
`Plaster_Wall`, `Walnut_Planks`), or `[Name, {tint: "<css colour>"}]` for a
tinted copy: the colour, such as `rgba(90, 90, 90, 0.8)` (0 to 255 per
channel), is blended over the material's textures with its alpha as the
strength, so an alpha of 1 gives a solid colour.

`sides` takes `left` and `right`, each with a wall `kind` or a `material`
(default the staircase's `material`) and a `height` above the walking surface
(default 1.0 m, 0 for no side wall). On the bottom half a side follows the
incline, on the top half it guards the well along the upper floor. A wall
kind on a side keeps its layers and tiled models along it: a layer starting
above `height` is dropped, one crossing it is cut there, and the layer
reaching highest is stretched up to `height` if it falls short. Each layer
keeps its own `width` (thickness).

`rails` lists handrails along the walking surface. Each takes `across`
(`left` and `right` sit 8 cm inside the side walls, a number is metres left of
the axis), then either a `material` (default the staircase's `material`) for a
5 cm bar whose top sits at `height` (default 0.9 m), or a wall `kind` whose
parts stand on the walking surface, cut and stretched to `height` the way a
side's are. The package ships the wall kind `handrail` (in
`arena_simulation_setup/assets/Common/Wall/`, resolved without copying): a
5 cm bar at 0.85 to 0.9 m on 3 cm posts every 0.9 m. A plain bar has no posts.
`solid: true` works on both forms: such a rail also blocks the floor plan, so
a center rail at `across: 0` splits a staircase into two lanes of equal
width for robots and pedestrians. Other rails are seen and collided with but
not routed around.

Every part of a staircase can carry a captured look, such as a 3D Gaussian
splat (`arena-assets splat --object`), which Isaac renders and the other
simulators skip:

- `steps`, `risers`, `landings` and `fixture` take `{material, model}` in
  place of a bare material. The `model` (any object) is placed on the middle
  of the top face of every box of that surface, turned with it: on each step
  and riser, each landing plate, each base slab, and on each moving tread,
  which it rides. The box keeps its `material` and stays the collision,
  lidar and backdrop, the way a splat wall keeps its segment.
- `models` lists objects for the staircase as a whole, each with `model`,
  `at: [along, across, z]` (default `[0, 0, 0]`) from the middle of the
  mouth on the lower floor, `along` running up the whole staircase, `across`
  to the left, and `yaw` from the axis (default 0). A capture of a whole real
  staircase goes here, as does anything that dresses the brushes or the
  linings.
- Side walls and rails carry anything a wall kind can, a splat included.

```yaml
escalators:
- name: esc_up
  kind: escalator
  mouth: [2.0, 7.2]
  size: [6.8, 1.0]
  to: '1'
  surfaces:
    steps: {material: Aluminum_Anodized, model: mall_escalator_step}
  models:
  - {model: mall_escalator_shell, at: [3.4, 0.0, 0.0]}
```

#### Staircase kinds

A staircase kind keeps a build and a look for reuse, the way a wall kind
keeps a wall style. `kind: <name>` on an entry reads
`<domain>/Stairs/<name>/<name>.yaml` (a bare name under `Common`), searched in
the world's own `assets/`, the `assets/` of the worlds it links levels from,
the local asset root (`$ARENA_ASSETS_DIR_LOCAL`), the package's own
`arena_simulation_setup/assets` and then the buckets. `arena asset ls stairs`
lists them and `arena asset find stairs <name>` shows which source answers.

A kind file is a flat mapping of the keys `style`, `speed`, `tread`,
`landing_length`, `plank`, `headroom`, `material`, `surfaces`, `sides`,
`rails` and `models`, in the same form as on an entry. Any other key, `kind` included, is
an error named by the entry's YAML path, so kinds do not chain. Placement
(`mouth`, `mouth_side`, `size`, `to`, `exit`, `direction`) always stays on
the entry.

```yaml
# assets/Common/Stairs/office_stairs/office_stairs.yaml
material: Walnut_Planks
surfaces: {risers: Paint_Matte}
sides: {left: {material: Paint_Gloss}, right: {material: Plaster_Wall}}
rails:
- {across: left, kind: handrail}
- {across: right, kind: handrail}
```

The entry's own keys win over the kind's. `surfaces` and each side merge key
by key, a side's own `kind` or `material` replacing the kind's, and the
entry's `rails` and `models` replace the kind's whole lists. Writing the world back keeps
`kind` and only the keys the entry sets itself. An entry without `kind` takes
the defaults listed in the field reference.

The package ships three kinds in `arena_simulation_setup/assets/Common/Stairs/`:

| kind | style | build and look |
| --- | --- | --- |
| `stairs` | steps | `handrail` rails on both sides, everything else default |
| `escalator` | steps | `Aluminum_Anodized` steps and landing plates, `Concrete_Smooth` fixture, `Paint_Gloss` sides, `Carpet_Charcoal` handrails at 1.0 m, everything else default |
| `belt_escalator` | belt | as `escalator` with `Carpet_Charcoal` planks |

The kinds set no `speed`, `tread`, `landing_length` or `brushes`, so those
take their defaults unless the entry sets them.

```yaml
escalators:
- {name: esc_up, kind: belt_escalator, mouth: [2.0, 5.0], size: [14.0, 1.0], to: '1', speed: 0.6}
```

#### Moving stairs

Stairs run both ways. An escalator carries riders `direction: up` (default)
or `down` at `speed` (default 0.5 m/s), and its `style` is `steps` (moving
step blocks) or `belt` (an inclined conveyor of moving planks). Stairs take
only `steps`. `tread` sets the step depth a half aims for (default 0.28 m): a
half takes `round(length / 2 / tread)` treads, and the riser is the half rise
over that count. `plank` (default 0.03 m) is a belt's plank thickness.
`size`, `speed`, `tread`, `plank` and `headroom` are positive,
`landing_length` lies in `[0, length / 2)`, and `to` names a level above.
Loading names the YAML path of a broken entry.

```yaml
layout: stacked
levels:
- id: '0'
  zones:
  - {name: lobby, polygon: [[0, 0], [20, 0], [20, 10], [0, 10]]}
  stairs:
  - {name: stairs, kind: stairs, mouth: [6.0, 2.0], size: [5.2, 2.0], to: '1'}
  escalators:
  - {name: esc_up, kind: belt_escalator, mouth: [2.0, 5.0], size: [14.0, 1.0], to: '1'}
  - {name: esc_down, kind: escalator, mouth: [6.0, 7.2], size: [6.8, 1.0], to: '1', direction: down}
- id: '1'
  elevation: 3.0
  zones:
  - {name: gallery, polygon: [[0, 0], [20, 0], [20, 10], [0, 10]]}
```

This is the `stairs_demo` world without its stair dressing: the belt rises
3 m over 14 m (13.6 degrees past its landings), the stairs at 30 degrees and
the step escalator at 30 degrees past its landings.

#### Stacked levels

`world.levels:=stack|spread|auto` (default `auto`) chooses how the simulator
places the levels. `spread` lays them side by side, and a crossing of a seam
teleports to the other half. `stack` puts each level of a stacked world at
its elevation over the shared floor plan, and each stair is one body walked
and ridden straight through, without a seam wall. `auto` stacks a stacked
world and spreads a spread one, `stack` on a spread world fails. The map,
the pedestrian simulation, the judge and evaluation keep the side-by-side
layout in either case: a robot's `map -> odom` follows the level it stands
on, and pedestrians reach the simulator on `sim_peds` in its own frame.

### Microphones

A level's optional `microphones:` list places audio listeners for the
acoustics pipeline. Full field reference:
[microphones](worlds/README.md#microphones).

### Semantic annotations

Door, elevator and articulated object state is intrinsic: every spawned
door, elevator and articulated object publishes its full vocabulary with no
annotation needed:

| kind | discrete | continuous | predicates | members |
| --- | --- | --- | --- | --- |
| door | `state` | `progress` | `open`, `in_transit`, `triggered`, `blocked_close` (a close is refused because an agent is in the slab path) | agents within the activation radius |
| elevator | `cabin_door`, `target` (writable, see above) | `arriving_eta`, `occupants` (riders committed to a ride), `cabin_door_progress`, `live_occupants` (agents inside now), `capacity` (-1 = unlimited) | `departing`, `in_transit`, `dispatched`, `just_arrived`, `full` | agents inside the cabin now |
| articulated | `state`, `target` (writable), `setting.<joint>` (see [Articulated objects](#articulated-objects)) | `openness.<joint>`, `pos.<joint>`, `angle.<joint>`, `q.<joint>` | one per state name, `moving`, `blocked` | none |

Agents are named as the runtime tracks them: robots by sim path, pedestrians
by their ped name. `semantics:` on a `doors:`/`elevators:` entry is only for
attaching a *scripted* kind (`gate`, `pressure_plate`) to that entry. Zones
accept an optional `semantics:` list of literal state/predicate primitives:

```yaml
zones:
- name: lobby
  polygon: [...]
  semantics:
  - {state: max_speed, value: 1.5}
  - {predicate: quiet, value: true}
```

An `Elevator` entry configures its fire-recall regime as a first-class field,
not a semantics annotation:

```yaml
elevators:
- name: 1_elevator
  position: [5.0, 0.0]
  destinations: [2_elevator]
  recall_on: alarm
```

Full field reference: [worlds/README.md](worlds/README.md).

### Articulated objects

An object whose annotation carries `articulation:` (links, joints, named
states, built by arena_assets) gets an `articulated` semantic entity under
its env-prefixed name. MuJoCo, Isaac and Gazebo spawn it as one physics
model: its root fixed in place, each other link a body on a damped joint
with friction, so robots push its doors and drawers and the fields read the
joints the simulator measures. flatsim and dummy spawn one static obstacle
per link and move the links to the joint values. Conditions and timelines name it as authored, a level
object of world.yaml as much as a scenario object, and a scenario object
takes the name over from a world object of the same name. Its `joints:` key
sets the joint values every episode starts from (see the objects keys in
[worlds/README.md](worlds/README.md)).

| field | role | meaning |
| --- | --- | --- |
| `state` | discrete | the first state all of whose joint values the joints hold, empty between states |
| `target` | discrete, writable | the state the targets match, empty otherwise. Writing a state name sets the targets of the joints that state names, the others keep theirs |
| `openness.<joint>` | continuous, writable | an opening part (a joint the `closed` and `open` states both name): 0 closed, 1 open, its distance from `closed` over the distance from `closed` to `open`, so a lid swinging either way from a closed rest reads open both ways. Writing a fraction in [0, 1] sets its target, writing bare `openness` sets every opening part |
| `pos.<joint>` | continuous, writable | any other limited joint (knob, dial, handle, button): 0 at its lower limit, 1 at its upper. Writing a fraction sets its target |
| `angle.<joint>` | continuous | an unlimited joint (wheel, endless knob): its value wrapped to [0, 2 pi) |
| `setting.<joint>` | discrete, writable | a joint with named `detents` (in the object's annotation, see [Object articulation](../arena_assets/README.md#object-articulation)): the detent it sits at, empty between them. Writing a detent name moves it there |
| `q.<joint>` | continuous, writable | every movable joint's raw value (radians or meters). Writing sets its target, clamped to its limits |
| `<state>` | predicate | the joints hold that state's values |
| `moving` | predicate | a joint is away from its target, or in a physics simulator moved since the last tick |
| `blocked` | predicate | a driven joint is held: in a physics simulator something keeps it from following its drive, in flatsim and dummy a moving link would have entered a robot or pedestrian disc |

Every field is named `<family>.<joint>`, so the parts of an object are
listed by family without knowing its joints, and `assets_articulation` lists
the families and their members. The family names, `state`, `target`,
`moving` and `blocked` are reserved: no state, joint or detent takes one.
Joints move toward their targets at their `velocity` (1 rad/s or 0.25 m/s
when unset). In a physics simulator a written target switches on a drive
of bounded effort, which follows a reference ramped at that velocity and
lets go once the joint is within 0.05 rad or 1 cm, setting it onto the
target. A joint no drive holds moves only when pushed, and its target
follows it. Link masses come from the link boxes (150 kg/m3, 0.2 to 20 kg).
A joint's friction, damping, armature and spring are the ones its asset
authors (see [Object articulation](../arena_assets/README.md#object-articulation)),
else its role's (a drawer rubs harder than a bare slide), else its kind's,
and a default friction holds the joint against its own weight, so a lid or
flap stays where it is left. A sprung joint swings back to its rest when no
drive holds it, and a link's material sets the friction and bounce of its
surfaces. An unknown state or detent name, a fraction outside [0, 1] or a
non-number joint value is refused as a malformed value, and an episode reset returns every joint to its initial
value. Loading a scenario under the `scenario` obstacles mode, and the MCP
`worlds_scenario_validate`, check every `joints:`, every field and family a
condition names and every timeline `set` against the articulated objects of
the scenario and its world, and fail naming each one they refuse. A scenario
timeline opens a fridge at `t=5s`, and a phase or episode condition judges it
with the entity atom, or by family without naming its doors:

```yaml
timeline:
- at: 5.0
  set:
  - {entity: fridge, field: target, value: open}
robots:
- start: [1.0, 1.0, 0.0]
  phases:
  - {goto: kitchen}
  - {until: "fridge.state == open"}
  - {until: {each: fridge.openness, ge: 0.9, for: 2}}
conditions:
- {op: eventually, p: {field: oven.setting.dial, eq: "off"}, text: "Turn the oven off."}
```

### KG

What a world means, as opposed to what the runtime executes, lives in an
optional `worlds/$WORLD/kg.yaml` beside `world.yaml`: a knowledge-graph
scene of entities, facts and constraints (`format: arena-kg/1`), typed
by the vocabulary packages in
[kg/packages](src/arena_simulation_setup/kg/packages) and shared
with the airchitect world generator.

```yaml
format: arena-kg/1
uses: [core@2]
entities:
- {id: room:reception, props: {type: Main Entrance, description: A spacious reception and waiting hall}}
- {id: room:triage}
- {id: obj:desk_1, props: {label: Reception Desk}}
- {id: obj:kiosk_0, props: {label: Information Kiosk, placed: false}}
facts:
- {subject: room:reception, verb: contains, object: obj:desk_1}
- {subject: room:triage, verb: contains, object: obj:desk_1}
- {subject: room:emergency, verb: left_of, object: room:triage}
```

- Ids are `<kind>:<world name>`: `room:<zone>`, `door:<door>`, `obj:<object>`,
  `light:<light>`, `sound:<sound>`, `elevator:<cabin>`, `floor:<level id>`.
  Those must name something in `world.yaml`, except an `obj` with
  `placed: false` (an object a generator asked for but never placed). Roles,
  regions, agents, profiles, purposes and materials live only here.
- The KG carries no geometry (`boundary`, `line`, `pose`, `model`, ...):
  `world.yaml` owns it.
- Membership is `contains` (a thing may sit in several zones), connectivity
  `room connects room` with `door: door:<name>`. Route instructions, the
  acoustic room graph and the prompt modes read them.
- Facts are stated (`source: generated`, the default) or derived (`source:
  derived`). On load, derived facts are recomputed: a door belongs to the
  zones whose boundary it lies on and connects them, an object, light, sound
  or elevator to the zones holding its position. Only things no stated fact
  covers get derived ones, so a stated `contains` always wins.
  `kg.derive:=all` ignores the stated memberships and connections of
  things in `world.yaml` and derives them all, the rest of the KG stays.
  `kg.derive:=off` keeps the stated facts only, `auto` is the default.
- A linked level brings the KG of its source world along: its
  entities, stated facts and constraints join this world's wherever what they
  name is part of this world, `floor:<id>` follows the level's new id, and
  this world's own KG wins every conflict. Saving writes only this
  world's own part back.
- What people use a zone for is `room has_purpose purpose:<work|break|transit|private|unknown>`
  (package `space@1`). The scenario generator places its crowds by it, a
  `params: {by: llm, ...}` marks a model's proposal that a human has not
  reviewed, see
  [generative/scenario](src/arena_simulation_setup/generative/scenario/README.md#zone-roles).
- Nothing in the KG drives the runtime. Mechanism semantics (presets,
  `light_on`, `recall_on`, zone states) stay in `world.yaml`.

### M2 kinds, params, and timelines

M2 adds scriptable/derived kinds on top of the intrinsic `door`/`elevator`
vocabularies and `zone`. Each kind takes its config under a `params:` dict on
the primitive or preset item, which round-trips opaquely (omitted on
serialize when empty):

| kind | attaches to | preset expansion | `params` keys |
| --- | --- | --- | --- |
| `signal` | a world `signals:` entry | `state`, `phase_remaining`, `stop` | `phases` (list of `{name, duration}`), `stop_phases`, `regime`, optional stop line `position` (`[x, y]`, level frame) and `radius` (evaluation only, `ran_red_signal`) |
| `schedule` | a world `schedules:` entry | `state`, `active`, `window_remaining` | `windows` (list of `{start, end, value}`), `default`, `regime` |
| `gate` | a `doors:` entry | `locked`, `blocked` | `authorized` (sim_paths/robot names), `unlock_on`, `locked` (initial value, default `true`: a gate authored without it starts locked and logs so) |
| `pressure_plate` | a `doors:`/`elevators:` entry (own `position`) | `pressed` | `position` (`[x, y]`, level frame like all geometry), `radius`, `drives` (a door name, `<level_id>.<name>` for a door on another level), `latch`, `press_on`, `regime` |
| `occupancy_cap` | a `zones:` entry | `occupancy`, `cap`, `over_cap` | `cap` |
| `sound` | a `sounds:` entry (level or scenario) | `sounding`, `volume_db` | `sound_on`, `regime`, `sounding`/`volume_db` (initial values) |
| `light` | every `lights:` entry and every zone ceiling rig, attached from the light's own fields | `lit`, `dim`, `dead_fraction` (rigs only) | `light_on` |

`signal` and `schedule` have no geometry, so they sit at world scope beside
`levels:`. A `sound` needs a placement: exactly one of `position`, `attach`
(name of an object of the world, whose yaw frame `offset` is applied in) or
`frame` (a TF frame the sound rides, `offset` local to it, so `frame:
jackal/base_link` is a speaker bolted to that robot). Other `Sound` fields:
`asset_id` (required, the audio catalog entry), `loop` (default `true`) and
`reference_distance_m` (default `1.0`, must be positive, sets the falloff
reference distance).

```yaml
schedules:
- name: fire_alarm
  semantics:
  - {preset: schedule, params: {windows: [], regime: alarm}}
signals:
- name: crosswalk_light
  semantics:
  - {preset: signal, params: {phases: [{name: go, duration: 20.0}, {name: stop, duration: 10.0}]}}
levels:
- id: '0'
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
is rejected at load: there is no single primitive it could mean. Presets are
written back as authored.

A scenario may carry its own `sounds:` list with the same schema. Those are
episode-scoped: attached at reset, gone at the next one, and their `attach`
may also name one of the scenario's own `objects:`.

#### Lights

A light is one fixture, an ambient light, or a zone's ceiling rig. Each is a
`light` semantic entity with `lit` (on/off), `dim` (dimming in 0..1) and,
for rigs, `dead_fraction` (share of dead fixtures), all writable through
`semantics/set` and the scenario timeline and recorded with every other kind.
Worlds that declare no light render as before. A world that declares any
light gets no default lighting at all, so what it declares is all there is
and a blackout is dark. Lights that objects carry (below) do not count as
declared, and neither do the lights a robot carries, so a lamp or a headlight
adds its light on top of the default lighting.

| Where | Fixtures | Placement |
| --- | --- | --- |
| level `lights:` | `dome`, `sun` | none, a `pose` aims the `sun` |
| level `lights:` | `panel`, `tube`, `downlight`, `spot`, `bulb` | exactly one of `pose`, `attach` or `frame`, the latter two with an `offset` |
| zone `ceiling: {lights: ...}` | `panel`, `tube`, `downlight`, `spot`, `bulb` | a grid at `spacing` metres over the zone polygon, just below the ceiling |
| object `annotation.yaml` `lights:` | `panel`, `tube`, `downlight`, `spot`, `bulb` | `pose` in the object frame, moved, turned and scaled with every placement |
| scenario `lights:` | `panel`, `tube`, `downlight`, `spot`, `bulb` | as for level `lights:`, `attach` names one of the scenario's `objects:`, `level` the floor of a literal `pose` |
| robot `model_params.yaml` `lights:` | `panel`, `tube`, `downlight`, `spot`, `bulb` | `frame` of the robot plus `offset`, see [robots/README.md](../arena_robots/arena_robots/robots/README.md#model_paramsyaml) |

`Light` fields: `name`, `fixture`, `lumens` (local fixtures, per fixture for a
rig) or `lux` (`dome`, `sun`), `cct_K` (default 6500), `cast_shadows` (default
false), `cone_deg` (`spot` only, default 40), and one
of `light_on` (a regime, `!` negates) or `lit` (initial value, default true),
plus `dim` (default 1.0). A ceiling rig takes `fixture` (default
`panel`), `spacing` (default 2.4), `lumens`, `cct_K`, `cast_shadows`,
`light_on`/`lit`, `dim` and `dead_fraction`, needs a ceiling, and is
named after the zone. The same fixtures of a rig die at the same
`dead_fraction` in every run, and raising it only adds dead ones.

`pose` places a fixture in the level frame, with a height, since a light sits
above the floor, `offset` in the frame of the object it is attached to or of
its TF `frame`. A fixture emits along the local -z axis of that pose, a
`panel` or `tube` lying along its local x and y, so a pose without an
orientation shines straight down and `yaw` turns a tube about the vertical.
`bulb` and `dome` take no orientation, a `sun` takes a `pose` whose position
is zero. A frame light follows its TF frame, so `frame: jackal/base_link`
with `offset: {x: 0.2, y: 0, z: 0.3, pitch: -1.42}` is a headlight that looks
ahead and slightly down. A scenario, robot or episode object light whose name
a spawned light already holds is left out with a warning.

A world without lights can still run lit: `world.lighting:=auto` gives every
zone that has a ceiling, no ceiling rig of its own and no level light placed
inside its polygon a rig of 7200 lm panels at 2.4 m spacing, without touching
the world file. The default `world.lighting:=authored` renders exactly what
the world declares.

```yaml
levels:
- id: '0'
  zones:
  - name: central_hallway
    polygon: [[8, 0], [12, 0], [12, 34.2], [8, 34.2]]
    ceiling: {lights: {fixture: panel, spacing: 2.4, lumens: 3600, light_on: "!blackout"}}
  lights:
  - {name: ambient, fixture: dome, lux: 5}
  - {name: exit_strip, pose: {x: 10, y: 34, z: 2.2}, fixture: tube, lumens: 300, light_on: blackout}
schedules:
- name: power
  semantics:
  - {preset: schedule, params: {windows: [{start: 60, end: 90, value: out}], regime: blackout}}
```

An object's `annotation.yaml` can carry `lights:` entries with `name` (default
`light`), `fixture` (default `bulb`), `pose`, `lumens`, `cct_K`,
`cone_deg`, `glow` and `link`. `arena_assets` writes them when it builds an object whose
source model contains lights (hand-written entries win, and the built models
carry no light themselves). A light of an articulated object carries `link:`,
the link of the source node it sits under (the root link otherwise), and
follows that link as the joints move: its position rides on the link and its
beam turns by the yaw of the link's turn only, so the beam of a tilting link
keeps its yaw part. `glow` names the material that glows with the
light, a lamp shade for instance: Isaac, Gazebo and MuJoCo make it emit the
light's color scaled by its dim, and it is dark while the light is off. Every
placement of the object, in a level's `objects:`, a scenario's `objects:` or
by an obstacle mode, then gets the light `<entity>_<name>` for as long as the
placement exists, and a `light:` key on the placement sets its `lit`,
`light_on` or `dim`:

```yaml
objects:
- {name: desk_lamp, model: Residential/Desk_Lamp, pose: {x: 2, y: 1, z: 0.75}, light: {light_on: "!daylight"}}
```

A scenario dims the hall with `{entity: central_hallway, field: dim, value:
"0.3..1.0"}` in its timeline. Isaac renders every fixture as an area light, and
shades asset materials that only emit a texture by that texture, so they
darken with the lights.
Gazebo renders a rig as one shadowless point light per 2 by 2 block of
fixtures, plus one unattenuated fill at the room center for the light the
room surfaces reflect (40 % reflectance), since its renderer traces no
indirect light.
MuJoCo shades with at most seven lights per view and without shadows from
fixtures, so each camera gets the fixtures merged the coarser the smaller they
look from where it stands, plus the fill of the rig whose room it stands in.
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
  span: [[4.0, 1.55], [4.0, 2.45]]
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

`scenario.yaml` accepts an optional `timeline:` list. Each entry fires its
actions against exactly one trigger and needs at least one action:

| trigger | meaning |
| --- | --- |
| `at: <seconds>` | fire once at episode second `t` |
| `every: <seconds>` | fire at `offset` (default: one period in), then each period, optional `until` |
| `when: <condition>` | fire on the false-to-true edge of a condition, a text one or a structured node as in `until` (see [Add a scenario](#4-add-a-scenario)), `{some: fridge.openness, ge: 0.9, for: 1}` included |
| `when: {entity, field, is}` | the same as `when: {field: <entity>.<field>, eq: <is>}` |

In a `when` condition the subject `robot` stands for every robot of the env: the
condition holds while it holds for at least one of them, so
`not robot within 3 of 2,2` holds while some robot is farther than 3 m from
`(2, 2)`. A robot's own name (anonymous robots are named after their model,
e.g. `jackal`) judges only that robot. Without a placed robot such an atom
does not hold.

| action | meaning |
| --- | --- |
| `set: [{entity, field, value}]` | write semantic values |
| `stimulus: [{agents, need, intensity}]` | raise `need` on each pedestrian named in `agents` through humansim's `notify_stimulus`, `intensity` 1 by default (humansim sets the need to 100 x `intensity`) |

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

A `stimulus` drives behavior-tree agents whose agent type declares the need
and a transition on it, as for [pedestrian stimuli](#pedestrian-stimuli). Here
`edge_0` waits until the robot comes within 4.2 m of `(12.1, 4.0)` and the
`d_*` group switches at `t=8s`:

```yaml
timeline:
- when: "robot within 4.2 of 12.1,4.0"
  stimulus: [{agents: [edge_0], need: go, intensity: 1.0}]
- at: 8.0
  stimulus: [{agents: [d_1, d_2, d_3], need: form}]
```

Every agent of a stimulus is one of the scenario's `humans`, loading fails on
any other name. With the `arena` human backend loading also fails when the
agent type of an agent (its `agent.agent_type`) neither declares the need nor
waits for it as the `until` event of an autonomous step. A stimulus the human
simulator does not take at runtime is logged and skipped, the other agents
still get theirs. Only the `arena` human backend acts on stimuli: under any
other backend a scenario with `stimulus` entries fails to load with an error
naming `human:=arena`.

Every semantic snapshot of the env carries a `timeline` entity (kind
`timeline`). Its discrete field `stimulus.<agent>.<need>` holds the sim time
in seconds of the tick on which that stimulus was first taken by the human
simulator in the episode, and a stimulus that never landed has no field.

## 3. Generate the map

```bash
touch_world my_world
```

This reads `worlds/my_world/world.yaml` and writes a preview map per level,
`map/<level id>.png` and `map/<level id>.yaml`. The runtime does not need
them, it renders its own map from `world.yaml`.

To regenerate with visible static obstacle footprints:

```bash
touch_world my_world \
    --assets red \
    --assets-label blue \
    --assets-bbox "((-.5, .5), (-.5, .5))"
```

To rewrite `world.yaml` itself in canonical form (short forms, rounded
numbers, defaults left out) with its `kg.yaml`:

```bash
touch_world my_world --all
```

`touch_world` accepts `--resolution <float>` (metres per pixel, default `0.05`).

### Format 1 worlds

Worlds written before format 2 (one `<level id>/world.yaml` of `zones:` per
level, walls, doors, objects and lights nested in zones) still load, with a
deprecation warning, through the legacy loader in
[tree/World/legacy.py](src/arena_simulation_setup/tree/World/legacy.py). It
hoists zone walls, doors, elevators, entities, sounds and lights to the
level, schedules and signals to the world, turns `destination: "1.a"` into
`destinations: [a]`, drops exact repeats, renames different things that share
a name, and keeps zone descriptions and which zone listed which thing in the
KG. A `graph.yaml` beside a format 1 world (the airchitect scene
graph) is lifted into the KG too: a node becomes a room with its
`type`, `description` and attributes, a wish-list asset an unplaced `obj`, and
an edge a connection through the door named `door_<edge id>` or
`door_<a>__<b>` plus its position facts (an edge with neither door states no
connection, geometry derives it). Migrating writes it into `kg.yaml`
and removes `graph.yaml`. Format 1 scenarios (`static:`/`dynamic:`, the `obstacles:`
wrapper, `goal:`, `start_floor`) load the same way. Rewrite a world and its
scenarios in place with:

```bash
touch_world my_world --migrate
```

Nothing writes format 1.

## 4. Add a scenario

Create `worlds/$WORLD/scenarios/default/scenario.yaml`:

```yaml
format: arena-scenario/2
robots:
  - start: [1.0, 1.0, 0.0]
    phases:
      - {goto: [8.0, 8.0, 0.0]}
```

An empty scenario (robot start and phases only) is valid, and `robots:` is
optional: a robot without an entry stays parked, an episode with only parked
robots runs until `task.episode.timeout`. Phases run in order and are `{goto: <pose or zone/door/elevator/pedestrian name>}`, `{gesture: <name>}`,
`{reach: <named target | random | pose>}`, or a hold phase with neither (the robot
stays where it is), or `{loop: [<phases>], until: "<...>"}` (the phases repeated
in order until the `until` holds, which drops the running phase and continues
after the loop). Every phase takes `until: "<clause>[ or <clause>...]"`, a
clause being `<atom>[ and <atom>...][ for <seconds>]` (completes once any clause
has held, every atom together, for its dwell, after arrival for a goto, so
`caller plays beckon for 0.5 or robot within 2.5 of caller`), or a structured
node (`until: {some: fridge.openness, ge: 0.9, for: 2}`, see the structured
conditions of [worlds/README.md](worlds/README.md)), `hold_time` (park
seconds on a goto), `signal`
(the robot ends the goto itself by sending that signal, such as `arrived`, and
is judged against the tolerance at that moment), `conditions:` (clauses judged
over this phase only), `text:` (the instruction handed to language-conditioned
planners, `instruction:` is accepted as the same key) and `on_failure`. A robot
entry may also carry `level` (the level it starts on), `goal_level` (the level
its goto phases aim at) and `conditions:` judged over its whole phase list.

```yaml
robots:
  - start: [1.0, 1.0, 0.0]
    phases:
      - {goto: kitchen}
      - {until: "not visitor_1 in kitchen"}
      - {goto: sofa, hold_time: 2, conditions: [{op: never, p: "robot in hallway"}]}
      - loop:
        - {goto: [10.0, 18.0, 1.5708]}
        - {goto: [10.0, 2.0, -1.5708]}
        until: "robot sees caller within 12 and caller plays beckon for 0.5"
      - {goto: caller, tolerance_radius: 2.5, hold_time: 0}
```

Atoms are the `conditions:` grammar of [worlds/README.md](worlds/README.md):
`<entity>.<field> == <value>`, `<subject> in <zone>`, `<subject> within <r> of
<subject>`, `<subject> sees <subject> [within <r>]` (in range, default 12 m, with
no wall of the rendered wall grid between them), `<ped> plays <clip>` and `not
<atom>`. Either `within` subject may also be a point `x,y` (no spaces), in the
frame of a scenario pose without `level`, so `robot within 4.2 of 12.1,4.0`.
A goto to a pedestrian by name is judged by `tolerance_radius` around the
pedestrian.

Add static obstacles under `objects:` and pedestrians under `humans:`, with
the same schema as a world level's lists plus `level` (the level an entry
stands on, multi-level worlds). An object's `humansim:` block (`type`,
`capacity`, `satisfies`, `interaction_radius`, `formation`, `seats`) is
forwarded to humansim as its WorldObject, so a `go_to` step or an interaction
`target:` can name the entry by `type`.

For HumanSim (arena_humansim) pedestrians include an `agent:` block. The
`agent_type` is either a built-in type (`adult`, `elder`) or a path-relative
agent-type YAML file (`./doctor.yaml`) defining scripted behavior
(sequences, interactions). `waypoint_mode` (`repeat`, `reverse`, `once`,
`random`) sets how the waypoints replay, and a `hunav:` block carries the
HuNav backend's own keys:

```yaml
humans:
- name: agent_1
  model: female_adult_medical_01
  agent: {agent_type: adult, desired_velocity: 1.2}
  pose: [2.0, 5.0, 0.0]   # yaw in radians
  waypoints:
  - [2.0, 5.0]
  - [8.0, 5.0]
```

A scenario can also carry `regions:` (named pedestrian source/sink spawners
for arena_humansim) and `conditions:` (episode success/failure clauses). Its
`sounds:`, `lights:` and `timeline:` keys are the ones covered above. Scenario
lights count as declared, like the world's, and go at the next reset, and so
do its regions. See
[worlds/README.md](worlds/README.md) for the full key list of every
top-level key.

### Robots: floors, elevator rides, mechanism map

Robot poses are level-local when the robot entry carries `level` /
`goal_level` (the level ids of a multi-level world): `start` is shifted by
`level`'s origin, every `goto` by `goal_level`'s. A pose given as a
zone/door/elevator name is resolved on the merged world and is never shifted.
When a `goto` lies on another level than the robot, elevator rides are inserted
automatically (shortest ride sequence over every elevator's `destinations`).

`via:` on a `goto` phase lists elevators to ride explicitly, in order, before heading for the pose. This is how a robot
rides a same-level pair such as `elevator_ground` -> `elevator_top`. An entry is
an elevator name, or `{elevator: <name>, to: <destination>}` to pick
one destination of a multi-destination elevator (default: the destination
cabin closest to where the robot heads next).

```yaml
robots:
  - start: [2.0, 5.0, 0.0]
    phases:
      - {goto: [31.0, 22.0, 0.0], via: [elevator_ground]}
  - start: lobby
    level: "0"
    goal_level: "2"
    phases:
      - goto: [4.0, 4.0, 0.0]
        via: [{elevator: hub, to: east_lift}]
```

Each ride is its own phase: the robot drives into the boarding cabin, stays
there until the elevator teleports it, and continues once it stands in the
destination cabin. The ride is judged like any other `goto`, with the boarding
cabin as its target and `robot in <destination cabin>` as its `until` atom, and
a robot waiting in the cabin does not count as stalled. For a multi-destination elevator the robot writes the
elevator's `target` field (the destination cabin, e.g. `east_lift`) when the
ride starts and again when it boards, and writes `""`
(back to round-robin) when it arrives. Single-destination elevators are never
written.

The planner map is blind to door state by default: doors and elevators are
painted free and a closed door is only seen by the robot's sensors. Setting
the scenario task param `mechanism_map: true` (suite stage
`config: {scenario: {mechanism_map: true}}`) serves the robot's planners a map
in which every door the robot cannot open right now is occupied, through the
env's normal `map_server` (`LoadMap`, same topic as the static map). A door
counts as unopenable when its `activation_distance` is `<= 0`, or when it
carries a `gate` that is effectively locked and does not list the robot (robot
name or sim path, as in the gate's `authorized`) and it is not open or opening
at the moment (a plate, regime or authorized pedestrian may hold it open). It
is freed again as soon as that changes. With several robots in one env, one
authorized robot frees the door for all. Benchmarks keep the default (`false`)
unless the suite is about mechanism-aware planning.

### Pedestrians and world mechanisms

Everything in this section needs the humansim backend (`human:=arena`); other
backends ignore the extra keys and walk the plain waypoint list.

**Waypoints.** A `humans:` waypoint is a pose list (`[x, y]`, `[x, y, z]`), a
zone/door/elevator name, or a dict with exactly one of `zone:`, `elevator:` or
flat `x`/`y`[/`z`], plus optional `level:` (the coordinates are then
level-local) and `dwell:` (seconds to stand there before walking on). `{elevator: <name>}` is a ride: the ped walks
to the cabin centre, waits inside until the elevator teleports it, then
continues with the next waypoint. When two consecutive waypoints (or the
ped's start and its first waypoint) lie on different levels, the boarding
cabins of the shortest elevator route are inserted as rides automatically.
A ped waiting at an open cabin lets those getting out leave first, for up to
`humansim.zone_let_out_timeout` seconds (default 10, `0` boards at once).
For a multi-destination elevator the boarding ped writes the elevator's
`target` (the destination toward its next waypoint) when it reaches the
cabin centre. A ped holding for a ride never sees the next level's waypoint,
so it stays in the cabin however long the elevator takes. `waypoint_mode:
reverse` becomes an explicit round trip with rides both ways, `repeat` also
routes the way back to the first waypoint, `random` with dwell or rides walks
the list in order.

```yaml
humans:
- name: commuter
  model: arenian
  pose: [2.0, 3.0, 0.0]
  agent: {agent_type: adult}
  spawn_delay: 4.0                          # appears 4 s into the episode
  waypoint_mode: once
  waypoints:
  - {x: 6.0, y: 3.0, dwell: 5.0}            # wait at the board for 5 s
  - {elevator: elev_ground}                 # same-level pair: explicit ride
  - {x: 4.0, y: 4.0, level: "2"}            # other level: rides inserted
```

`spawn_delay: <s>` (default 0) keeps the ped out of the simulation (engine,
roster, physics sim) until that many sim seconds after the episode's spawn.

**Mechanism state as stimuli.** Every discrete or predicate change of a door,
elevator, signal, schedule, gate, pressure plate or sound reaches every ped as
the stimuli `<entity>.<field>==<old>` at 0 and `<entity>.<field>==<new>` at 1,
after each ped's sampled `reaction_time`. Entity names are authored (no env
prefix, no level suffix), the atom is the `conditions:` atom without spaces:
`elev_a.cabin_door==open`, `door_a.state==closed`, `gate_1.locked==true`,
`sig_1.state==red`. Continuous fields (progress, eta, occupants) are not sent.
An agent type reads one as a need pinned at 0 (it then holds 100 while the
state lasts) and waits on it with a transition. A ped spawned later starts
from the current values. `until:` on an autonomous step can name the same
atom, it fires when the state is entered.

```yaml
needs:
  "elev_a.cabin_door==open":
    initial: {mean: 0.0, clip_low: 0.0, clip_high: 0.0}
    decay_rate: {mean: 0.0, clip_low: 0.0, clip_high: 0.0}
sequences:
  wait:
    steps: {stand: {duration: {mean: 3600.0}}}
    then: wait
    transitions:
    - when: {"elev_a.cabin_door==open": {above: 50.0}}
      goto: board
```

To reuse one agent type for several elevators, give the `humans:` entry
`semantic_alias: {elevator: elev_a}`: that ped also receives
`elevator.<field>==<value>` for `elev_a`, and the agent type keys on
`"elevator.cabin_door==open"`. A ped an elevator teleports receives the
stimulus `teleported` at 1 (declare it with a fast `decay_rate` for a pulse).

**World objects.** Every door is a world object named after it with type
`door` at the doorway centre. Every elevator is a world object named after it
with type `elevator` at the cabin centre, plus `<name>/queue` of type
`elevator_queue` in the cabin doorway with a line formation that grows out
into the lobby (first slot 0.7 m outside the door, 0.8 m apart). A `go_to` or
`QUEUE_USE` step targets them by name or, by type, the nearest one.

**Shipped agent types** (`configs/agent_types/mechanisms/`, from a scenario
directory `../../../../configs/agent_types/mechanisms/<file>`):

- `elevator_rider.yaml`: `QUEUE_USE` on the nearest `elevator_queue`, board
  the nearest `elevator` when `elevator.cabin_door==open` (needs
  `semantic_alias: {elevator: <name>}`), back into the line if the door
  closes first (20 s patience), wait in the cabin, after `teleported` walk to
  the nearest marker of type `elevator_exit`. The whole line is
  released at once when the door opens.
- `door_holder.yaml`: walk to the nearest `door`, stand in the doorway for
  about 8 s (4 to 14), then walk to the nearest marker of type
  `door_holder_exit`.

```yaml
humans:
- name: rider_1
  model: arenian
  pose: [2.0, 3.0, 0.0]
  semantic_alias: {elevator: elev_a}
  agent: {agent_type: ../../../../configs/agent_types/mechanisms/elevator_rider.yaml}
objects:
- {name: exit_a2, model: Common/Object/painting_1, pose: [21.0, 5.5, 0.0], humansim: {type: elevator_exit}}
```

A marker is any `objects:` entry with a `humansim: {type: ...}` block (a wall
painting, as the `exit` of the fire-alarm scenario). Peds walk to its pose, so put it in
reach.

## 5. Add world-local assets (optional)

Assets placed under `worlds/$WORLD/assets/<domain>/<Type>/<name>/` are
resolved before shared-local (`$ARENA_ASSETS_DIR_LOCAL`), imported (`$ARENA_ASSETS_DIR_IMPORTED`) and network-fetched
assets. `<domain>` defaults to `Common` if you have no reason to scope the
asset further. Six asset kinds can be shipped this way:

| Type | Directory contents | Referenced as |
|---|---|---|
| `Object` | 3D model files (SDF and/or USD), optional `annotation.yaml` with a `bounding_box` | `model:` on an `objects:` entry |
| `Human` | 3D model files (SDF) | `model:` on a `humans:` entry |
| `Material` | `<name>.mdl` plus its texture files | a zone `floor`, a ceiling `material`, a wall or door `material` |
| `Wall` | `<name>.yaml`, a `WallDescription` (see [configs/walls/README.md](configs/walls/README.md)) | `kind:` on a `walls:` entry or a zone outline |
| `Sound` | `<name>.yaml` (a sound manifest, see [configs/sounds/README.md](configs/sounds/README.md)) plus its wav files | `asset_id:` on a `sounds:` entry |
| `Stairs` | `<name>.yaml`, a staircase kind (see [Staircase kinds](#staircase-kinds)) | `kind:` on a `stairs:` or `escalators:` entry |

To ship a custom wall style:

```bash
mkdir -p worlds/$WORLD/assets/Common/Wall/my_style
# create worlds/$WORLD/assets/Common/Wall/my_style/my_style.yaml
```

Then reference it in `world.yaml` walls as `kind: my_style`. The other five
kinds follow the same `assets/<domain>/<Type>/<name>/` layout. Objects, humans
and materials carry model or material files instead of a preset YAML, a sound
carries its `<name>.yaml` manifest beside the wav files, and a staircase kind
is its `<name>.yaml` alone.

A wall style can also carry a captured look: a 3D Gaussian splat that Isaac
renders in front of the wall. The
[splat wall example](configs/walls/README.md#example-splat-wall) shows the wall
kind, the [arena_assets README](../arena_assets/README.md) how to build the asset.
A captured object (`arena-assets splat --object`) is placed like any other
object, as an `objects:` entry. Isaac renders it and its sensors see it. It is
visual only, so robots drive through it, unless the bundle carries a collider.

Any object bundle can carry a collider mesh that `annotation.yaml` names under
`collider:`. It is baked into the bundle's USD and SDF models, where it takes
the place of the visual meshes as the collision. Without one, objects collide
with their visual meshes in Isaac and Gazebo alike. `arena-assets collider`
attaches one to any object bundle, see
[Colliders](../arena_assets/README.md#colliders).

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
from arena_simulation_setup.tree.World import WorldIdentifier

world = WorldIdentifier('my_world').resolve_sync().load()
for level_id, level in world.levels.items():
    print(level_id, f"{len(level.zones)} zone(s), {len(level.all_walls)} wall segments")
```

This raises `FileNotFoundError` if `world.yaml` is missing and `ValueError`
naming the YAML path if it is malformed. The MCP `worlds_scenario_validate`
checks a scenario text against its world, its zone references and the
articulated objects its `joints:`, conditions and timeline name included. The JSON Schema of the KG
comes from `arena_simulation_setup.kg.schema_for`.

List all scenarios:

```python
view = WorldIdentifier('my_world').resolve_sync()
for scenario_id in view.scenario.listall():
    print(scenario_id.name)
```

Load a scenario:

```python
scenario = view.scenario('default').resolve_sync()
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

### Navigation mesh

MeshNav plans on a triangle mesh of every walkable surface, built from the
world's geometry by `arena_simulation_setup.shared.navmesh`. The mesh samples
a regular grid (0.1 m by default) aligned to the simulator frame over each
level's floors and elevator cabins, at the floor height, and over every stair
and escalator half, at the height of its treads at rest (the belt for a belt
escalator, the landing plates over the pits). A grid point within a wall's
half thickness plus half a cell is left out where the wall stands between just
above the surface and 0.4 m over it. The walls are those of `all_walls`, cut
open at the doors, plus the sides of every elevator cabin but its door side,
so a cabin is entered only through its door. Doors stay open unless named in
`shut`, which closes them across their span, and static entities are not
subtracted. A grid cell becomes two triangles facing up when its four corners
lie within `climb` (0.25 m by default) of each other in height, so the floor
runs onto the first step and, in a stacked world, a bottom half's seam meets
its top half's seam on the level above as one surface. MeshNav builds the mesh
with the robot's `meshnav.max_step` as the climb, so for a robot whose step
limit lies below the riser height a staircase no longer joins its levels, and
with the doors that never open for the robot shut.

```python
from arena_simulation_setup.shared.navmesh import build_navmesh, navmesh_levels
from arena_simulation_setup.tree.World import WorldIdentifier

world = WorldIdentifier('stairs_demo').resolve_sync().load()
mesh = build_navmesh(navmesh_levels(world, world.stack_offsets()), climb=0.2)
mesh.write_ply('stairs_demo.ply')
print(len(mesh.vertices), len(mesh.faces), mesh.components().max() + 1)
```

The `worlds_navmesh` MCP tool does the same for any world name, stacking a
stacked world and spreading the levels of a spread one, and returns the PLY
path with its vertex, face and component counts and height range.
