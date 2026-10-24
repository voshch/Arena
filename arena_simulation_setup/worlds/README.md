# Worlds

Each world is a directory under `arena_simulation_setup/worlds/<name>/`. The
directory name is the canonical identifier consumed by `WorldIdentifier`.

## Directory layout

```
worlds/<name>/
|-- world.yaml          # the world, every level (format: arena-world/2)
|-- kg.yaml      # optional KG (format: arena-kg/1)
|-- map/
|   |-- <level_id>.png  # preview map per level, written by touch_world
|   `-- <level_id>.yaml
|-- scenarios/
|   `-- <scenario>/
|       |-- scenario.yaml     # Scenario (format: arena-scenario/2)
|       |-- scenario.json     # alternative; ScenarioView tries .yaml first
|       `-- <agent_type>.yaml # optional scenario-local arena_humansim agent types
`-- assets/
    `-- <domain>/
        `-- <asset_type>/
            `-- <name>/       # world-local asset; overrides network-fetched assets
```

A world without `format:` in its `world.yaml`, or with one
`<level_id>/world.yaml` per level, is format 1 and loads through the legacy
loader with a deprecation warning. `touch_world <name> --migrate` rewrites it
as format 2, see [AUTHORING.md](../AUTHORING.md#format-1-worlds).

## `world.yaml` schema

| scope | key | content |
| --- | --- | --- |
| world | `format` | `arena-world/2`, required |
| world | `defaults` | `floor` (material or `false`), `ceiling` (mapping or `false`), `wall` (`{kind}` or `{material}`), `door` (any door field but `name`, `span`, `semantics`), inherited where an item does not set the key |
| world | `schedules`, `signals` | standalone semantic entities, see [AUTHORING.md](../AUTHORING.md#m2-kinds-params-and-timelines) |
| world | `levels` | list of levels, each with a unique `id`, written out or linked as `{from: <world>[:<level id>], id, prefix}` (a world name or a `./` path under this world, see [AUTHORING.md](../AUTHORING.md#linked-levels)) |
| level | `zones` | `name`, `polygon`, `floor`, `ceiling`, `walls`, `semantics` |
| level | `walls` | `{path, closed, kind or material}` polylines |
| level | `doors` | see [Door fields](#door-fields) |
| level | `elevators` | see [Elevator fields](#elevator-fields) |
| level | `objects` | static obstacles, the scenario `objects:` schema |
| level | `humans` | pedestrians, the scenario `humans:` schema |
| level | `lights` | fixtures and ambient lights, see [AUTHORING.md](../AUTHORING.md#lights) |
| level | `sounds` | standalone sounds, see [AUTHORING.md](../AUTHORING.md#m2-kinds-params-and-timelines) |
| level | `microphones` | see [microphones](#microphones) |
| level | `pedestrian_walls` | `true` by default, `false` drops the occupancy-detected pedestrian collision walls (robot collision tracking reads the map regardless) |

```yaml
format: arena-world/2
levels:
- id: '0'
  zones:
  - name: central_hallway          # required, unique across the world
    polygon: [[8, 0], [12, 0], [12, 34.2], [8, 34.2]]   # open ring, interior is floor
    floor: Concrete_Smooth          # MaterialIdentifier: Name, Domain/Name, [Name, {modifiers}], false for none
    ceiling:                        # false for an open zone
      height: 3.0                   # top height (m), derived from the walls when absent
      material: Concrete_Smooth
      cast_shadows: false
      lights: {fixture: panel, spacing: 2.4, lumens: 3600}   # ceiling rig
    walls: outline                  # walls along the polygon, or {outline: true, open_to: [zone], kind|material}
    semantics: [{state: max_speed, value: 1.5}]
  walls:
  - {path: [[8, 0], [9.25, 0]], material: Plaster_Wall}
  - {path: [[0, 25.1], [8, 25.1]], kind: patient_room_walls}   # WallIdentifier name
  doors:
  - {name: main_hallway_entrance, span: [[9.25, 0], [10.75, 0]], kind: sliding, material: Aluminum_Anodized}
```

A wall path without `kind` generates plain wall segments with its
`material` (or the `wall` default). A path with `kind: <name>` is resolved via
`WallIdentifier(name)` at simulation time, see
[tree/Wall.py](../src/arena_simulation_setup/tree/Wall.py) and
[shared/walls.py](../src/arena_simulation_setup/shared/walls.py). Repeated
walls are merged and every door cuts the walls it lies on, the runtime reads
the result from `Level.all_walls`.

### `Door` fields

| key | default | meaning |
| --- | --- | --- |
| `name` | required | unique within the world |
| `span` | required | opening endpoints `[[x, y], [x, y]]` |
| `kind` | `sliding` | `sliding`, `sliding_top`, `teleport` implemented. `hinged` is accepted but falls back to `teleport` at runtime with a warning |
| `width` | `0.1` | metres |
| `height` | `2.0` | metres |
| `material` | door default | `MaterialIdentifier` |
| `activation_distance` | `3.0` | scalar or `[a, b]`, stored as a 2-tuple, only `max()` of the two drives the trigger radius |
| `transition_time` | `1.0` | seconds to fully open/close |
| `hold_time` | `2.0` | seconds held open after trigger clears |
| `semantics` | `[]` | attaches a scripted kind, see AUTHORING.md |

### `Elevator` fields

| key | default | meaning |
| --- | --- | --- |
| `name` | required | unique across the world |
| `position` | required | cabin center |
| `size` | `[2.0, 2.0, 2.0]` | cabin `[x, y, z]` |
| `door_side` | inferred | cabin face the door opens on (`+x`, `-x`, `+y`, `-y`), inferred from the nearest wall (<= 1 m) at load when absent, `+x` without one |
| `material` | elevator default | `MaterialIdentifier` |
| `destinations` | `[]` | names of the cabins a ride goes to, on any level. The runtime rides to the first |
| `activation_distance` | `3.0` | metres, the reach of an outside call while the cabin is empty; a cabin with a rider aboard reopens only for a body within 0.6 m of its door |
| `transition_time` | `1.0` | seconds |
| `hold_time` | `2.0` | seconds |
| `travel_time` | `3.0` | seconds between levels |
| `accept_outside_calls` | `true` | whether hallway buttons can summon the cabin |
| `recall_on` | `null` | regime name. While asserted, refuses calls and holds the cabin door open |
| `semantics` | `[]` | attaches a scripted kind, see AUTHORING.md |

### `microphones`

A level list of audio listeners for the acoustics pipeline:

```yaml
microphones:
- zone: lobby       # required, must reference a zone in this level
  placement: ceiling # required, free-form. "ceiling" is validated against the zone's ceiling
  position: [4.0, 2.0, 2.4]
  frame: map          # default map. with map, position must fall inside the zone polygon
  index: 1             # default 1, disambiguates duplicates, must be >= 1
```

`(zone, placement, index)` forms the listener id
`microphone:zone:<zone>:<placement>:<index>` and must be unique across the
whole world. See `WorldMicrophone` in
[tree/World/model.py](../src/arena_simulation_setup/tree/World/model.py).

### `kg.yaml`

The optional KG: zone types, descriptions and purposes, which zone holds
which thing, which door joins which zones, relations and intent constraints,
see [AUTHORING.md](../AUTHORING.md#kg). A world that links levels
also reads the `kg.yaml` of every world it links from.

## Semantic annotations

Door and elevator state is intrinsic: every spawned door and elevator
publishes its full state/predicate vocabulary (the kind's
`DISCRETE`/`CONTINUOUS`/`PREDICATES` in
[arena_runtime `_semantics.py`](../../arena_runtime/arena_runtime/arena_runtime/sim/_semantics.py))
with no annotation required. An `Elevator`'s fire-recall regime is a
first-class `recall_on: <regime>` field, not a semantics annotation:

```yaml
elevators:
- name: 1_elevator
  position: {x: 5.0, y: 0.0, z: 0.0}
  destinations: [2_elevator]
  recall_on: alarm
```

An optional `semantics:` list can still be added to any `doors:` entry,
`elevators:` entry, or to a zone itself. On a door
or elevator entry it attaches a *scripted* kind (`gate`, `pressure_plate`),
not door/elevator state. Each item is either a preset or a primitive, parsed
into a `SemanticCfg`
([shared/semantics.py](../src/arena_simulation_setup/shared/semantics.py)):

```yaml
doors:
- name: north_fire_door
  span: [[4.0, 1.55], [4.0, 2.45]]
  semantics:
  - {preset: gate, params: {authorized: [], unlock_on: alarm}}
zones:
- name: lobby
  polygon: [...]
  semantics:
  - state: max_speed
    value: 1.5
  - predicate: quiet
    value: true
  - predicate: restricted
    value: false
```

A preset (e.g. `{preset: gate}`) expands to the fixed set of states and
predicates that kind's runtime handler tracks. Any other keys on the preset
item (e.g. `distance`) apply as overrides to every expanded primitive, except
`value`, which is rejected on a multi-primitive preset: a `params:` key named
after one of the expanded primitives becomes that primitive's `value` instead.
Expansion happens at parse time, the list keeps the items as authored, so a
written world keeps its presets. Annotating a door/elevator entry with a field from that mechanism's own
published vocabulary (e.g. `{state: progress}` on a door) is rejected at
world-load time: remove the annotation, the field publishes on its own.

A primitive is `{state: <name>, ...}` or `{predicate: <name>, ...}`, with
optional keys:

| key | default | meaning |
| --- | --- | --- |
| `value` | `null` | literal for zone annotations; omitted/`null` means runtime-derived for doors/elevators |
| `binding` | `kinematic` | `joint` is reserved and raises a parse error in v1 |
| `trigger` | `proximity` | `contact` is reserved and raises a parse error in v1 |
| `distance` | `-1.0` | proximity radius in metres; `< 0` inherits the entity's `activation_distance` |

Zone semantics have a seed vocabulary (`max_speed` continuous state,
`quiet`/`restricted` predicates) that is inert in v1: parsed and published,
with no runtime behavior. Author-defined state/predicate names beyond the
seed are accepted verbatim.

Every item omits an empty `semantics:` list when written.

Presets with no door/elevator geometry of their own (`signal`, `schedule`,
`sound`) instead attach to a standalone `signals:`/`schedules:` entry of the
world or a `sounds:` entry of a level. `occupancy_cap`
attaches directly to a `zones:` entry.

### Annotated reference worlds

`hospital_1` and `reception` carry zone `semantics:` annotations and are the
reference for authoring new ones: throughput corridors/lobbies get
`max_speed`, wards/waiting/rest rooms get `quiet`, staff-only or hazard rooms
get `restricted`. Their doors carry no annotation: door state publishes on
its own.

| World | Zones |
| --- | --- |
| `hospital_1` | `max_speed`: `central_hallway`, `reception`, `sub_hallway`. `quiet`: `patient_ward`, `nurse_resting_room`, `waiting_area`, `doctors_resting_room`. `restricted`: `exam_room_1`, `pharmacy`, `operating_room`, `laboratory` |
| `reception` | `restricted`: `destination` (behind the counter). `quiet`: `entrance` (waiting chairs) |

## `map/` directory

A preview `map/<level_id>.png` + `map/<level_id>.yaml` per level can be
rendered with `touch_world` (see [scripts/touch_world](../scripts/touch_world))
for inspection, and `debug.map_source:=disk` loads them. They are not required
at runtime: the runtime renders its own map from `world.yaml` in-process, so
editing `world.yaml` is enough. Use `touch_world --all` to also rewrite the
canonical `world.yaml`.

`map.world.yaml` is a Gazebo world file and is not produced by `touch_world`,
it is hand-authored or produced by the `generate_world` pipeline.

## `scenarios/<name>/` schema

Resolved by `World.scenario` property which returns a per-world
`ScenarioIdentifier` class
([tree/World/view.py](../src/arena_simulation_setup/tree/World/view.py)).
`ScenarioView.load()` tries `scenario.yaml` first, then `scenario.json`.
A scenario `generate_scenario` wrote carries its `GENERATED_MARKER` header, see
[generative/scenario](../src/arena_simulation_setup/generative/scenario/README.md).

```yaml
format: arena-scenario/2
objects:
  - name: desk_1
    model: office_desk             # ObjectIdentifier
    pose: [x, y, yaw]
humans:
  - name: agent_1
    model: female_adult_medical_01 # PedestrianIdentifier
    agent: {agent_type: adult, desired_velocity: 1.4}
    pose: [4.0, 2.0, 0.0]          # yaw in radians
    waypoints:
    - [4.0, 2.0, 0.0]
robots:
  - start: [x, y, yaw]           # required. [x,y], [x,y,yaw], {x,y,yaw} or a zone name
    level: ""                    # optional, multi-level worlds
    goal_level: ""               # optional, level for goto phases
    phases:                      # ordered. each entry is one of:
      - {goto: [x, y, yaw]}      #   navigate to pose (same formats as start)
      - {goto: kitchen}          #   navigate into a zone, door, elevator or to a pedestrian by name
      - {gesture: wave}          #   arm gesture, "random" allowed, optional instance: <mount>
      - {reach: stow}            #   arm reach: named target, "random", or a pose (optional frame:)
      - {until: "door_1.open == true"}  # hold where the robot is until the atom holds
      - {loop: [{goto: [x, y, yaw]}, {goto: [x, y, yaw]}], until: "robot sees caller within 12 and caller plays beckon for 0.5"}
    conditions: []               # optional, clauses judged over this robot's whole phase list
```

Every phase also takes `until: "<clause>[ or <clause>...]"` with a clause being
`<atom>[ and <atom>...][ for <seconds>]` (` or ` binds loosest, a ` for ` dwell
belongs to its clause, ` and ` binds tightest; the phase completes once any
clause has held, every atom together, for its dwell, after arrival for a goto),
`hold_time: <s>` (park time at a goto goal), `conditions:` (clauses judged
over this phase only), `text:` (instruction for language-conditioned planners)
and `on_failure: continue | stop_task | abort_episode`. A `loop:` entry repeats
its phases in order; its `until` is judged on every tick, drops the running
phase the moment it holds and continues with the entry after the loop (without
`until` the loop runs until the episode ends). A loop takes only `until` and
`text`.
A pose given as a bare string samples a point inside that zone with identity
orientation, and a named `goto` is judged met by zone membership rather than by
distance to the sampled point. Which robot model and adapter runs is not part of
the scenario, see
[arena_robots setup](../../arena_robots/arena_robots/config/setup/README.md).
Schema: [tree/World/Scenario.py](../src/arena_simulation_setup/tree/World/Scenario.py).

A scenario without `format:` is format 1 (`static:`/`dynamic:`, the older
`obstacles:` wrapper, `goal:`, `start_floor`/`goal_floor`) and loads through
the legacy loader with a deprecation warning.

The `agent:` block configures the arena_humansim backend. `agent_type` is
either a built-in type (`adult`, `elder`) or a scenario-local agent-type YAML
file referenced by relative path (`agent_type: ./doctor.yaml`) for scripted
behavior (sequences, interactions, needs), see
`humansim/arena_humansim/config/agent_types/README.md` and the
`hospital_1/scenarios/showcase` scenario for a worked example.

`objects` (`Obstacle`) also takes `light: {lit, light_on, dim}` for the lights its
object annotation carries (see [AUTHORING.md](../AUTHORING.md#lights)), `scale: [x, y, z]` (default identity),
`level` (multi-level worlds) and a `humansim:` block (`type`, `capacity`,
`interaction_radius`, `formation`, `satisfies`, `seats`) that becomes its
humansim `WorldObject` info. `humans` (`DynamicObstacle`) also takes
`velocity` (m/s, default `1.0`), `waypoint_mode` (`repeat`, `reverse`, `once`,
`random`), `level`, the humansim `agent:` block and a `hunav:` block for the
HuNav backend. A human's `track: ./<file>.csv` names a recorded track beside the
scenario ([shared/track.py](../src/arena_simulation_setup/shared/track.py)), which `human:=replay` plays in
place of simulating the pedestrian, and `prespawn: true` stands it on its first pose until the track begins.
Its `stays:` list the timed legs of each continuous stay of that track (`start`, entry `pose` and `velocity`,
`legs: [[x, y, speed, dwell], ...]`), which the scenario importers write for a path follower.
Any other key fails the load. A waypoint is `[x, y[, z]]`,
`{x, y[, z][, level]}` or a zone name.

`sounds` is a list of `Sound` entries, same schema as a world level's
`sounds:` but episode-scoped (attached at reset, gone at the next one).

`lights` is a list of local `Light` fixtures, same schema as a world level's
`lights:` but episode-scoped, with `attach` naming one of the scenario's
own `objects`. `dome` and `sun` stay in the world.

Every placed entry (object, human, waypoint, sound, light, robot `start`)
sits on one floor by the same rule. A zone name as its `pose`, `position` or
waypoint samples the zone wherever it lies, whatever the entry's `level`.
Otherwise its point is local to its own `level`, else to the level of what
holds it (a waypoint's human, an attached sound's or light's object), else to
the sole level the env runs. Entries a world level lists sit on that level
and take no `level`.

`timeline` is a list of scripted `set` and `stimulus` actions fired on
`at`/`every`/`when` triggers. See [AUTHORING.md, Scenario timelines](../AUTHORING.md#scenario-timelines).

`regions` is a mapping of name to source/sink spawner consumed by the
arena_humansim backend:

```yaml
regions:
  entrance_source:
    type: source                  # source | sink
    polygon:                      # CCW corners [x, y]
      - {x: 9.25, y: 0.2}
      - {x: 10.75, y: 0.2}
      - {x: 10.75, y: 1.2}
      - {x: 9.25, y: 1.2}
    config:
      rate_profile:                # source only: list of {t, rate} keyframes
        - {t: 0.0, rate: 1.5}
      max_concurrent: 5             # source only, default -1 (unbounded)
      max_total: 50                 # source only, default -1 (unbounded)
      agent:                        # source only, spawn template
        agent_type: adult             # built-in name or ./relative.yaml
        desired_velocity: {min: 1.0, max: 1.5}  # or a scalar
        agent_radius: 0.35
        behavior_tree: default
        sink_affinity:
          - {sink: reception_sink, weight: 1.0}

  reception_sink:
    type: sink
    ref: lobby                    # instead of polygon:, takes the named zone's polygon
    config:
      absorption_radius: 0.5        # sink only, default 0.5
      capacity: -1                  # sink only, default -1 (unbounded)
```

An unrecognized `type` is rejected when the region is spawned.

`conditions` is a list of episode success/failure clauses:

```yaml
conditions:
- {op: eventually, p: "robot in pharmacy", text: "Reach the pharmacy."}
- {op: never, p: "robot in operating_room", text: "Stay out of the operating room."}
- {op: before, p: "robot in central_hallway", q: "robot in pharmacy", text: "Take the hallway before the pharmacy."}
- {op: never_during, p: "robot in exam_room_1", q: "exam_room_1_door/0.open == false", text: "Never be in the exam room while its door is closed."}
```

| key | meaning |
| --- | --- |
| `op` | `always`, `never`, `eventually`, `before`, or `never_during` |
| `p` | first atom, required |
| `q` | second atom, required for `before`/`never_during`, rejected (parse error) with any other `op` |
| `text` | human-readable description, optional |

An atom is one of these bare forms (no `env_N/` prefix): `<entity>.<field> ==
<value>` (an entity field test), `<subject> in <zone>` (zone membership),
`<subject> within <r> of <subject>` (planar distance), `<subject> sees <subject>
[within <r>]` (planar distance at most `r`, default 12, and no wall cell of the
world's rendered wall grid on the segment between them, doors open), `<ped>
plays <clip>` (the pedestrian publishes that gesture clip) and `not <atom>`. A
subject is `robot` (the robot being judged), another robot's name or a
pedestrian's name, and a `within` subject may also be a point `x,y` (no
spaces).
Every robot is also an entity of kind `robot` with the fields `phase`, `met`,
`failed`, `dropped` and `violated`, so `robot_1.phase == 2` is a valid atom.
A world entity (zone, door, elevator, schedule, sound) is named
`<name>/<level>`, or `<name>` alone when no other level has one.
Clauses take an optional `on_failure` (`continue` by default, `stop_task`,
`abort_episode`). Conditions are judged online at the node tick and replayed
offline by arena_evaluation with the same code.
Schema: [shared/conditions.py](../src/arena_simulation_setup/shared/conditions.py).

## `assets/` directory

World-local assets take precedence over network-fetched assets because
`DynamicPaths.WORLD / 'assets'` is registered before `NetResolver` for every
domain asset type, followed by the `assets/` of every world a level links to
(`DynamicPaths.LINKED`, see `DynamicPaths.as_resolvers` in
[tree/__init__.py](../src/arena_simulation_setup/tree/__init__.py)).

Layout mirrors the network structure: `assets/<domain>/<asset_type>/<name>/`.
A world that ships a custom `Wall` preset places it at
`assets/Common/Wall/<name>/<name>.yaml`.

## How `WorldIdentifier` resolves a name

Resolvers run in order, first hit wins
([tree/World/view.py](../src/arena_simulation_setup/tree/World/view.py), bottom of file):

1. Each `ARENA_WORLD_PATH` root (colon-separated env var), if set.
2. The writable local root `$ARENA_ASSETS_DIR_LOCAL/worlds` (default
   `$ARENA_ASSETS_DIR/local/worlds`), a `LocalResolver` that reads the variable
   again at every use. It is also the only write target
   (`resolve_write_path()` / `resolve_write_sync()`), so every saved world lands
   here: the world generator node, `save_generated`, `WorldView.save` on a
   `resolve_write_sync()` view and the `worlds` MCP tools. A world written here
   under the name of a shipped world shadows the shipped one.
3. `ASS_DIR / 'worlds'`, the worlds shipped in this package, existence-checked.
4. One `NetResolver` per `WORLD_BUCKETS` provider (default
   `arena-worlds-prod-public`), fetching into `$ARENA_ASSETS_DIR/<bucket>/<name>/`.
   A world keys off its bare name, so the worlds bucket is itself the namespace.

Worlds not shipped in this package are hosted in the public bucket(s) named
by `WORLD_BUCKETS` and pulled on demand. `arena asset find world <name>`
prints every resolver's verdict for a name (which one hit, which were
shadowed). `arena asset pull world <name>` fetches one into the local cache, and
`arena asset expire world <name>` makes the next use download it again.
`arena asset push world <name>` publishes a locally-authored world after a
preflight that checks every asset it references would resolve for someone
else.

`arena asset import world houseexpo <plan id>` writes one HouseExpo floor plan
as `houseexpo_<plan id>` into the imported worlds root
(`$ARENA_ASSETS_DIR_IMPORTED/worlds`, else `$ARENA_ASSETS_DIR/_imported/worlds`),
which `WorldIdentifier` reads after the `ARENA_WORLD_PATH` roots and before the
shipped worlds. The plan's vertex ring is the free space and its outline the
walls, so interior walls are the slits of the ring, doors are its gaps and
there are no door entries. Zones are the upstream room boxes clipped to the
floor, named `<room type>_<n>` after the n-th box of that category
(`living_room_0`): a smaller box owns what it shares with a larger one, a box
listed under several categories is one room named after the first of them by
name with all of them in `description`, a piece narrower than 0.3 m joins the
zone beside it, and floor under no box becomes `area_<n>`. All 35,126 plans
come in one 25 MB archive, downloaded once into
`$ARENA_ASSETS_DIR/.import/houseexpo` and indexed there, and a bare
`arena asset import world houseexpo` lists the plan ids. The repository is MIT
licensed and its plans are built on SUNCG, so the registry lists the source
with license verdict `unclear`.

`arena asset import world procthor <split>/<index>` (such as `val/0`) writes one
ProcTHOR-10K house as `procthor_<split>_<index>` into the same imported worlds
root, with zones named by room type, the walls cut around every door hole and
the furniture imported through the `ai2thor` object source and placed as static
entities. An openable door becomes a `hinged` door, as it swings in AI2-THOR, a
doorframe without a door leaf stays an opening, and a door that does not open
stays wall.

`WorldIdentifier('hospital_1').resolve()` returns a `WorldView`
wrapping whichever of the above resolved first. Call `view.load()` to parse
`world.yaml` (and `kg.yaml`) into a `World`.

## Shipped worlds

| Name | Notes |
|---|---|
| `airport` | Airport terminal layout |
| `hospital_1` | Multi-zone hospital with patient rooms, reception, hallways |
| `hospital_2` | Hospital variant |
| `map_empty` | Single open zone |
| `office_1` | Office floor plan |
| `reception` | Reception area |
