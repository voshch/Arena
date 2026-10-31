# task_generator human simulator

`BaseHumanSimulator` manages the pedestrian lifecycle (spawn, move, remove)
and the per-episode obstacle bookkeeping layer. Implementations are registered
in `HumanSimulatorRegistry`. See also [sim interface](../../../../arena_runtime/arena_runtime/arena_runtime/sim/README.md) for the
physics-simulator counterpart (`BaseSim`).

## `BaseHumanSimulator`

[`__init__.py:19`](__init__.py#L19)

```python
class BaseHumanSimulator(NodeInterface, abc.ABC):
```

Holds a `KnownObstacles` table that tracks every spawned obstacle by name,
its `spawned` flag, and its `ObstacleLayer`. On `__init__` it subscribes to
`<namespace>/arena_peds`, caches ped positions in `_ped_positions_xy`, and
calls `self._simulator.attach_human_simulator(self)` so the mechanism shim
(`MechanismITF`, see [sim interface](../../../../arena_runtime/arena_runtime/arena_runtime/sim/README.md))
can read ground-truth ped positions and dispatch ped teleports through this
class. Public methods delegate to both the physics `BaseSim` and the
human-sim `_*_impl` methods:

| Public method | Purpose |
| --- | --- |
| `spawn_obstacles(obstacles, layer)` | spawn or move static obstacles, layer defaults to `INUSE` |
| `spawn_dynamic_obstacles(obstacles)` | spawn or move dynamic obstacles (`INUSE`) |
| `spawn_world(walls, doors, collision_walls=())` | spawn world geometry in both sim and human-sim layers. `collision_walls` register in the human-sim layer only (avoidance), never spawned visually |
| `spawn_elevators(elevators)` | hand the realized cabins of the world to the human-sim layer through `_set_elevators_impl`, replacing those of the world before. The environment manager calls it before the physics sim's `spawn_elevators`, and `remove_obstacles(purge >= WORLD)` clears them |
| `elevator_doors_changed(doors)` | cabin name to door open, called by the mechanism shim when a cabin door turns open or stops being open, forwarded to `_set_elevator_doors_impl` |
| `spawn_stairs(stairs)` | sync, remember the realized stair halves of the world with every belt running. The environment manager calls it before `spawn_elevators`, which sends them along, and `remove_obstacles(purge >= WORLD)` clears them |
| `stairs_changed(running)` | realized stair half name to belt running, called by the runtime when an escalator's `running` semantic changes. Kept per half and sent again with every zone through `_set_elevators_impl` |
| `unuse_obstacles()` | call `_remove_obstacles_impl`, then flip all `INUSE` layers to `UNUSED` |
| `remove_obstacles(purge)` | remove all obstacles at or below `purge` layer from both layers. `WORLD` survives unless `purge >= WORLD` |
| `spawn_robot(robots)` | spawn in physics sim, start tracking the robot, then call `_spawn_robot_impl` |
| `remove_robot(robots)` | remove from physics sim, stop tracking, then call `_remove_robot_impl` |
| `move_robot(robots)` | move in physics sim, set the tracked pose at rest, then call `_move_robot_impl` |
| `tracked_robots()` | every spawned robot as a `TrackedRobot`: map-frame pose from the `map` to base-frame TF and the velocity between its last two TF stamps |
| `notify_stimulus(agent, stimulus, intensity)` | stimulus seam, returns False by default, fed edge-triggered by the acoustics backend's `PedestrianHearing` (arena: `continuous_heard_sounds` for `agent:<id>` listeners) and by scenario timeline `stimulus` actions. `agent` is an engine id or a spawned agent's name or `sim_path` |
| `check_stimuli(scenario, humans, timeline)` | raise `ValueError` at scenario load for any timeline stimulus on a backend whose `TAKES_STIMULI` is False (only arena sets it), and for one whose target's agent type does not act on it, judged by `stimuli_taken(obstacle)` (None by default, arena: the type's needs and autonomous step `until` events) |
| `notify_semantic_change(entity, field, value)` | sync, called by the node for every discrete/predicate `SemanticChange` (authored entity name). Delivered in order, in the background, as the broadcast stimuli `<entity>.<field>==<old>` at 0 and `<entity>.<field>==<new>` at 1, so it is inert on backends without stimuli |
| `set_mechanism_walls(key, segments)` | sync, no-op by default. Named wall set for pedestrian navigation (map frame), `None` removes it. Called by the mechanism runtime with `elevator:<name>` (cabin back and side walls) and `door:<name>` (doorway while the door cannot open for peds) |
| `authored_name(realized)` | env prefix and trailing `/<level_id>` stripped |

### `HumanSimulator` Protocol surface

`BaseHumanSimulator` satisfies the Protocol the mechanism shim reads from.
Inherited defaults work for all current subclasses, override only for
specialized teleport semantics (e.g. resetting an internal agent list).

| Method | Signature |
| --- | --- |
| `pedestrian_discs()` | `() -> Iterable[tuple[str, tuple[float, float], float]]` (sync, reads `_ped_positions_xy`, radius is `PED_RADIUS`) |
| `pedestrian_clips()` | `() -> Mapping[str, frozenset[str]]` (sync, the gesture clips each ped is playing, from the same `arena_peds` message) |
| `pedestrian_teleport(destinations)` | `(Mapping[str, tuple[float, float]]) -> bool` (async, dispatches via `relay_pedestrian_update`, `ArenaHumanSimulator` teleports in the engine instead, then sends each moved ped its next itinerary leg and the `teleported` stimulus) |
| `elevator_doors_changed(doors)` | `(Mapping[str, bool]) -> None` (async) |
| `stairs_changed(running)` | `(Mapping[str, bool]) -> None` (async) |
| `configure(params)` | `(Sequence[rcl_interfaces.msg.Parameter]) -> list[Parameter]` (async). It runs at every episode reset with `QueueEpisode.human_params`. It strips the prefix from the names under `PARAM_NAMESPACE`, sorts them, and hands them to `_configure_impl`, also when there are none. It returns the accepted ones. |

### Abstract `_impl` methods

Every subclass must implement:

| Method | Purpose |
| --- | --- |
| `_spawn_obstacles_impl(obstacles)` | human-sim side: register or prepare static obstacles |
| `_spawn_dynamic_obstacles_impl(obstacles)` | human-sim side: register pedestrian agents |
| `_remove_obstacles_impl(obstacles)` | human-sim side: remove static obstacles, keyed the way the backend spawned them (arena_humansim by `sim_path`, hunav and dummy by `name`) |
| `_spawn_walls_impl(walls)` | human-sim side: ingest wall geometry |
| `_spawn_doors_impl(doors)` | human-sim side: ingest door geometry |
| `_spawn_robot_impl(robots)` | human-sim side: register robots |
| `_remove_robot_impl(robots)` | human-sim side: deregister robots |
| `_move_robot_impl(robots)` | human-sim side: update robot positions |

`_set_elevators_impl(elevators)` and `_set_elevator_doors_impl(doors)` default to no-ops, a backend whose
pedestrians ride elevators overrides them.

## GaitGenerator as articulation ground truth

`BaseHumanSimulator.publish_arena_peds` is the single point where skeletal
joint angles are committed to the `arena_peds` bus.  For each `Pedestrian` in
the outgoing message it checks `ped.joint_state.name`:

- **non-empty**: upstream backend supplied its own joint state, published unchanged (override path: an upstream producer that already computes joint angles).
- **empty**: `publish_arena_peds` calls `AnimationManager.compute` (`GaitGenerator` for idle/walk/run, clip playback and overlays for everything else) and fills the field with bare semantic joint names (40 DOF, no body suffix).

The filled field feeds the ROS4HRI skeleton in rviz through `hri_producer` and the Isaac and Gazebo pedestrian rigs.

A model whose bundle ships a `rig.yaml` with `drivers:` (wheelchair bundles: `l_wheel`, `r_wheel`) gets those joints
set on top, whoever filled the field: [`drivers.py`](drivers.py) rolls each wheel over the distance its contact
point covered since the last message, see the Driven joints section of [`JOINTS.md`](JOINTS.md).

The same message goes out again on `<namespace>/sim_peds`, which the Gazebo pedestrian plugin reads: unchanged in a spread
world, every ped moved into the simulator frame by `Realizer.placement()` when the levels are stacked (`world.levels`).

## Resting on furniture

`Pedestrian.pose` is the ped's body frame: the origin at the soles under the pelvis, z up the body. Every renderer
moves its character along that z axis until its lowest sole point is at body height 0, so a walker stands on the floor
and a sit clip, which folds the legs under hips kept at standing height, lowers the body as the legs fold.

A backend reports the seat or bed a ped rests on through `_support()` (humansim: the seat an agent is parked on, with its
surface height), and `publish_arena_peds` places it with [`support.py`](support.py) from its joint angles and the
character's measurements (`pelvis_clearance` and `legs` in the asset annotation, measured by the arena-assets human
build, else measured off its skin, typical values with a warning without either). The seated clearance depends on the hip
flexion: `pelvis_clearance.seated` tabulates the pelvis joint's height above the lowest buttock vertex per flexion, the
buttocks within 12 cm below the hip joints skinned through it, and `sit_at` interpolates it. A sitter whose planted
pelvis would end below its seat's surface plus the seated clearance at its current hip flexion is lifted to it. A lying ped is tipped onto its back, the head away from its
yaw, with the pelvis over its seat at the mattress plus the lying clearance. A possessed ped is left as its stream sends
it.

A `sit` clip on the body slot waits until the ped is parked on its seat, then plays a sit-down planned per ped and seat
instead ([`plan_seat`](support.py)). The pelvis sits back against the seat's rear edge (`Support.back`, from the sit
slot's box) by the character's lying clearance, forward only as far as the backs of the calves need to clear the front
edge (`Support.front`) by `SHIN_GAP`, and at least `PELVIS_MARGIN` behind it. The feet are planted where the calves
end up vertical, the pelvis at the seat's surface plus the seated clearance at the flexion that height gives. The pelvis drops along the `squat` Animation clip's drop and slides back from over the feet onto the
seat, the legs solved for the planted feet with flat soles and the torso taking the squat's forward lean. The arms
rise with the fold to `SEATED_ARMS`, forearms forward, clear of the thighs and hips. Once down the
torso reclines to vertical over `RECLINE_S`, where the layer parks. When the slot ends the ped leans forward again over
`LEAN_S` and rises the way it sat down. A seat too high for the shins to reach the floor ends with horizontal thighs and
dangling shins, the pelvis held at the seat's surface and sliding from the seat's standing point onto the seat as the
hips fold. Without a `squat` clip the `sit` clip plays as before.

The body is placed every tick so the feet stay where they were planted. It blends from where the ped stood into its
resting pose when it is parked, over the sit-down's descent (`SeatPlan.down`, at least `SETTLE_S`), keeps its seat
until the rise is over after the engine releases it, and blends over `SETTLE_S` to wherever the engine put it. The
engine keeps a released agent where it stood before it was parked for its `SEAT_RISE_S` (1.5 s), so the rise ends where
it walks off from. A lying ped blends over `LIE_S` both ways. A character whose annotation lacks `pelvis_clearance` and `legs`
is measured off its skin (`meshes/<name>.dae`) with the arena-assets human build's code.

## Fitting a gait profile from a clip

The walk and run poses `GaitGenerator` emits come from the gait profiles in [`profiles/`](profiles/)
([`profile.py`](profile.py): a mean plus harmonics per joint, read at the gait phase). [`fit.py`](fit.py) turns a
walking clip into such a profile:

```
python3 -m task_generator.simulators.human.fit walk_slow.npy -o walk_slow.yaml --speed 0.8 --report walk_slow.json
```

The clip is a `.npy` object array of frames, each `{"angles": {<joint>: <rad>}}` with bare `JOINT_NAMES` (the
per-frame layout of an [Animation clip](#animation-clips)), holding at least two whole cycles of steady walking. The
fitter takes the cycle from the sagittal hips (`l_r_hip`, `r_r_hip`), sets phase 0 so the left hip lines up with
`walk_cmu_12_01`, and fits every joint over whole cycles. It writes one signal per limb pair when the right side is
the left half a cycle later (within 5 percent of the pair's amplitude), per-side signals with `symmetric: false`
otherwise, a `phase_warp.split` when the two half-cycles differ in length by more than 0.02, and `limits` for
joints the clip takes past `GaitGenerator.LIMITS`. The printed report lists the rms error per joint, the cycles
used, the measured cadence, the split and the worst joint.

| Option | Meaning |
| --- | --- |
| `-o PROFILE.yaml` | Output profile. Move it into `profiles/`, the library goes by file stem, then name it from an agent type as `pose: {walk: {base: <stem>}}`. |
| `--fps F` | Frame rate of the clip, default `20` (the rate `AnimationManager` plays the shipped clips at). |
| `--harmonics N` | Harmonics per joint, `1` to `8`, default `3`. |
| `--speed V` | Walking speed of the clip in m/s: the cadence law keeps the default base and takes `per_speed` through the measured point. Without it the law is fixed at the measured cadence. |
| `--report REPORT.json` | Also write the report as json. |
| `--name NAME` | `name:` field of the profile, default the output stem. |

## Gestures

`Pedestrian.gestures` (`arena_people_msgs/Gesture[]`: `slot` in `head|arm|arm_l|arm_r|body`, `at` world-frame
point for the aimed slots, `clip` name on `body`, `hand` on `arm`) are attention channels, not poses. Backends
only forward them (arena_humansim copies `AgentGestures` per owner one to one, the possession stream carries them
as-is), and `publish_arena_peds` hands them per ped to the [`GestureLayer`](gestures/__init__.py) as a
`GestureRequest` (a `Channel(slot, at, clip, hand)` per entry, ped pose, moving flag). The layer resolves each
aimed target into the ped frame once per clip, asks the slot's generator for frames (`head` -> `look`,
`arm*` -> `point`, `body` -> `clip`, the Animation asset named on the wire over the joints its
annotation lists, see [Animation clips](#animation-clips)), and plays them as independent `AnimationManager`
overlay slots (`head`, `arm`, `body`) over the locomotion base. Nothing is synthesized: no channel published =
that body part idle. See [gestures/README.md](gestures/README.md) for slots, hand resolution,
timing, and how to add a kind.

An empty list releases every slot, a missing entry releases that slot. Unknown slots warn once per ped and are
ignored. Clips come from the baked pointing table and are generated inline on the publishing tick, so their
timing is sim-deterministic under lockstep without any extra plumbing.

## Animation clips

Canned clips (`wave`, `hug`, the pointing template `point_at_template`, ...) are `Animation` assets
([Animation.py](../../../../arena_simulation_setup/src/arena_simulation_setup/tree/assets/Animation.py)), not
repo files. `AnimationManager` loads every clip it can list at startup through [`AssetClips`](clips.py), and
`AnimationManager.clip(name)` fetches one published after launch on first request. Resolution order: the active
world's `assets/`, `$ARENA_ASSETS_DIR_LOCAL` (= `arena_simulation_setup/assets/`), then the asset bucket (cached
under `$ARENA_ASSETS_DIR/<bucket>/`). Both local dirs gitignore `*/Animation/`.

```
Common/Animation/<name>/
  annotation.yaml   fps, loop, reverse, loop_from, release_from, joints (a JOINT_GROUPS name from clips.py or a list;
                    omitted = whole body), pair, note
  <name>.anim       the clip's arrays, settings over the annotation, and for point_at_template the baked pointing
                    table (pointing/table.py), format in the tree README
```

A looping clip's `loop_from` (s) makes the frames before it an intro that plays once, the loop wraps back to it.
`release_from` (s) cuts the frames from it off as the release: they play when the gesture ends, blended out of
wherever the loop is, instead of the default fade. A two-person clip (`arena_animation_retarget --pair`) carries
`pair: {role, partner, partner_offset: [dx, dy, dyaw], separation}`, the partner's stance seen from this role, which
humansim's clip sets (`ClipSet.pair`) place the pair with.

Adding a clip: write it with `write_clip(Path("arena_simulation_setup/assets/Common/Animation/<name>"), frames, meta)`
(frames are the players' `{"t", "angles", "root_xy_yaw", "animation_state"}` dicts), check it with
`arena asset find animation <name>`, then publish it with `arena asset push animation <name>` (needs `GCS_ACCESS_TOKEN`).
A clip from CMU mocap (the cgspeed BVH conversion, the `cmu_mb` rig of the human assets) comes from
[`bvh.py`](bvh.py): `python3 -m task_generator.simulators.human.bvh 22_14.bvh --skin <human>/meshes/<human>.dae -o
.../Animation/squat --from 0.72 --to 1.75` turns each bone's rotation into its bind frame, read off any human asset's
skin (they share the rig), and solves it for its wire joints through `bone_map.json`, the root held upright with its
tilt folded into the thighs and the lower back. Only the legs and the spine are converted by default, the rest stays at the
neutral stance, and the feet keep their soles flat. `--parallel` holds the hips' yaw and abduction at 0 for a squat with
parallel legs. The shipped `squat` is the first squat of CMU 22_14 (0.72 to 1.75 s) converted with `--parallel`.
A scenario's clips (`ScenarioView.agent_clips`) are resolved when the scenario task loads it: bucket clips are
fetched before the first agent plays one, and missing ones are named in one warning. Publishing a world
(`arena asset push world`) refuses clips that only resolve locally.
Tests that need clips carry `@pytest.mark.clips(...)` and skip when the clips cannot be resolved.

## Visualization topics

Pedestrian visualization flows through one data feed plus a few marker layers, all at
env level (`<env_ns>/`) and shared by every backend:

| Topic | Producer | QoS | Role |
| --- | --- | --- | --- |
| `arena_peds` | each backend (`publish_arena_peds`) | reliable, volatile | Pedestrian state feed (positions/velocities/joint_state). `joint_state` carries bare semantic joint names filled by `GaitGenerator` unless the backend overrides it (non-empty = upstream wins). |
| `humans/bodies/tracked`, `humans/persons/*`, `humans/bodies/<id>/joint_states` | `hri_producer` node | per REP-155 | **Canonical** ROS4HRI projection of `arena_peds`: id lists, per-person engagement, per-body joint states, per-body URDF on param `human_description_<id>`. |
| `humans/tf` | `hri_producer` node and its `robot_state_publisher` pool | reliable, volatile | TF `body_<id>` and the joint frames, kept off the env `/tf` and published only while subscribed. |
| `pedestrian_markers/extra` | base class (`publish_markers`) | best-effort, volatile | Backend-internal debug overlay (e.g. arena_humansim forwards its module markers and retargets `arena_humansim_viz_node` here with the env offset). Off by default. The arena_humansim adapter subscribes to the engine's `viz` only while this topic has a subscriber, so an unwatched env costs the engine no marker work. |
| `pedestrian_markers/static` | base class (`publish_static_markers`) | reliable, transient-local, depth 1 | Latched static scene as one combined topic. |
| `pedestrian_markers/static_*` | adapter | reliable, transient-local, depth 1 | Latched static scene split per bucket (`/static_walls`, `/static_objects`, ...). |

**Rendering contract.** The canonical human view is an animated articulated skeleton, produced by the
[`hri_producer`](../../../../utils/rviz_utils/rviz_utils/scripts/hri_producer.py) node, which subscribes
`arena_peds` and projects it into the ROS4HRI (REP-155) `humans/` namespace: id lists, per-person
engagement, and a per-body `robot_state_publisher` (pooled) driven from `humans/bodies/<id>/joint_states`
against the `human_description` URDF rig. The [`hri_rviz/Skeletons3D`](https://github.com/ros4hri/hri_rviz)
display renders one kinematic model per body.

`hri_producer` is a relay: it re-suffixes joint names from `arena_peds.joint_state` per body ID and
publishes them directly.  The producer's own `GaitGenerator` instance is a **fallback only** for peds whose
`joint_state` arrives empty (backends that do not fill joint_state on the bus).
`extra` is backend debug, disabled by default.

**Display kinds** (`arena_viz.DisplayKind`):
- `PEDESTRIANS`: the canonical `hri_rviz/Skeletons3D` skeleton display, keyed on the env `humans/`
  namespace. Note: the upstream display uses absolute `/humans` paths via libhri, so per-env namespacing
  is a known limitation.
- `MARKER_ARRAY`: generic MarkerArray passthrough, no namespace assumptions, used for `extra` and all
  `static*` layers.

The auto-rviz manifest ([`node.py` `_publish_viz_manifest`](../../node.py)) groups these into a
**Pedestrians** folder (skeleton display + `extra`, off) and a separate **Static** folder (`static`,
`static_walls`, `static_objects`). The `pedestrian_markers/` prefix on the debug/static layers is
historical, they are overlays, not pedestrian data.

Current adapters:

| Adapter | Static topics published |
| --- | --- |
| `arena_humansim` | `pedestrian_markers/static_walls`, `pedestrian_markers/static_objects` |
| `dummy` | `pedestrian_markers/static_walls`, `pedestrian_markers/static_objects` |

## PROMPT registration

`TM_Prompt` is not registered centrally. Each `BaseHumanSimulator` subclass
that supports LLM-driven obstacle generation registers its own `TM_Obstacles`
variant (including the system-prompt text and response parser) via
`_register_task_modes` at class-definition time. This co-locates the prompt
with the simulator that will animate the resulting agents.

## Registered implementations

`HumanSimulatorRegistry` ([`__init__.py`](__init__.py)) maps
`Constants.HumanSimulator` keys to async factory functions. Per-sim defaults
(resolved in `task_generator.launch.py`): `gazebo`/`isaac` sims default to
`human:=arena`, the `dummy` sim (and bare/test contexts) to
`human:=dummy`.

| Key | Class | File | Notes |
| --- | --- | --- | --- |
| `dummy` | `DummyHumanSimulator` | [`dummy.py`](dummy.py) | no-op stubs, used in test/offline contexts |
| `none` | `DummyHumanSimulator` | [`dummy.py`](dummy.py) | `human.launch.py` starts no node, backed by the same no-op stubs for registry lookups |
| `arena` | `ArenaHumanSimulator` | [`arena_humansim/arena_humansim.py`](arena_humansim/arena_humansim.py) | integrates with the arena_humansim pedestrian simulator (subsystem mode) |
| `replay` | `ReplayHumanSimulator` | [`replay.py`](replay.py) | plays the track each pedestrian entry names, `human.launch.py` starts no node |
| `hunav` | `HunavHumanSimulator` | [`hunav/hunav.py`](hunav/hunav.py) | integrates with HuNav, see [configs/hunav/README.md](../../../../arena_simulation_setup/configs/hunav/README.md) for the per-agent schema. `hunav_msgs` is imported lazily so the registry entry loads without it installed |

## arena_humansim agent types

`ArenaHumanSimulator` derives pedestrian parameters from the `agent:` block of
each `humans:` scenario entry (`ArenaHumanDynamicObstacle`,
[`arena_humansim/agent.py`](arena_humansim/agent.py)). `agent:` accepts
either a bare string or a dict:

```yaml
humans:
  - name: nurse_1
    ...
    agent: adult                 # shorthand for {agent_type: adult}
  - name: doctor_1
    ...
    agent: {agent_type: ./doctor.yaml, desired_velocity: 1.4, radius: 0.32}
  - name: source_ped
    ...
    agent: {agent_type: adult, desired_velocity: {min: 1.0, max: 1.5}}
```

| Key | Meaning |
| --- | --- |
| `agent_type` | Built-in arena_humansim type name (`adult`, `elder`, `robot`, shipped under `arena_humansim/config/agent_types/`) or a scenario-local YAML path resolved relative to the scenario (`./doctor.yaml`). Default `adult`. See [config/agent_types/README.md](../../../../humansim/arena_humansim/config/agent_types/README.md) for the file schema. |
| `desired_velocity` | Either a scalar (m/s), or `{min, max}`: a uniform range this instance's velocity is drawn from. Overrides (does not compose with) the sampled `AgentType.desired_velocity`, and a value above the sampled `max_velocity` raises it to match. Unset, the sampled value stays. |
| `radius` | Agent collision radius (m), overrides the sampled `AgentType.agent_radius`. Unset, the sampled value stays. |

`waypoint_mode` is a sibling key of `agent:` (not nested inside it), one of `repeat` (default), `reverse`, `once`, `random`: it controls how the entry's `waypoints:` list is replayed once exhausted.

A `regions:` source entry's `config.agent:` block uses a different, wider schema for continuously spawning pedestrians: `agent_type`, `desired_velocity: {min, max}`, `agent_radius` (note: `agent_radius`, not `radius`, here), `behavior_tree`, and `sink_affinity: [{sink, weight}]`. An unset `desired_velocity` or `agent_radius` keeps the agent type's sample. See `flow_agent_template` in [`arena_humansim/agent.py`](arena_humansim/agent.py).

A scripted entry renders with its `model:`. A source-spawned agent has none, so it takes a model from its agent type: when the type lists `assets:` tags, one of the human bundles on disk (world-local, shared-local or already cached) whose `annotation.yaml` carries every tag is drawn per agent from the episode seed. A type without `assets:`, or tags no bundle carries (warned once per type), keeps the default model.

### Static world objects

An `objects:` scenario or world entry that pedestrians can interact with (a bench, a
reception desk) is registered as a `WorldObject` by `_spawn_obstacles_impl` in
[`arena_humansim/arena_humansim.py`](arena_humansim/arena_humansim.py). The
keys below go in the entry's `humansim:` block (`Obstacle.humansim`):

| Key | Meaning |
| --- | --- |
| `type` | Object type string, matched against a behavior-tree step/action `target:` (falls back to the model's `annotation.yaml` name/desc if omitted). |
| `capacity` | Queue length at which a needs-driven agent looking for an object of this type passes this one over. Default the count of the entry's `seats`, else the most of the annotated sit slots that can be held together, at least `1`. |
| `satisfies` | `{need: amount}` applied to an agent that completes an interaction here, same shape as an agent-type step's `satisfies:`. |
| `interaction_radius` | Overrides the interaction kind's default approach radius for this object. |
| `formation: {type, params}` | Provider-side formation for agents interacting here (`type` one of `line`, `cluster`, `f_formation`, `dyad`, see [config/agent_types/README.md](../../../../humansim/arena_humansim/config/agent_types/README.md)). |

```yaml
objects:
  - name: reception_desk
    ...
    humansim:
      type: desk
      capacity: 3
      interaction_radius: 1.0
      formation: {type: line, params: {base_step: 0.8, front_offset: 0.6}}
```

## Possession

Every backend supports upstream override through the possession layer in
`BaseHumanSimulator`, with the pure logic in
[`possession.py`](possession.py).

`human/stream` (`Pedestrians`, env-level, reliable/volatile depth 1) carries
full pedestrian state, and each batch is the publisher's complete claim set:
a validated entry claims or renews its ped (deep copy stored), a possessed
name absent from a valid batch is released instantly, and silence past
`POSSESSION_TIMEOUT_S` (1 s) releases everything as the crash fallback,
evaluated lazily on sim time with no timers. Batches are dropped while the
reset gate is closed and when their header stamp predates the gate-open
stamp (ped names persist across episodes). Unknown names and non-bare joint
names drop per entry (suffixing stays `hri_producer`'s job), positions pass
through unclamped (`gait.LIMITS` is advisory and generator-side), `model_uri`
is ignored.

Enforcement is substitution at the two outbound funnels: `publish_arena_peds`
(the bus) and `relay_pedestrian_update` (the sim). Substitution is
copy-on-write: possessed entries swap to deep copies of the stored state,
everything else passes through untouched, and the caller's message is never
mutated (hunav hands a persistent container to both funnels). Substituted
entries get `gait_phase` re-stamped from the base's own `GaitGenerator`
phase table, engines never see the publisher's zero.

`human/move` (`srv/MovePedestrians`, env-level) is a teleport served by the
base, only `name` and `pose` are read. Known peds are moved in place (no
`forget()` round-trip, that would reset spawn bookkeeping) and relayed via
`pedestrian_move`, everything else returns `NOT_FOUND` per item. A move does
not claim possession.

The reset gate opens at the end of `spawn_dynamic_obstacles` (which also
indexes each ped under both its name and its sim_path, the bus name differs
per backend) and closes in `unuse_obstacles` and the `remove_obstacles`
dynamic purge, which clear the table without release hooks (those peds cease
to exist). Backends observe possession through synchronous no-op hooks
(`_on_stream_merged`, `_on_ped_released`, `_on_ped_moved`) and the
`possessed_peds()` accessor.

## Dummy adapter (`human:=dummy`)

`DummyHumanSimulator` ([`dummy.py`](dummy.py)) has no locomotion engine,
the possession stream is the sole source of motion. It seeds a roster cache
at spawn (sim_path names, IDLE poses) and republishes it through
`publish_arena_peds` at 30 Hz of sim time, letting the possession
substitution overlay driven peds on the way out. A paused sim repeats no
roster, only a merge or a move publishes then.

Its hooks keep the cache honest: a merge publishes the full roster, a
release adopts the last driven pose with an empty `joint_state` so the
synthesized gait eases the ped back to idle, a move adopts the teleport pose
and publishes. Nothing moves on its own: without a publisher peds hold
position with idle gait, and closing the GUI freezes them, it does not
despawn them. Reset closes the gate and clears the cache, respawn reseeds it
at the spawn poses and reopens the gate. Walls and static obstacles have no
engine to live in, so they go out as latched
`pedestrian_markers/static_walls` / `static_objects` MarkerArrays (ns
`dummy_walls` / `dummy_objects`), each led by a `DELETEALL` so removals
clear stale markers.

## Replay adapter (`human:=replay`)

`ReplayHumanSimulator` ([`replay.py`](replay.py)) is the dummy adapter plus recorded motion. A `dynamic:` entry
whose `track:` key names a track file ([`shared/track.py`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/track.py):
rows of `t,x,y,yaw`, seconds from episode start, meters and radians in the level frame, a `nan` row between
segments) is played from it on the sim clock at 30 Hz, counted from the reset that spawned it and shifted by the
env and level origin. The pedestrian is on `arena_peds` and relayed to the sim exactly while a segment covers the
episode time: it enters with its first sample, leaves in every gap and for good after the last sample, and each
edge calls the sim's `pedestrian_spawn` / `pedestrian_delete`. With `prespawn: true` beside `track:` it instead stands
at rest on its first pose from the episode start until the track begins. Entries without a track stay dummy puppets, and the
possession stream overlays both. Nothing reacts to the robot. The roster goes out on the sim clock, so a paused sim
publishes none.
`arena asset import scenario eth_ucy <eth|hotel|univ|zara01|zara02>` writes such a scenario from the ETH and UCY
sequences and, where it is not imported yet, the world it plays in, whose floor is the concave hull of the paths their
pedestrians walked, kept 0.5 m clear of every step and straightened into a plain polygon. Its tracks run at 10 Hz, with
gaps up to 1 s bridged. Each entry also carries `stays:`, one per continuous stay (split at gaps and at jumps over
0.5 m faster than 6 m/s, stays under 1 s dropped) with its `start`, entry `pose` and `velocity` (over the first 0.4 s)
and `legs: [[x, y, speed, dwell], ...]`, corners within 0.25 m of the stay in space and time with each stop as the dwell
of the leg before, for a path follower. zara01 and zara02 share one world, with zara02 moved 1.15 m
along y into zara01's frame, the placement SocNavBench gives the two. SocNavBench's eth and hotel copies are in meters,
its zara and univ copies are not: zara is stretched about 23% along x and squeezed about 9% along y, and univ also lacks
the camera's perspective. The importer takes both to meters, zara with the affine map that gives the parked car of the
Social-GAN and TrajNet++ zara calibration its 1.81 x 4.63 m, univ with the homography that gives the 2.95 x 3.00 m paving
rectangle of OpenTraj's univ ground-plane measurement its size.

`arena asset import scenario socnavbench <episode>` writes one of SocNavBench's curated episodes on these sequences as a
replay scenario that also carries the robot task: `robots:` holds the episode's start and its goal as one `goto` phase,
mapped back from SocNavBench's map frame into the frame of the eth_ucy worlds and taken to meters the same way, and `humans:` holds the pedestrians
SocNavBench plays in the episode (ids from the episode's first pedestrian on, up to the first one who enters later than
the time budget), on tracks cut to the budget and timed from that first pedestrian's first frame. Positions are the ones
the eth_ucy scenario plays. The world is `socnavbench_eth` or `socnavbench_zara`: the eth_ucy floor with the start and
goal of every episode of the scene added, each 0.5 m clear of the walls. The 17 episodes on eth, zara01 and zara02 are
importable. The 16 on hotel, univ and the double hotel are refused, because SocNavBench plays those sequences at 15 fps
although they were recorded at 25 fps and places the hotel as a mirror image. A scenario has no field for the time
budget: pass `task.episode.timeout:=<s>` when launching one by hand, or run the
[SocNavBench suite](../../../../arena_evaluation/arena_evaluation/configs/benchmark/README.md#socnavbench), whose stages carry it.

Dense crowds come from three more sources, their tracks resampled to 10 Hz with stays like the ETH and UCY ones.
`arena asset import scenario juelich <set>/<run>` writes a run of the Juelich pedestrian dynamics archive (BaSiGo 2013
corridor and 90 degree crossing, HERMES 2009 corridors, bottleneck and T-junction) in the world of its lab layout,
`juelich_<set>_<k>`, whose floor is the layout's free area with every pillar and inner wall walled in, one world per
distinct layout of a set. Positions are read in meters, also from the run files kept in centimeters.
`arena asset import scenario madras <topview_1a ... topview_2f>` writes a TopView sequence of the Fete des Lumieres 2022
in Lyon to `madras_topview`, and `arena asset import scenario glow <YYYY-mm-ddTHHMM>` a 5-minute period of the GLOW
2019 recording in Eindhoven to `glow` (the first import indexes the 1 GB recording once). Both play on an open floor
2 m around every point their recording tracked, as neither places its obstacles in the frame of its tracks.

## Arena adapter (`human:=arena`)

`ArenaHumanSimulator` feeds possession back to the engine: a loop at the
possession stream rate (20 Hz) publishes every possessed ped and every
tracked robot on the engine's `world_state` topic, pose and world-frame
velocity both. Possessed peds that are known engine agents go out under
the engine's own agent id, so the engine keeps its internal copy in full
sync: pose slaved to the feed, velocity carried through so neighboring
agents anticipate the motion, autonomy suspended while the lease is
fresh, and the agent resumes from right there when possession ends. Ids
the engine does not recognize become transient external obstacles in its
force pool instead. Either way the crowd sees and avoids a possessed ped,
walking or parked.

Every tracked robot goes out twice under one agent id: under its own name
(`jackal`, `jackal_1`) and under `robot_<i>`, its index in spawn order. The
engine binds both names to the same agent, so an agent-type file reaches the
robot of any model through an attention ref such as `point: robot:robot_0`.
An index name that is already a robot's own name is left out. In a fleet
`robot_0` is the first spawned robot, not the one nearest to the ped.

With `human.robots.visible:=false` the loop leaves every robot out of the feed
and publishes the possessed peds alone. The engine then holds no robot agent,
its pedestrians neither avoid a robot nor resolve a `robot:` attention ref.

`PARAM_NAMESPACE` is `humansim`, and a `humansim.<param>` launch arg sets the same engine param for the whole
run. At every episode reset `_configure_impl` sends the staged `humansim.*` names of
`QueueEpisode.human_params` to the engine node as one `set_parameters_atomically` batch and then calls the
engine's `reset` with `soft: true`. The batch applies as a whole or not at all, so an episode never runs on
half of a stage's human config. The reset call is the engine's episode boundary, where it applies the params,
moves the live agents onto them, and prunes unused planners. See
[Runtime reconfiguration](../../../../humansim/README.md#runtime-reconfiguration). The engine rejects a batch
that holds a param it cannot reconfigure. A rejected batch logs an error with the engine's reason and stays
out of `EpisodeRecord.human_params`, and the episode continues.

A new backend with episode-level params adds its namespace to `Constants.HUMAN_PARAM_NAMESPACES`, sets
`PARAM_NAMESPACE` from it, and implements `_configure_impl`. The `human.` prefix is reserved for params every
backend honors. None are defined, so `queue_episode` rejects it.

World mechanisms (doors, elevators, see `arena_simulation_setup/AUTHORING.md`, Pedestrians and world
mechanisms):

- `pedestrian_teleport` moves the engine agents through `teleport_agents` (see the elevator cabins below),
  then hands an itinerary ped its next leg and sends every moved ped the stimulus `teleported`.
- `set_mechanism_walls` keeps one named engine wall set per key (`__mech/<key>/<i>`), applied in the
  background in call order, the latest state per key wins. An `elevator:<name>` set also registers two
  world objects: `<name>` (type `elevator`, cabin centre, facing out of the door) and `<name>/queue`
  (type `elevator_queue`, in the doorway, line formation growing into the lobby). Every world door is a
  world object `<name>` of type `door` at the doorway centre. Names are authored (env and level stripped).
- `notify_semantic_change` also feeds per-ped aliases: a `dynamic:` entry with
  `semantic_alias: {elevator: elev_a}` receives `elevator.<field>==<value>` for `elev_a`, replayed from
  the current values when the ped spawns.
- `spawn_delay` holds a scenario ped out of the engine (and so out of the roster and the physics sim)
  until that many sim seconds after the spawn call. Its engine id is taken from a range above `1 << 24`.
- A ped whose waypoints carry `dwell` or `{elevator: <name>}` rides (`extra["itinerary"]`, written by the
  scenario task) gets its waypoints one leg at a time (`itinerary.py`): up to the next dwell stop, where it
  stands for `dwell` seconds, or up to the named cabin, where it selects its destination and waits until the
  elevator teleports it. Other cross-level legs are left to the engine's own cabin routing.

The engine runs in the world frame (authored coordinates, levels laid out, no env offset). The adapter owns
the env offset on both sides: spawn poses, waypoints, walls, world objects, regions, robot and possessed poses
are un-shifted going in, agent poses, gesture targets, flow agents and the engine's marker layers are shifted
coming out. Nothing engine-side knows the env.

World objects go to the engine with what the model annotation and the scenario entry say: `bounding_box`,
`hoi`, `capacity`, `satisfies`, `interaction_radius`, `formation`, and seats. The annotation's seats are its
placed `affordances.sit` slots, each at its bbox center facing one of its yaws, drawn per object from the
episode RNG. A scenario entry's `seats` (object-local poses in the `pose` grammar without zone names) override
them, and an empty list clears them. Without a `capacity`, it is the count of the scenario's
`seats`, else the most of its sit slots that can be occupied at once with no two overlapping, and at least one. Seats become the cluster slots a sitter is placed on, so a `SIT_ON` parks the
ped on the seat, not on a ring around the object. An object without seats keeps the ring. A ped approaches
the object's origin, not its seat, so an object with seats and no `interaction_radius` gets one that reaches its
whole bounding-box footprint from beside it: the farthest corner from the origin plus the ped radius. Annotated
seats also go out with their conflicts, the pairs of sit slots that overlap, so the engine never lets two agents
hold overlapping seats and seats as many sitters as fit together.

Each seat also has the height of the surface a body rests on, which stays in the adapter: a sit slot's is the top of
its z range, which the asset build measures; a scenario seat's is its `z`, else that of the sit slot it lies in or
within 15 cm of, so a lying seat along a bed gives its mattress as `z`. Both add the object's own z. A seat inside a
sit slot also goes out with its standing point (`seat_approach`, [`seat_stand`](support.py)): ahead of the seat along
its facing by the slot's front edge plus `SHIN_GAP` and the leg radius, where the calves clear the edge, and never
inside the object's bounding box. The engine walks a sitter there facing where it walks, turns it to the seat's
heading within reach of the point, and parks it on the seat once it faces that way. A seat without one, and any seat a
`LIE_ON` takes, is arrived at from anywhere within 0.9 m. The engine parks an
arrived agent exactly on its seat, and the latched `interactions` frame names the seat each participant of an active
`SIT_ON` or `LIE_ON` holds (`seat_holds()`). An agent parked on its held seat rests on that seat's surface (see
Resting on furniture). An agent on its feet, on a generated ring slot, beside a full object or on a seat of unknown height
stays where the clip puts it.

The adapter follows the engine's latched `interactions` frame and reports the seat indices the participants of
every `SIT_ON` and `LIE_ON` hold on an object with annotated seats through `held_seats()`, by object sim path,
notifying `on_held_seats_change` listeners when they change. Two held seats that overlap compromise the episode
once through the incident callback the node sets with `set_incident_callback`. The base class reports no held
seats.

Elevator cabins reach the engine with the world as teleport zones: `spawn_elevators` sends one
`set_teleport_zones` call with every realized cabin as a `TeleportZone` (`name` and `destinations` as the
mechanism shim knows them, `hold` at the cabin center facing the door, `entry` outside the door side at the
cabin half-size along the door normal plus the 0.3 m doorway depth plus half of `PED_RADIUS`, facing the cabin,
`wait` on the same normal a further `CABIN_DOOR_THRESHOLD` (0.6 m) plus `PED_RADIUS` out, clear of the reach in
which a body reopens a loaded cabin, `footprint` the cabin corners with the door side pulled in by the door
inset plus half the door width plus `PED_RADIUS`, the band in which a body blocks the closing door, `open`
false, the state the shim spawns its cabin doors in, and `delay` -1, so the engine never teleports a rider
itself), an empty list when the world goes. Door changes go out as one `set_zone_access` batch per mechanism
tick. A ride ends in `pedestrian_teleport`, which maps the published
pedestrian names to engine agent ids (the scenario or flow name, else the id itself) and calls `teleport_agents`
with engine-frame poses and the pedestrian's facing. The engine then moves the agent, the physics sim follows
from the next engine frame, and the interpolation snaps a teleported agent instead of sliding it from the
source cabin.

Stair and escalator halves go out in the same `set_teleport_zones` call, after the cabins, as zones the engine
teleports by itself: `entry` at the mouth and `wait` 0.8 m before it, both facing the seam, `hold` on the
axis at the trigger line (`STAIRS_SEAM_INSET` before the seam), `footprint` the half's corners, `destinations`
the partner half for a boarding half and empty for an exit-only escalator half or a half whose partner is
on a level not loaded, `delay` 0, `kind` `stairs` or
`escalator` and `belt` the escalator speed along the axis, positive towards the seam on the boarding half and
negative on the exit half, 0 on stairs. A half counts as stopped once it, the half it boards or a half boarding
it is reported not running through `stairs_changed` (`stair_running`): both halves of a stopped escalator go
out with `belt` 0 and its boarding half with `open` false, every other half is open. Any step between two
engine frames longer than `TELEPORT_JUMP_M` (2 m) is snapped rather than interpolated, so a crossing does not
slide across the map.

`publish_arena_peds` sets each pedestrian's z to the walking surface of the stair half it stands on
(`stair_height`): rising from 0 at the mouth to half the stair's rise at the seam on a bottom half, sinking
to minus half the rise on a top half, 0 off every stair.

## HuNavSim default agent template

`HunavHumanSimulator` derives pedestrian parameters from a `HunavDynamicObstacle`
instance. At module import time
([`hunav/agent.py:295`](hunav/agent.py#L295)), the class-level default is
loaded from:

```
arena_simulation_setup/configs/hunav/default.yaml
```

via `_load_config()`. This file sets the default behavior parameters
(`behavior`, `desired_velocity`, `radius`, `behavior_tree`, etc.) for every
pedestrian not otherwise configured. The path is resolved via
`get_package_share_directory("arena_simulation_setup")` at startup.

## Adding a new BaseHumanSimulator

1. Create `simulators/human/<name>.py` with a class extending
   `BaseHumanSimulator`, implement all eight `_*_impl` abstract methods.
2. Add `<NAME> = "<name>"` to `Constants.HumanSimulator` in
   [`constants/__init__.py`](../../constants/__init__.py).
3. Register a lazy async factory in [`simulators/human/__init__.py`](__init__.py):

```python
@HumanSimulatorRegistry.register(Constants.HumanSimulator.MY_SIM)
async def _my_sim(**kwargs: typing.Unpack[HumanSimulatorOptions]) -> BaseHumanSimulator:
    from .my_sim import MyHumanSimulator
    return await MyHumanSimulator.create(**kwargs)
```

If the implementation supports LLM-driven obstacle generation, call
`_register_task_modes` in the class body to register a `TM_Obstacles`
subclass (with prompt text) under `Constants.TaskMode.TM_Obstacles.PROMPT`.
