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
| `unuse_obstacles()` | call `_remove_obstacles_impl`, then flip all `INUSE` layers to `UNUSED` |
| `remove_obstacles(purge)` | remove all obstacles at or below `purge` layer from both layers. `WORLD` survives unless `purge >= WORLD` |
| `spawn_robot(robots)` | spawn in physics sim, then call `_spawn_robot_impl` |
| `remove_robot(robots)` | remove from physics sim, then call `_remove_robot_impl` |
| `move_robot(robots)` | move in physics sim, then call `_move_robot_impl` |
| `notify_stimulus(agent_id, stimulus, intensity)` | stimulus seam, no-op by default, fed edge-triggered from `continuous_heard_sounds` for `agent:<id>` listeners |

### `HumanSimulator` Protocol surface

`BaseHumanSimulator` satisfies the Protocol the mechanism shim reads from.
Inherited defaults work for all current subclasses, override only for
specialized teleport semantics (e.g. resetting an internal agent list).

| Method | Signature |
| --- | --- |
| `pedestrian_discs()` | `() -> Iterable[tuple[str, tuple[float, float], float]]` (sync, reads `_ped_positions_xy`, radius is `PED_RADIUS`) |
| `pedestrian_teleport(destinations)` | `(Mapping[str, tuple[float, float]]) -> bool` (async, dispatches via `relay_pedestrian_update`) |

### Abstract `_impl` methods

Every subclass must implement:

| Method | Purpose |
| --- | --- |
| `_spawn_obstacles_impl(obstacles)` | human-sim side: register or prepare static obstacles |
| `_spawn_dynamic_obstacles_impl(obstacles)` | human-sim side: register pedestrian agents |
| `_remove_obstacles_impl()` | human-sim side: signal removal of current episode's obstacles |
| `_spawn_walls_impl(walls)` | human-sim side: ingest wall geometry |
| `_spawn_doors_impl(doors)` | human-sim side: ingest door geometry |
| `_spawn_robot_impl(robots)` | human-sim side: register robots |
| `_remove_robot_impl(robots)` | human-sim side: deregister robots |
| `_move_robot_impl(robots)` | human-sim side: update robot positions |

## GaitGenerator as articulation ground truth

`BaseHumanSimulator.publish_arena_peds` is the single point where skeletal
joint angles are committed to the `arena_peds` bus.  For each `Pedestrian` in
the outgoing message it checks `ped.joint_state.name`:

- **non-empty**: upstream backend supplied its own joint state, published unchanged (override path: an upstream producer that already computes joint angles).
- **empty**: `publish_arena_peds` calls `AnimationManager.compute` (`GaitGenerator` for idle/walk/run, clip playback and overlays for everything else) and fills the field with bare semantic joint names (36 DOF, no body suffix).

The filled field feeds the ROS4HRI skeleton in rviz through `hri_producer` and the Isaac and Gazebo pedestrian rigs.

## Gestures

`Pedestrian.gestures` (`arena_people_msgs/Gesture[]`: `slot` in `head|arm|arm_l|arm_r|body`, `at` world-frame
point for the aimed slots, `clip` name on `body`, `hand` on `arm`) are attention channels, not poses. Backends
only forward them (arena_humansim copies `AgentState.gestures` one to one, the possession stream carries them
as-is), and `publish_arena_peds` hands them per ped to the [`GestureLayer`](gestures/__init__.py) as a
`GestureRequest` (a `Channel(slot, at, clip, hand)` per entry, ped pose, moving flag). The layer resolves each
aimed target into the ped frame once per clip, asks the slot's generator for frames (`head` -> `look`,
`arm*` -> `point`, `body` -> `clip`, the cached `.npy` named on the wire over the joints
[annotations.yaml](animations/annotations.yaml) lists for it), and plays them as independent `AnimationManager`
overlay slots (`head`, `arm`, `body`) over the locomotion base. Nothing is synthesized: no channel published =
that body part idle. See [gestures/README.md](gestures/README.md) for slots, hand resolution,
timing, and how to add a kind.

An empty list releases every slot, a missing entry releases that slot. Unknown slots warn once per ped and are
ignored. Clips come from the baked pointing table and are generated inline on the publishing tick, so their
timing is sim-deterministic under lockstep without any extra plumbing.

## Visualization topics

Pedestrian visualization flows through one data feed plus a few marker layers, all at
env level (`<env_ns>/`) and shared by every backend:

| Topic | Producer | QoS | Role |
| --- | --- | --- | --- |
| `arena_peds` | each backend (`publish_arena_peds`) | reliable, volatile | Pedestrian state feed (positions/velocities/joint_state). `joint_state` carries bare semantic joint names filled by `GaitGenerator` unless the backend overrides it (non-empty = upstream wins). |
| `humans/bodies/tracked`, `humans/persons/*`, `humans/bodies/<id>/joint_states` | `hri_producer` node | per REP-155 | **Canonical** ROS4HRI projection of `arena_peds`: id lists, per-person engagement, per-body joint states, per-body URDF on param `human_description_<id>`, TF `body_<id>`. |
| `pedestrian_markers/extra` | base class (`publish_markers`) | best-effort, volatile | Backend-internal debug overlay (e.g. arena_humansim forwards its planner viz). Off by default. |
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
| `hunav` | `HunavHumanSimulator` | [`hunav/hunav.py`](hunav/hunav.py) | integrates with HuNav, see [configs/hunav/README.md](../../../../arena_simulation_setup/configs/hunav/README.md) for the per-agent schema. `hunav_msgs` is imported lazily so the registry entry loads without it installed |

## arena_humansim agent types

`ArenaHumanSimulator` derives pedestrian parameters from the `agent:` block of
each `dynamic:` scenario entry (`ArenaHumanDynamicObstacle`,
[`arena_humansim/__init__.py`](arena_humansim/__init__.py)). `agent:` accepts
either a bare string or a dict:

```yaml
dynamic:
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
| `desired_velocity` | Either a scalar (m/s), or `{min, max}`: a uniform range this instance's velocity is drawn from. Overrides (does not compose with) the sampled `AgentType.desired_velocity`. Default `{min: 1.0, max: 1.5}`. |
| `radius` | Agent collision radius (m), overrides the sampled `AgentType.agent_radius`. Default `0.35`. |

`waypoint_mode` is a sibling key of `agent:` (not nested inside it), one of `repeat` (default), `reverse`, `once`, `random`: it controls how the entry's `waypoints:` list is replayed once exhausted.

A `regions:` source entry's `config.agent:` block uses a different, wider schema for continuously spawning pedestrians: `agent_type`, `desired_velocity: {min, max}`, `agent_radius` (note: `agent_radius`, not `radius`, here), `behavior_tree`, and `sink_affinity: [{sink, weight}]`. See `_add_source_region` in [`arena_humansim/arena_humansim.py`](arena_humansim/arena_humansim.py).

### Static world objects

A `static:` scenario entry that pedestrians can interact with (a bench, a
reception desk) is registered as a `WorldObject` by `_spawn_obstacles_impl` in
[`arena_humansim/arena_humansim.py`](arena_humansim/arena_humansim.py). The
keys below are written directly on the entry, as siblings of `name:`/`model:`/`pose:`
(they land on `Obstacle.extra`, a copy of the entry's full raw dict):

| Key | Meaning |
| --- | --- |
| `type` | Object type string, matched against a behavior-tree step/action `target:` (falls back to the model's `annotation.yaml` name/desc if omitted). |
| `capacity` | Max simultaneous occupants. Default `1`. |
| `satisfies` | `{need: amount}` applied to an agent that completes an interaction here, same shape as an agent-type step's `satisfies:`. |
| `interaction_radius` | Overrides the interaction kind's default approach radius for this object. |
| `formation: {type, params}` | Provider-side formation for agents interacting here (`type` one of `line`, `cluster`, `f_formation`, `dyad`, see [config/agent_types/README.md](../../../../humansim/arena_humansim/config/agent_types/README.md)). |

```yaml
static:
  - name: reception_desk
    ...
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
`publish_arena_peds` at 2 Hz so the topic never goes stale, letting the
possession substitution overlay driven peds on the way out.

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

## Arena adapter (`human:=arena`)

`ArenaHumanSimulator` feeds possession back to the engine: a loop at the
possession stream rate (20 Hz) publishes every possessed ped (plus dirty
robot poses) on the engine's `world_state` topic, pose and world-frame
velocity both. Possessed peds that are known engine agents go out under
the engine's own agent id, so the engine keeps its internal copy in full
sync: pose slaved to the feed, velocity carried through so neighboring
agents anticipate the motion, autonomy suspended while the lease is
fresh, and the agent resumes from right there when possession ends. Ids
the engine does not recognize become transient external obstacles in its
force pool instead. Either way the crowd sees and avoids a possessed ped,
walking or parked.

The engine runs in the world frame (authored coordinates, levels laid out, no env offset). The adapter owns
the env offset on both sides: spawn poses, waypoints, walls, world objects, regions, robot and possessed poses
are un-shifted going in, agent poses, gesture targets, flow agents and the engine's marker layers are shifted
coming out. Nothing engine-side knows the env.

World objects go to the engine with what the model annotation and the scenario entry say: `bounding_box`,
`hoi`, `capacity`, `satisfies`, `interaction_radius`, `formation`, and `seats` (object-local `{x, y}` with an
optional `yaw`, the same grammar as `pose`, from the annotation or overridden per scenario entry, and an empty
list clears the annotation's). Seats become the cluster slots a sitter is placed on, so a `SIT_ON` parks the
ped on the seat, not on a ring around the object. An object without seats keeps the ring. A ped approaches
the object's origin, not its seat, so an object whose seats sit outside `interaction_radius` needs that radius
widened to cover the object or the ped never gets close enough to sit.

## HuNavSim default agent template

`HunavHumanSimulator` derives pedestrian parameters from a `HunavDynamicObstacle`
instance. At module import time
([`hunav/__init__.py:326`](hunav/__init__.py#L326)), the class-level default is
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
async def _my_sim(**kwargs):
    from .my_sim import MyHumanSimulator
    return await MyHumanSimulator.create(**kwargs)
```

If the implementation supports LLM-driven obstacle generation, call
`_register_task_modes` in the class body to register a `TM_Obstacles`
subclass (with prompt text) under `Constants.TaskMode.TM_Obstacles.PROMPT`.
