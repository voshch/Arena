# arena_runtime sim interface

`BaseSim` and the four sub-interfaces it combines. Implementations are
registered in `SimulatorRegistry` and instantiated by key at runtime. See also
[human simulator](../../../../arena_runtime/arena_runtime/arena_runtime/humans/README.md) for the pedestrian-logic counterpart
(`BaseHumanSimulator`).

## Sub-interfaces

Defined in [`_interface.py`](_interface.py):

### `ObstacleITF`

[`_interface.py`](_interface.py)

| Abstract method | Signature |
| --- | --- |
| `obstacle_spawn` | `(Sequence[Obstacle]) -> Sequence[bool]` |
| `obstacle_move` | `(Sequence[Obstacle]) -> Sequence[bool]` |
| `obstacle_delete` | `(Sequence[Obstacle]) -> Sequence[bool]` |

### `PedestrianITF`

[`_interface.py`](_interface.py)

| Abstract method | Signature |
| --- | --- |
| `pedestrian_spawn` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_move` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_delete` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_update` | `(Pedestrians) -> Sequence[bool]` |

### `RobotITF`

[`_interface.py`](_interface.py)

| Abstract method | Signature |
| --- | --- |
| `robot_spawn` | `(Sequence[Robot]) -> Sequence[bool]` |
| `robot_move` | `(Sequence[Robot]) -> Sequence[bool]` |
| `robot_delete` | `(Sequence[Robot]) -> Sequence[bool]` |

### `WorldITF`

[`_interface.py`](_interface.py)

| Abstract method | Signature |
| --- | --- |
| `spawn_walls` | `(Sequence[Wall]) -> bool` |
| `spawn_floors` | `(Sequence[Floor]) -> bool` |
| `remove_world` | `() -> bool` (default raises `NotImplementedError`) |

### `MechanismITF`

[`_interface.py`](_interface.py): door, elevator and stair orchestration. Provides concrete default implementations driven by an internal sim-time tick loop ([`_mechanism_shim.py`](_mechanism_shim.py)) on top of four box/robot primitives. Any simulator that implements the primitives gets door animation, elevator rides, and ped/robot teleport for free, and simulators with native support can override the four top-level methods.

| Default method | Signature | Purpose |
| --- | --- | --- |
| `spawn_doors` | `(Sequence[Door]) -> bool` | spawn door geometry + register runtime; starts the tick loop |
| `remove_doors` | `(Sequence[str]) -> bool` | delete door geometry by name |
| `spawn_elevators` | `(Sequence[Elevator]) -> bool` | spawn cabin walls + synthesized door per elevator |
| `remove_elevators` | `(Sequence[str]) -> bool` | delete cabin geometry + synthesized door by name |
| `spawn_stairs` | `(Sequence[Stairs]) -> bool` | spawn the boxes of each stair or escalator half, register its runtime and `stairs`/`escalator` semantics |
| `remove_stairs` | `(Sequence[str]) -> bool` | delete stair boxes and runtimes by name |
| `remove_mechanisms` | `() -> bool` | remove every stair, elevator and door (world swap) |
| `stop_mechanisms` | `() -> None` | cancel the tick loop (call on shutdown) |
| `attach_human_simulator` | `(HumanSimulator) -> None` | bind the human-sim Protocol that supplies ped positions + ped teleport |
| `spawn_objects` | `(Sequence[Obstacle]) -> Sequence[bool]` | spawn obstacles, an articulated one as a physics model or its link obstacles (see [Articulated objects](#articulated-objects)) |
| `move_objects` | `(Sequence[Obstacle]) -> Sequence[bool]` | move obstacles, an articulated one reset to the initial values of its `joints` |
| `delete_objects` | `(Sequence[Obstacle]) -> Sequence[bool]` | delete obstacles, an articulated one with its link obstacles and semantics |

| Primitive (override required for default behavior) | Signature |
| --- | --- |
| `spawn_box` | `(name, size, pose, material=None, *, velocity=None, comb=False) -> bool` |
| `move_box` | `(name, pose) -> bool` |
| `move_boxes` | `(Mapping[str, Pose]) -> bool` (optional, defaults to one `move_box` per entry) |
| `delete_box` | `(name) -> bool` |
| `set_robot_pose` | `(sim_path, pose) -> bool` |

Robot tracking is not a primitive to override: call `_register_agent_robot(robot, model_params)` on spawn and `_forget_agent_robot(sim_path)` on remove, and the base class serves `robot_discs()` / `robot_pose()` from TF.

Lights follow the same pattern. `spawn_lights(lights)` and `remove_lights()` keep the light table, and every tick pushes each `light` semantic entity whose output changed, after moving frame-anchored and link-riding lights that moved more than 2 cm or 0.02 rad. A simulator renders lights by overriding four no-op hooks:

| Light hook | Signature | Purpose |
| --- | --- | --- |
| `_spawn_lights` | `(Sequence[Light]) -> bool` | create the light entities, dome and sun replacing the default lighting |
| `_remove_lights` | `(Sequence[Light]) -> bool` | delete them and restore the default lighting |
| `_apply_light` | `(light, level, alive) -> None` | render at `level` in 0..1 with the per-fixture alive mask |
| `_move_light` | `(light, position, yaw) -> None` | place a frame-anchored or link-riding light, its direction turned by `yaw` |

A light with `glow` also makes that material of its `owner` entity emit the light's color at `level`: Gazebo recolors the model's `glow_<material>` visual over `/world/default/material_color`, Isaac sets the material's emissive color on the session layer, MuJoCo sets the emission and color of the placement's own glow materials before each render.

The `HumanSimulator` Protocol the shim consumes from the attached human-sim:

| Method | Signature |
| --- | --- |
| `pedestrian_discs` | `() -> Iterable[tuple[str, tuple[float, float], float]]` (sync, ground truth: name, xy, radius) |
| `pedestrian_teleport` | `(Mapping[str, tuple[float, float]]) -> bool` (async) |
| `elevator_doors_changed` | `(Mapping[str, bool]) -> None` (async, cabin name to door open) |
| `stairs_changed` | `(Mapping[str, bool]) -> None` (async, escalator half name to running) |

Optional, looked up with `getattr` so human sims without it keep working:

| Method | Signature |
| --- | --- |
| `set_mechanism_walls` | `(key: str, segments: list[((x0, y0), (x1, y1))] \| None) -> None` (map frame, `None` removes the set) |

The shim calls it with `elevator:<name>` (back and side cabin walls) at elevator spawn and `None` at removal,
and with `door:<name>` (the door segment) only when a door changes between "cannot open for pedestrians"
(gate effectively locked with no authorized agent near, or `activation_distance <= 0` with no pressed plate
driving it) and "can open".

The attachment happens in `BaseHumanSimulator.__init__` (see [human simulator](../../../../arena_runtime/arena_runtime/arena_runtime/humans/README.md)). Without an attached human-sim, the shim still drives doors against `robot_discs` alone and logs a no-op for ped teleports.

Pedestrians ride like robots: the shim counts a ped inside a cabin footprint as an occupant, and when the cabin door has closed behind it the ride hands the destinations to `pedestrian_teleport`. An empty cabin opens for a robot or ped outside within `activation_distance` of its door, a cabin with an occupant only for one within `CABIN_DOOR_THRESHOLD` (0.6 m), so riders queuing for the next trip do not hold a loaded cabin. The shim also tells the human-sim which cabin doors are passable: each tick it collects the cabins whose door turned `open` or left `open` since the last report, starting from closed at spawn and including the close an episode reset causes, and sends them in one `elevator_doors_changed` call, nothing on ticks without a change.

#### Shim behaviour

- Door kinds: `sliding` (slides along start->end), `sliding_top` (lifts), `teleport` (drops away) and
  `hinged` (rotates about `start` by `swing_direction * progress * 90 deg`). Sliding and hinged slabs are tested
  against agent discs: an opening step holds when the slab would enter an agent (the swing arc of a hinged
  door, the wall pocket past the end of a sliding one), so an agent pressed against a half-open sliding slab
  does not stop it, and a blocked close re-triggers the door and sets `close_blocked` (semantic
  `blocked_close`) until the remaining path is clear.
- Elevators: the cabins linked through their destinations form a group served by one car, which starts at
  the group's first cabin by name and returns there on reset (`_CarState` per cabin: `present`, `departing`,
  `absent`, `arriving`). Only the cabin holding the car opens its door, so at most one door per group is open,
  and a tick with two is reported as an incident. A call at a cabin without the car (an agent in the call
  radius, or anyone shut inside) waits in `call_since`, and the idle car fetches the oldest call with an empty
  trip. `destination` may hold several realized cabin names (comma separated). A loaded departure goes to
  `_ElevatorRuntime.current_destination()`: the written `target` if set, else round-robin (one step per
  departure). `capacity` (live occupants) suppresses outside calls when full.
  `call_mode: edge` makes outside calls edge-triggered (newcomer in the call radius, or a shut idle cabin),
  so agents waiting in the lobby do not hold a boarded car's door open, while an empty car stays open for
  them instead of closing and reopening. `level` (default) re-fires every tick.
  Door holds are bounded: outside callers keep a loaded car open at most `hold_time` after its first boarder
  entered, an empty car stops answering them once a call elsewhere has waited `hold_time`, and arrivals hold
  the door at most `TELEPORT_GRACE` (10 s sim time) while a call waits, riding along if still inside when
  the car leaves. A recall fetches the car to the recalled cabin and holds it there open.
- Failed rides: a refused/raising robot or ped teleport releases the riders at once (source unsealed, the
  destination stops waiting), and the riders left in the source cabin call the car back. A silent failure is
  released after `TELEPORT_GRACE` (10 s sim time) with a warning.

#### Stairs and escalators

A stair half ([`Stairs`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/world.py)) arrives realized: name prefixed, position in the map frame, `to` naming the partner half on the other level. Its side linings, side walls, seam wall and solid rails come in as ordinary walls through `Level.all_walls`, the open rails and an escalator's skirt brushes through `Level.closed_walls`, each line following the walking surface, so a sloped segment (end higher than start) spawns as boxes pitched along it, stacked in strips (`upright_strips`) thin enough that their ends stand within `WALL_LEAN` (1 cm) of vertical. `spawn_stairs` adds the solid geometry agents walk on, each box in the material of its surface (`Stairs.material_of`).

| Half | Boxes | Velocity |
| --- | --- | --- |
| stairs | `<name>/step_<i>`, one per `steps()` box, solid from the floor (or the well bottom) to the tread, with `risers` set a tread plate on `<name>/riser_<i>` | static |
| escalator | `<name>/base_<i>` from `base()`, slabs just under the treads along the ramp and under the landing | static |
| escalator | `<name>/plate_<i>` from `plates()`, landing plates flush with the floor over the pit before each mouth, spawned with `comb` | static |
| escalator | `<name>/tread_<k>` from `treads(0.0)`: step blocks, or planks on a `belt` | `tread_velocity` |

Objects dressing a staircase (`Stairs.static_models()`, a surface `model` per box plus the staircase's `models`) spawn through `obstacle_spawn` as `<name>/model_<part>`, the ones riding treads (`Stairs.tread_models`) move with them through one `obstacle_move` per tick, and `remove_stairs` deletes them through `obstacle_delete`. A simulator that cannot load a model skips it.

Under `world.levels:=stack` both halves of a pair are joined (`Stairs.joined`): the bottom half spawns the whole stair from mouth to mouth (`Stairs.whole_stair()`, one tread loop, a landing at each end), the top half spawns nothing, and the halves stay for membership, semantics and the pedestrian simulation. A box spawned with a velocity, zero included, is a gravity-free body that moves at it and carries what stands on it through contact friction. Isaac's `Wall` and `Prim` and MuJoCo's `Prim` carry this as `is_static` (default true), which the adapters set from `velocity is None`. `tread_velocities` gives each tread the belt speed along the axis and the climb of its top, level over a step escalator's landing, and the shim sends a tread's new velocity when it changes. The treads never touch the base, the floor, the side walls or each other. They pass through the landing plates, which `comb` makes possible: Gazebo gives comb boxes collide bitmask `0x01` and moving boxes `0xfffe`, MuJoCo gives both contype 1 and conaffinity 0, and Isaac needs nothing since PhysX makes no contacts between its kinematic treads and static boxes.

| Simulator | Moving box |
| --- | --- |
| Gazebo | non-static 1000 kg model with gravity off and the `VelocityControl` system, `set_box_velocities` publishes `/model/<name>/cmd_vel` in the model frame |
| Isaac | `Wall.velocity` with `is_static` false: a kinematic rigid body the server advances by `RigidBodyView.set_kinematic_targets` every physics step, `EditPrims` with `velocity` changes it, a pose edit teleports it |
| MuJoCo | `Prim.velocity` with `is_static` false: a free joint body with `gravcomp = 1` whose qvel the server sets before every step, `EditPrims` with `velocity` changes it |
| flatsim | an ordinary box that follows `move_boxes`, Box2D boxes carry nothing |
| dummy | `spawn_stairs` is a no-op |

Isaac spawns every box through `SpawnWalls`, which carries yaw only, and sets the full orientation of a tilted box (escalator base, belt planks) with a follow-up `EditPrims`. The well of a top half sinks to `-rise/2`, so the Gazebo, Isaac and MuJoCo ground planes lie at z = -3 m, under the level floors.

Under `world.levels:=stack` the simulator frame differs from the abstract one the rest of the env uses. `Realizer.placement()` gives the shift of every level, and a simulator with `STACKS_LEVELS` (Gazebo, Isaac, MuJoCo) places each pose it receives by the level whose abstract bounds hold it: walls, floors and ceilings through their spawn pose, boxes, entities, lights and robot teleports directly. It publishes each robot's `map -> odom` itself as the inverse shift of the level the robot stands on, updated on a teleport and on `robot_changed_level`, which the shim calls when a robot walks past the seam of a joined pair (`Stairs.joined`, no teleport there). Pedestrians reach the simulator on `sim_peds`, the placed copy of `arena_peds`, which the Gazebo pedestrian plugin reads. The Isaac and MuJoCo viewports take a robot frame from `map` minus that robot's map offset, and publish `camera_pose` plus the offset of the robot they track.

Each tick the shim then:

- Counts the robots (`robot_discs`) and pedestrians (`pedestrian_discs`) inside each half's footprint.
- Teleports robots across boarding seams: a robot on a half with a spawned partner (`Stairs.boards` and the `to` half among the spawned stairs) that stands past `Stairs.crossed` goes through `set_robot_pose` to `Stairs.landing(partner, x, y, yaw, travel, partner_travel)`, roll and pitch kept. The landing lies on the partner tread at the spot matching the robot's spot on its own tread, at most `STAIRS_LANDING_BACKOFF` short of the partner's trigger line, heading away from its seam, and z changes by the difference of the two tread tops, so the robot keeps its clearance. Both halves then hold the robot latched until it is seen outside their trigger band, so a stale pose on the source or a robot standing at the landing never sends it back, while a robot that walks back through the partner's seam crosses again. A refused or raising teleport is one incident (`<sim_path>: stair teleport '<from>' -> '<to>' failed: <reason>`) and is retried only after the robot left the trigger band. A half whose partner is not spawned (outside a `level_filter`, or dropped at load) lands nowhere and logs one info line at spawn, an exit-only escalator half never teleports. Pedestrians are teleported by the human simulator, the shim only observes their jump.
- Runs escalator belts: `travel` advances by `Stairs.belt * dt` (towards the seam on a boarding half, mouth-ward on an exit half or one whose partner was dropped) and every spawned tread is re-posed to `treads(travel)`, where the velocity took it, all treads of all halves in one `move_boxes` call. Every tread stays on the stair: one that runs off the far end comes back at the mouth sunk `Stairs.sink` (1 mm on steps) through the landing plate (`STAIRS_PLATE` thick, flush with the floor, over a pit the floor opens for it), slides out just below the floor and rises into place once clear of the plate. On a whole stair the wrap itself happens under the plates at both ends. A stopped escalator keeps `travel` and gets zero tread velocities through `set_box_velocities`, a restarted one its belt velocity again. Gazebo sends that call as one gz `/world/default/set_pose_vector` request run off the event loop and drops a tick's batch while the previous one is still in flight, Isaac and MuJoCo send one fire-and-forget `EditPrims` with every tread, flatsim skips the treads it never spawned. The stairs_demo world (four escalator halves of nine treads) costs one request and 36 poses per tick, which take about 0.2 ms of tick time to compute, instead of 36 `SetEntityPose` calls per tick (1080 per second at 30 Hz).
- Reports `running` changes of escalator halves to the human simulator in one `stairs_changed` call, starting from running at spawn and including the restart an episode reset causes.

Semantic kinds `stairs` and `escalator` attach to every spawned half on gazebo, isaac, mujoco and flatsim:

| Field | Kind | Type | Meaning |
| --- | --- | --- | --- |
| `occupants` | stairs, escalator | continuous | agents inside the half's footprint, robots by sim path and pedestrians by name |
| members | stairs, escalator | | those agents, robots first |
| `just_crossed` | stairs, escalator | predicate | true for the one tick after a robot teleport through this half's seam, or after a pedestrian on this half shows up on its partner |
| `running` | escalator | discrete, writable | `true` (initial) or `false`, a write sets both halves of the pair, a stopped belt freezes its treads |
| `direction` | escalator | discrete | `up` when the boarding half of the pair is the bottom one, `down` when it is the top one, on both halves, empty without a boarding half |

An episode reset puts every belt back at rest and running, and clears latches, members and `just_crossed`.

#### Articulated objects

The human simulator places every static obstacle through `spawn_objects`,
`move_objects` and `delete_objects` ([`_articulation.py`](_articulation.py)),
which hand a plain obstacle unchanged to the simulator's `obstacle_*` verb.
An obstacle whose object annotation carries `articulation:` and that has link
bundles (`links/<link>/` under the object) spawns as a physics model in a
simulator with `ARTICULATED_PHYSICS` (MuJoCo, Isaac, Gazebo, see below) and
otherwise (flatsim, dummy) expands into one static obstacle per link bundle, named and sim-pathed `<obstacle>__<link>`, its model the link
bundle, posed at `obstacle.pose * link_pose` with the link translation scaled
componentwise by `obstacle.scale` and the scale kept on the link. Each call
sends all its obstacles to the simulator in one batch. Links without meshes
have no bundle and no obstacle. These simulators move joints through the
`obstacle_move` they already have, flatsim places each outline at the z span
of the link box through its full pose.

Each articulated obstacle keeps initial (`articulation.resolve(obstacle.joints)`),
current and target joint values. A `move_objects` call, which is how the human
simulator re-places a known obstacle at an episode reset, takes the new pose and
`joints`, resets current and target to the new initial values and moves every
link. On the mechanism tick each joint steps toward its target at its speed and
the links whose pose changed go to `obstacle_move` in one batch per tick. A step
that would put a moving link's footprint (its link box through the object and
link pose, projected to xy) onto a robot or pedestrian disc holds the whole
object for that tick and marks it `blocked`, it resumes once clear. The
`articulated` semantic kind is attached under the env-prefixed obstacle name on
spawn and detached on delete, its `reset()` returns the joints to their initial
values and the next tick places the links, see
[AUTHORING.md](../../../../arena_simulation_setup/AUTHORING.md#articulated-objects)
for its fields. A light whose `owner` is an articulated obstacle and whose
`link` names one of its non-root links rides on that link: `_apply_lights`
carries its spawn pose through the link motion at the current joint values
(`link_light_pose`) and moves it through `_move_light` with the yaw of that
motion, on the same tick that steps the joints.

A physics simulator implements `_articulated_spawn` and `_articulated_drive`.
`make_physical` derives each link's mass, center and principal moments from
its box (`link_inertias`: `DENSITY` 150 kg/m3 clamped to `MASS_RANGE`, a link
without a box a 0.05 kg point at its joint) and each movable joint's
`JointDynamics` (`joint_dynamics`): the friction, damping, armature and
spring the joint's `dynamics` authors, a `friction_coefficient` taken as
that many times the weight the joint moves. What it leaves out comes from
the joint's role (`ANGULAR_ROLES`, `LINEAR_ROLES`), else from its kind
(`ANGULAR`, `LINEAR`), with a default friction raised to `HOLD_MARGIN` times
the worst static gravity load over the joint's range. The drive effort is
`EFFORT_PER_FRICTION` times the friction plus the spring's pull across the
range plus the damping at the joint's speed, with the kind's floor, its
stiffness saturates at the tuning's gap and its damping is critical on the
inertia the joint moves. A link material reaches the simulators as
`surface`: static and dynamic friction, an unset one taking the other's
value or `SURFACE_FRICTION`, and restitution. `_articulated_spawn`
builds the model from them, its root fixed at the obstacle pose and the
links nested at the model frame, contacts within the object and with the
floor off. `move_objects` and `delete_objects` hand the object itself to
`obstacle_move` and `obstacle_delete` (Gazebo respawns a moved one) and
place its joints at the initial values. On each tick `_articulated_drive`
gets one `JointCommand` per object: a semantic write
(`ArticulatedObject.command`) marks the joints it moves as driving, each
driving joint's reference ramps toward its target at the joint's speed but
at most `LEAD` (0.25 s of motion) ahead of the joint, and a joint within its
settle gap leaves the drive and is set onto its target (`place`). The
reading that comes back is the current value, clamped into each joint's
range, a joint no drive holds takes its target along, and `blocked` holds
while a driving joint lags its reference by `LAG_BLOCKED` of the lead. A
command whose placement got no reading is placed again on the next tick.

| Simulator | Model | Drive and reading |
| --- | --- | --- |
| MuJoCo | `/mujoco/SpawnArticulated`: one body tree with hinge or slide joints (damping, friction loss, armature, joint stiffness at `springref`), explicit inertia, the link meshes and their convex pieces at the material's dynamic friction and a contact `solref` from its restitution, contacts excluded within the object and with the world body | `/mujoco/DriveArticulated`: one position actuator per joint, gains switched in the compiled model and the spec, `qpos` read back in the same call |
| Isaac | `/isaac/SpawnArticulated`: a fixed-base articulation of rigid bodies referencing the link USDs, their meshes colliding through convex decomposition, revolute or prismatic joints with drives that carry the spring toward its rest while off, a bound physics material per link with one, self collision off, floor colliders filtered | `/isaac/DriveArticulated`: DOF gains, friction efforts, armature and targets through the articulation view once physics has parsed it |
| Gazebo | an SDF model with the root joined to the world, link geometry from the link SDFs with the material's dynamic friction (`mu`, `mu2`) and restitution, joints with damping, friction and `spring_stiffness` at `spring_reference`, moving links masked off the floor (`collide_bitmask`) | the `ArticulatedDrive` system of arena_gz_plugins on `/model/<name>/articulated/command` and `/state` (gz transport, `gz.msgs.Double_V`) |

### `SimLifecycle`

[`_interface.py`](_interface.py): process-singleton host counterpart to `BaseSim`. Owned by `arena_runtime_node` (one instance per process), drives sim-wide pause/unpause and namespace cleanup that cuts across env-scoped simulators.

| Abstract method | Purpose |
| --- | --- |
| `pause` / `unpause` | toggle the underlying sim clock |
| `cleanup_namespace(prefix)` | delete all entities under `prefix`, return count removed |
| `ensure_ready` | block until the sim's services are reachable |

Registered in `LifecycleRegistry` alongside `SimulatorRegistry` ([`__init__.py`](__init__.py)).

`arena_runtime_node` wraps the registered `SimLifecycle` in a liveness-accounting
proxy: two consecutive `pause`/`unpause` failures (`False` or raised) or step
failures mark the sim dead, published latched on `state/sim`
([`SimState.msg`](../../../arena_runtime_msgs/msg/SimState.msg)), see the
[arena_runtime README](../README.md) topic table.
Adapters signal a dead transport (a timed-out service call, not a genuine
rejection) by raising [`SimUnavailable`](_interface.py) instead of returning
`False`. `cleanup_namespace` should let it propagate. `arena_runtime_node` never
retries or recovers a dead sim on its own.

## `BaseSim`

[`__init__.py`](__init__.py)

```python
class BaseSim(NodeInterface, ObstacleITF, PedestrianITF, RobotITF, WorldITF, MechanismITF, abc.ABC):
```

Overridable hooks (concrete default no-ops):

| Method | Purpose |
| --- | --- |
| `before_reset_episode()` | episode-boundary hook before every reset, for sim work that must not ride the mid-episode pause path; default no-op |
| `after_reset_episode()` | episode-boundary hook after every reset; default no-op |
| `step(n=1)` | advance simulation by `n` ticks; default no-op returns `True` |

## Lockstep

`SimLifecycle.step_seconds(seconds) -> float` advances a held sim by an exact
sim-time delta, quantized to whole physics steps (`physics_dt` rosparam,
default 0.0333 matching `empty.sdf`). The base raises `NotImplementedError`.
Gazebo sends one `ControlWorld` request with `pause=True` and `multi_step=n`
(both fields in the same message, else gz free-runs). A step of more than one
tick first raises gz `real_time_factor` so the limiter does not pace it, and
the next unpause restores it. A single-tick step leaves the factor alone. A step
returns once `/clock` carries its target, and the next unpause first waits for
`/world/default/stats` to report gz paused at or past that target, since gz undoes
an unpause that lands while a batch is still running. Isaac calls
`/isaac/StepSimulationN` and awaits `/clock` reaching the projected target.
MuJoCo calls `/mujoco/Step` with `seconds / 0.002` physics steps (the server
publishes `/clock` and drains its sensor and control pumps every 20 ms of sim
time inside the call) and awaits `/clock` the same way. Its pause, unpause
and namespace cleanup calls are idempotent and are resent every 10 s until
answered, so a request or reply lost in transit cannot hold an env forever.
The dummy host adds the delta to its synthetic clock, so `sim:=dummy` runs the
whole control plane without a physics engine. The flatsim host sends
`control_world` with `multi_step` to its server and awaits `/clock` reaching
the time the server returns (see [flatsim backend](#flatsim-backend)). The
dummy, flatsim and gazebo hosts are what
[`tests/ros/`](../../tests/ros/) exercises (gazebo against a headless gz server on an empty world), the gate math itself is
[`lockstep_gates.py`](../lockstep_gates.py) and has ROS-free unit tests.

`arena_runtime_node` exposes it as `sim_lifecycle/step` (valid only while holds are
active and no unpause window is open), plus the lockstep control plane
driving the [`LockstepScheduler`](../lockstep.py). Every adapter advances at
least one physics step for any positive delta, so a delta below half a step
is exactly one step. The env node steps that way to teleport robots
during a reset hold (see the [manager README](../../../../arena_runtime/arena_runtime/arena_runtime/env/README.md)).

Components self-register their data channels via
`sim_lifecycle/lockstep/register` (LockstepRegister: `caller`, `env`,
`channels[]` of `{name, topic, type, period_s, hard}`. Re-registering under
a `caller` replaces its set, an empty `channels[]` clears it, per-env
registrations drop when their env despawns, and registrations merge into a
running lockstep live. The arena_humansim adapter self-registers `engine`
and `peds` (both hard). Its roster never trails the engine: every `arena_peds`
is stamped no earlier than the last engine frame, and removing all agents
publishes an empty roster at that stamp, so `peds` is covered whenever
`engine` is. The hunav adapter self-registers `roster` (hard).
The arena_auditory `array_renderer` registers `audio/<robot>` (hard, one
render block, 20 ms for `auditory.array.spec:=four_mic`) and renders off
`/clock` for the same reason as the humansim engine below.
`robot.hearing:=srp|seld` also registers `hearing/<robot>` per fleet robot
(hard, one front-end hop, `LockstepHeartbeat` stamped with the audio
consumed), and the arena_hearing stack registers `belief/<robot>` and
`policy/<robot>` per fleet robot (hard, their publish periods) so the sim
never runs ahead of a belief grid or a speed mask.
The arena_robots task_server registers per-robot beats only while a goal is
active: `nav/<robot>` (hard, pulsed per cmd_vel) during goto_pose for both
nav2 (one controller period) and the goal-window passthrough stacks whose
control loop Arena does not own (rosnav_rl/external, fixed 0.25 s liveness
period, action held open until arrival), `reach/<robot>` and
`gesture/<robot>` (hard, pulsed per JTC controller_state) during reach_pose
/ play_gesture. The arena_planners bridge (`mobile:=drl`) registers
`planner/<robot>` (hard, no keepalive) for the lifetime of its run loop,
with the period snapped to a whole number of physics steps and every pulse
stamped with the observation time its action was computed for, so a
lockstep run steps exactly one planner tick per action: obs at the frozen
boundary, action, step. The `none`, `manual`, and `test_collision` bringups
keep fire-and-forget goto_pose and stay ungated. `arena cam ... --lockstep` registers `cam`
(hard, 1/fps) for the take when a lockstep run is active and, recording or live, films each
frame at the tick the run froze on for it before beating that stamp, so the
take is gated like any other producer and frame-exact at any target rtf. With no
run active it holds and steps the sim itself. A camera server (`arena cam serve`) registers
`cam` under its env for each take and never holds the sim. Beats republish
`LockstepHeartbeat` coverage stamps, and a wall-clock grace watchdog publishes
forward keepalives through legitimately silent phases (MoveIt planning,
nav2 recovery behaviors), pacing the sim at wall rate so a producer that
itself needs sim time to pass cannot freeze the sim. Training-mode robots have no task_server and stay
outside the gate set.
Producers publish with `header.stamp` reflecting the sim time they have
actually covered. A hard channel is satisfied at sim time `T` when its
latest stamp is newer than `T - period_s`, compared at the clock's
nanosecond resolution: the gate asks whether the producer has covered the
clock, never whether it landed on a window boundary, so a producer whose
own grid is phased against the ledger's cannot deadlock the run. Windows
only pick the step sizes the scheduler takes, and a covered hard channel's
next check falls at its latest stamp plus `period_s`, so one beat never
covers more than one period past its stamp, even when physics steps
overshoot the due tick. A producer driven by the sim
clock registers `period_s` equal to its own tick length, which paces the
sim to the producer's true coverage. An elapsed-based rate loop registers
a period strictly above its interval. The arena_humansim engine ticks off
`/clock` itself rather than an rcl timer, so a clock the gate is holding
still owes it the tick the gate is waiting for.

`sim_lifecycle/lockstep/start` (LockstepStart: `target_rtf`, `ungated`)
starts a run or reconfigures one in place: `target_rtf` of 0 is unpaced
while gates hold the pace and wall rate while no channel is registered,
`ungated` free-steps ignoring every gate. Hard channels freeze-and-wait: the
scheduler will not advance past a tick until every hard channel has a
`header.stamp` covering it, so the weakest hard producer sets the pace by
design. `soft` channels are observed and reported only. The scheduler also
idles while a foreign hold is open (episode resets) and rebases its window
grid to whatever sim time passed meanwhile. An unpause window never unpauses
the sim during a run: the scheduler keeps stepping for it, also under a
foreign hold or `pause`, gated on every hard channel except those registered
in the window holder's env. A failed step is retried under the run's hold
after 1, 2, 4, 8 and 16 s, rebasing on the clock first, and a sixth
consecutive failure ends the run and counts one lifecycle failure.
`sim_lifecycle/lockstep/stop` ends the run and restores the pre-run
pause state and gz `real_time_factor`, responding as soon as the hold is
released (the guard against a draining gz chunk re-pausing the world runs
in the background). `sim_lifecycle/lockstep/pause` /
`sim_lifecycle/lockstep/resume` freeze and unfreeze the clock within a run
without ending it. Status is the latched `/arena/state/lockstep` topic
(LockstepStatus): `active`, `paused`, `ungated`, `target_rtf`,
`measured_rtf`, `tick`, `registrations`, `waiting_on`, `arrived`, and the
run's cumulative counters: `gated_ticks` (ticks that waited on a hard
channel) and `counts` (per channel topic, the ticks it was `due` on and
the messages `received`), both reset when a run starts.

`lockstep.autostart`/`lockstep.channels`/`lockstep.target_rtf`/
`lockstep.paused` params are read once at bringup (launch args
`lockstep:=true lockstep.channels:="..."`), registering the channels under
caller `launch`. Channel specs are `name|topic|type|period_s|hard-or-soft`
with `{env}` expanded per env. CLI: `arena lockstep run [rtf:=N] [ungated]`
(blocks, ctrl-c pauses), `on [rtf:=N] [ungated]`, `off`, `pause`, `resume`,
`status`, and `arena lockstep gate [engine|peds|<spec> ...]` to register
extra channels under caller `cli`.

## Sim-paused invariant

The sim is paused for the entire body of `EpisodeNode._reset_episode`. Only
node-discovery and lifecycle signals are observable while the sim is paused;
tf, costmap, and sim-clock topics are not advancing, except for the single
physics steps a robot teleport takes through `sim_lifecycle/step` until TF
shows the robot landed. `switch_controller` is the
one controller_manager call that needs the sim to tick (on both gz_ros2_control
and Isaac's `ros2_control_node`, whose update loops follow sim time), so
controller activation lives inside the robot bring-up unpause window.

## Registered implementations

[`__init__.py`](__init__.py): `SimulatorRegistry` maps
`SimSimulator` keys to async factory functions:

| Key | Class | File | Notes |
| --- | --- | --- | --- |
| `dummy` | `DummySimulator` | [`dummy_simulator.py`](dummy_simulator.py) | no-op entity verbs; `DummyHost` publishes a synthetic `/clock` (wall time minus paused time plus stepped time, also while paused) and supports `step_seconds` |
| `flatsim` | `FlatsimSimulator` | [`flatsim_simulator/flatsim_simulator.py`](flatsim_simulator/flatsim_simulator.py) | client of the flatsim server (2D Box2D, no renderer), one world for every env. `FlatsimHost` pauses and steps the server, which owns `/clock`. Optional feature |
| `gazebo` | `GazeboSimulator` | [`gazebo_simulator/gazebo_simulator.py`](gazebo_simulator/gazebo_simulator.py) | Gazebo (Ignition) via gz-transport. Discovery runs on a port pair per ROS domain (`GZ_DISCOVERY_MSG_PORT` = 5500 + 2 x `ROS_DOMAIN_ID`, `GZ_DISCOVERY_SRV_PORT` one above, below the DDS port blocks), set by `source arena`, by the `arena launch` / `arena env` supervisor for every process it spawns, and by `arena_runtime.launch.py` for everything it starts, so forks and lanes never receive each other's gz discovery. A shell that changes `ROS_DOMAIN_ID` after sourcing re-exports both before calling the `gz` CLI. `pose_to_tf` turns the bridged model pose into `odom -> base` and republishes a ros2_control robot's controller odometry on `odom` with that pose and those frames, so the pose follows teleports and the twist stays the wheel reading |
| `isaac` | `IsaacSimulator` | [`isaac_simulator.py`](isaac_simulator.py) | Isaac Sim integration |
| `mujoco` | `MujocoSimulator` | [`mujoco_simulator.py`](mujoco_simulator.py) | MuJoCo server over the `/mujoco` services, see [arena_mujoco](../../../../arena_mujoco/README.md) |

Flatland and Unity have stubs (commented out) in `__init__.py`; they are not
active. The active simulator is selected by `node.conf.Arena.SIM` and
instantiated via `SimulatorRegistry.get(key, **kwargs)`.

## flatsim backend

`sim:=flatsim` runs the [flatsim](../../../../flatsim/README.md) server, one
process for the whole runtime, started by `sim.launch.py` through
`arena feature flatsim launch`. It is an optional feature
(`arena feature flatsim install`): arena_runtime does not depend on it, and
selecting the sim without it fails with the install hint. The server owns
`/clock`, steps one world that holds every env at its realizer offset, as
Gazebo and Isaac do, and publishes each robot's
`map -> odom`, `odom -> base`, Odometry, zero joint states and sensor
messages. Its interface is documented in the flatsim README, the notes here
are what Arena sends it.

- [`FlatsimHost`](flatsim_simulator/flatsim_simulator.py): `pause` and `unpause` are
  `control_world` calls. `step_seconds` sends `multi_step` in chunks of at
  most one sim second, the server takes the steps before it answers, and the
  call returns once arena_runtime_node's own clock carries the returned time.
  `cleanup_namespace` deletes every entity, floor and robot whose name starts
  with the env prefix `env_<id>/`.
- [`FlatsimSimulator`](flatsim_simulator/flatsim_simulator.py): the per-env client. It
  keeps no world of its own. Spawns and deletes are service calls, pedestrians
  go out on `/flatsim/discs` and door and cabin moves on `/flatsim/poses`
  without waiting for an answer, as Gazebo's `move_box` does, since the
  mechanism moves every door on every tick.
- Geometry: wall segments and obstacle footprints are static outlines, sent in
  one call per spawn. Doors are boxes behind the four mechanism primitives,
  pedestrians are discs fed by `pedestrian_update`. Stairs steps and treads are
  boxes like any other, so a planar robot cannot climb them.
- Levels: floors define level slabs. A robot collides and casts rays only
  against the level containing its z. The shipped multi-level worlds lay their
  levels side by side in xy at one z, so an elevator ride there is an xy
  teleport within one level. An entity that ends within 0.05 m above its level
  floor, such as an elevator cabin floor, is no obstacle.
- Robots: body from the mobile cap footprint, driven by the Twist on
  `<robot>/cmd_vel` through a [drive model](flatsim_simulator/planar.py) read from
  the `planar` block of the robot's `caps/mobile.yaml`, the same keys for every
  robot: per-axis (linear, lateral, angular) velocity bounds, acceleration and
  gain, a first-order `lag`, a `traction` cap on planar acceleration, the
  `pivot` x the body turns about and the command `timeout`. A diff drive has
  lateral bounds `[0, 0]`. A robot without the block is unbounded and moves
  sideways only when `is_holonomic`. No controller_manager runs. The adapter
  launches `robot_state_publisher`, the server gets the URDF and the tf topic
  of the env.
- Odometry: the pose is the exact body pose. The twist is the wheel reading,
  the body twist divided by the gain and taken at the pivot, so it tracks the
  command like a wheel-odometry twist does while the IMU reports the body rate.
- `sim.flatsim.dynamics:=rigid|kinematic`: a runtime argument, the same for
  every env. `rigid` (default) lets walls and obstacles stop or push the robot
  while pedestrians touch it without pushing, `kinematic` lets it pass through.
  Collision events come from the env node's collision tracker in both.
- `sim.flatsim.physics:=auto|box2d|warp`: a runtime argument, the engine of the
  server. `box2d` steps on the CPU, `warp` runs physics and every ray cast on the
  GPU, `auto` takes `warp` when GPU passthrough is on, else `box2d`.
- Sensors: `laserscan`, `pointcloud`, `imu` and `contact` specs, each at its
  own update rate rounded to whole physics steps, and every sensor publishes
  once more when an episode reset ends (`republish`). Lidar and camera geometry
  comes from the URDF `<sensor>` block backing the spec (the mobile cap's
  `laser` block when there is none), the mount from the joint chain. Rays
  respect height: a beam hits an entity only where the entity's z extent
  holds it, so one cast per azimuth yields the scan and every ring of a 3D
  lidar cloud. A downward beam that meets nothing first returns its level
  floor, so a costmap raytraces through the voxels it crosses as it would on a
  real floor. Depth and RGB-D cameras yield a decimated cloud, images are not
  produced. The IMU reports yaw, yaw rate and the body acceleration of the
  last step plus gravity. Contacts list the entities the footprint touches.
- Lockstep: no channel is registered, so the other producers set the strides.
  A step is complete when the service answers, every message of it is already
  published.

## Adding a new simulator

1. Subclass `BaseSim`; implement all abstract methods from the four
   sub-interfaces. Define `SIM_NAME` (`ClassVar[str]`), consumed by
   `MechanismITF.__init__` to tag semantics with the simulator key.
   `before_reset_episode`/`after_reset_episode`/`step` default to no-ops,
   override only if the simulator needs episode-boundary or stepped-sim work.
2. Implement the four `MechanismITF` primitives (`spawn_box` with its
   `velocity` and `comb` keywords, `move_box`, `delete_box`, `set_robot_pose`)
   to get door, elevator and stair animation out of the box, override
   `move_boxes` when the simulator can move many boxes in one request, and call `_register_agent_robot` on spawn / `_forget_agent_robot`
   on remove so the door gate can see the simulator's robots. Override
   `spawn_doors`/`remove_doors`/`spawn_elevators`/`remove_elevators` only if
   the simulator has native door support that supersedes the shim, and call
   `await self.stop_mechanisms()` from `shutdown` if you use the defaults.
3. Register a lazy async factory:

```python
@SimulatorRegistry.register(SimSimulator.MY_SIM)
async def lazy_mysim(**kwargs: typing.Unpack[SimKwargs]) -> BaseSim:
    from .my_sim import MySimulator
    return await MySimulator.create(**kwargs)
```

4. Add `MY_SIM = "my_sim"` to `SimSimulator` in `arena_bringup/constants.py`.
