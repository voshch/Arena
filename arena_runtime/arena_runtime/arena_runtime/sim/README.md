# task_generator sim interface

`BaseSim` and the four sub-interfaces it combines. Implementations are
registered in `SimulatorRegistry` and instantiated by key at runtime. See also
[human simulator](../../../../task_generator/task_generator/simulators/human/README.md) for the pedestrian-logic counterpart
(`BaseHumanSimulator`).

## Sub-interfaces

Defined in [`_interface.py`](_interface.py):

### `ObstacleITF`

[`_interface.py:20`](_interface.py#L20)

| Abstract method | Signature |
| --- | --- |
| `obstacle_spawn` | `(Sequence[Obstacle]) -> Sequence[bool]` |
| `obstacle_move` | `(Sequence[Obstacle]) -> Sequence[bool]` |
| `obstacle_delete` | `(Sequence[Obstacle]) -> Sequence[bool]` |

### `PedestrianITF`

[`_interface.py:38`](_interface.py#L38)

| Abstract method | Signature |
| --- | --- |
| `pedestrian_spawn` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_move` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_delete` | `(Sequence[DynamicObstacle]) -> Sequence[bool]` |
| `pedestrian_update` | `(Pedestrians) -> Sequence[bool]` |

### `RobotITF`

[`_interface.py:61`](_interface.py#L61)

| Abstract method | Signature |
| --- | --- |
| `robot_spawn` | `(Sequence[Robot]) -> Sequence[bool]` |
| `robot_move` | `(Sequence[Robot]) -> Sequence[bool]` |
| `robot_delete` | `(Sequence[Robot]) -> Sequence[bool]` |

### `WorldITF`

[`_interface.py:116`](_interface.py#L116)

| Abstract method | Signature |
| --- | --- |
| `spawn_walls` | `(Sequence[Wall]) -> bool` |
| `spawn_floors` | `(Sequence[Floor]) -> bool` |
| `remove_world` | `() -> bool` (default raises `NotImplementedError`) |

### `MechanismITF`

[`_interface.py:138`](_interface.py#L138): door and elevator orchestration. Provides concrete default implementations driven by an internal sim-time tick loop ([`_mechanism_shim.py`](_mechanism_shim.py)) on top of four box/robot primitives. Any simulator that implements the primitives gets door animation, elevator pair-teleport, and ped/robot teleport for free; simulators with native support can override the four top-level methods.

| Default method | Signature | Purpose |
| --- | --- | --- |
| `spawn_doors` | `(Sequence[Door]) -> bool` | spawn door geometry + register runtime; starts the tick loop |
| `remove_doors` | `(Sequence[str]) -> bool` | delete door geometry by name |
| `spawn_elevators` | `(Sequence[Elevator]) -> bool` | spawn cabin walls + synthesized door per elevator |
| `remove_elevators` | `(Sequence[str]) -> bool` | delete cabin geometry + synthesized door by name |
| `stop_mechanisms` | `() -> None` | cancel the tick loop (call on shutdown) |
| `attach_human_simulator` | `(HumanSimulator) -> None` | bind the human-sim Protocol that supplies ped positions + ped teleport |

| Primitive (override required for default behavior) | Signature |
| --- | --- |
| `spawn_box` | `(name, size, pose) -> bool` |
| `move_box` | `(name, pose) -> bool` |
| `delete_box` | `(name) -> bool` |
| `set_robot_pose` | `(sim_path, pose) -> bool` |

Robot tracking is not a primitive to override: call `_register_agent_robot(robot, model_params)` on spawn and `_forget_agent_robot(sim_path)` on remove, and the base class serves `robot_discs()` / `robot_pose()` from TF.

The `HumanSimulator` Protocol the shim consumes from the attached human-sim:

| Method | Signature |
| --- | --- |
| `pedestrian_discs` | `() -> Iterable[tuple[str, tuple[float, float], float]]` (sync, ground truth: name, xy, radius) |
| `pedestrian_teleport` | `(Mapping[str, tuple[float, float]]) -> bool` (async) |

The attachment happens in `BaseHumanSimulator.__init__` (see [human simulator](../../../../task_generator/task_generator/simulators/human/README.md)). Without an attached human-sim, the shim still drives doors against `robot_discs` alone and logs a no-op for ped teleports.

### `SimLifecycle`

[`_interface.py:18`](_interface.py#L18): process-singleton host counterpart to `BaseSim`. Owned by `arena_node` (one instance per process), drives sim-wide pause/unpause and namespace cleanup that cuts across env-scoped simulators.

| Abstract method | Purpose |
| --- | --- |
| `pause` / `unpause` | toggle the underlying sim clock |
| `cleanup_namespace(prefix)` | delete all entities under `prefix`, return count removed |
| `ensure_ready` | block until the sim's services are reachable |

Registered in `LifecycleRegistry` alongside `SimulatorRegistry` ([`__init__.py`](__init__.py)).

`arena_node` wraps the registered `SimLifecycle` in a liveness-accounting
proxy: two consecutive `pause`/`unpause` failures (`False` or raised) or step
failures mark the sim dead, published latched on `state/sim`
([`SimState.msg`](../../../arena_runtime_msgs/msg/SimState.msg)), see the
[arena_runtime README](../README.md) topic table.
Adapters signal a dead transport (a timed-out service call, not a genuine
rejection) by raising [`SimUnavailable`](_interface.py) instead of returning
`False`. `cleanup_namespace` should let it propagate. `arena_node` never
retries or recovers a dead sim on its own.

## `BaseSim`

[`__init__.py:17`](__init__.py#L17)

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
(both fields in the same message, else gz free-runs). Isaac calls
`/isaac/StepSimulationN` and awaits `/clock` reaching the projected target.
The dummy host adds the delta to its synthetic clock, so `sim:=dummy` runs the
whole control plane without a physics engine. That is what
[`tests/ros/`](../../tests/ros/) exercises, the gate math itself is
[`lockstep_gates.py`](../lockstep_gates.py) and has ROS-free unit tests.

`arena_node` exposes it as `sim_lifecycle/step` (valid only while holds are
active and no unpause window is open), plus the lockstep control plane
driving the [`LockstepScheduler`](../lockstep.py).

Components self-register their data channels via
`sim_lifecycle/lockstep/register` (LockstepRegister: `caller`, `env`,
`channels[]` of `{name, topic, type, period_s, hard}`. Re-registering under
a `caller` replaces its set, an empty `channels[]` clears it, per-env
registrations drop when their env despawns, and registrations merge into a
running lockstep live. The arena_humansim adapter self-registers `engine`
and `peds` (both hard). The hunav adapter self-registers `roster` (hard).
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
keep fire-and-forget goto_pose and stay ungated. `arena cam ... lockstep=true` recording registers `cam`
(hard, 1/fps) for the take when a lockstep run is active, so the recording
is gated like any other producer and frame-exact at any target rtf. With no
run active it holds and steps the sim itself. Beats republish
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
only pick the step sizes the scheduler takes. A producer driven by the sim
clock registers `period_s` equal to its own tick length, which paces the
sim to the producer's true coverage. An elapsed-based rate loop registers
a period strictly above its interval. The arena_humansim engine ticks off
`/clock` itself rather than an rcl timer, so a clock the gate is holding
still owes it the tick the gate is waiting for.

`sim_lifecycle/lockstep/start` (LockstepStart: `target_rtf`, `ungated`)
starts a run or reconfigures one in place: `target_rtf` of 0 is unpaced,
`ungated` free-steps ignoring every gate. Hard channels freeze-and-wait: the
scheduler will not advance past a tick until every hard channel has a
`header.stamp` covering it, so the weakest hard producer sets the pace by
design. `soft` channels are observed and reported only. The scheduler also
idles while a foreign hold or an unpause window is open (episode resets)
and rebases its window grid to whatever sim time passed meanwhile.
`sim_lifecycle/lockstep/stop` ends the run and restores the pre-run
pause state and gz `real_time_factor`, responding as soon as the hold is
released (the guard against a draining gz chunk re-pausing the world runs
in the background). `sim_lifecycle/lockstep/pause` /
`sim_lifecycle/lockstep/resume` freeze and unfreeze the clock within a run
without ending it. Status is the latched `/arena/state/lockstep` topic
(LockstepStatus): `active`, `paused`, `ungated`, `target_rtf`,
`measured_rtf`, `tick`, `registrations`, `waiting_on`, `arrived`.

`lockstep.autostart`/`lockstep.channels`/`lockstep.target_rtf`/
`lockstep.paused` params are read once at bringup (launch args
`lockstep:=true lockstep.channels:="..."`), registering the channels under
caller `launch`. Channel specs are `name|topic|type|period_s|hard-or-soft`
with `{env}` expanded per env. CLI: `arena lockstep run [rtf:=N] [ungated]`
(blocks, ctrl-c pauses), `on [rtf:=N] [ungated]`, `off`, `pause`, `resume`,
`status`, and `arena lockstep gate [engine|peds|<spec> ...]` to register
extra channels under caller `cli`.

## Sim-paused invariant

The sim is paused for the entire body of `Task._reset_episode`. Only
node-discovery and lifecycle signals are observable while the sim is paused;
tf, costmap, and sim-clock topics are not advancing. `switch_controller` is the
one controller_manager call that needs the sim to tick (on both gz_ros2_control
and Isaac's `ros2_control_node`, whose update loops follow sim time), so
controller activation lives inside the robot bring-up unpause window.

## Registered implementations

[`__init__.py:57`](__init__.py#L57): `SimulatorRegistry` maps
`Constants.SimSimulator` keys to async factory functions:

| Key | Class | File | Notes |
| --- | --- | --- | --- |
| `dummy` | `DummySimulator` | [`dummy_simulator.py`](dummy_simulator.py) | no-op entity verbs; `DummyHost` publishes a synthetic `/clock` (wall time minus paused time plus stepped time, also while paused) and supports `step_seconds` |
| `gazebo` | `GazeboSimulator` | [`gazebo_simulator/gazebo_simulator.py`](gazebo_simulator/gazebo_simulator.py) | Gazebo (Ignition) via gz-transport |
| `isaac` | `IsaacSimulator` | [`isaac_simulator.py`](isaac_simulator.py) | Isaac Sim integration |

Flatland and Unity have stubs (commented out) in `__init__.py`; they are not
active. The active simulator is selected by `node.conf.Arena.SIM` and
instantiated via `SimulatorRegistry.get(key, **kwargs)`.

## Adding a new simulator

1. Subclass `BaseSim`; implement all abstract methods from the four
   sub-interfaces. Define `SIM_NAME` (`ClassVar[str]`), consumed by
   `MechanismITF.__init__` to tag semantics with the simulator key.
   `before_reset_episode`/`after_reset_episode`/`step` default to no-ops,
   override only if the simulator needs episode-boundary or stepped-sim work.
2. Implement the four `MechanismITF` primitives (`spawn_box`, `move_box`,
   `delete_box`, `set_robot_pose`) to get door + elevator animation out of
   the box, and call `_register_agent_robot` on spawn / `_forget_agent_robot`
   on remove so the door gate can see the simulator's robots. Override
   `spawn_doors`/`remove_doors`/`spawn_elevators`/`remove_elevators` only if
   the simulator has native door support that supersedes the shim, and call
   `await self.stop_mechanisms()` from `shutdown` if you use the defaults.
3. Register a lazy async factory:

```python
@SimulatorRegistry.register(Constants.SimSimulator.MY_SIM)
async def lazy_mysim(**kwargs):
    from .my_sim import MySimulator
    return await MySimulator.create(**kwargs)
```

4. Add `MY_SIM = "my_sim"` to `Constants.SimSimulator`.
