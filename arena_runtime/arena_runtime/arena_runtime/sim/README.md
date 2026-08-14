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

## Sim-paused invariant

The sim is paused for the entire body of `Task._reset_episode`. Only
node-discovery and lifecycle signals are observable while the sim is paused;
tf, costmap, and sim-clock topics are not advancing.

## Registered implementations

[`__init__.py:57`](__init__.py#L57): `SimulatorRegistry` maps
`Constants.SimSimulator` keys to async factory functions:

| Key | Class | File | Notes |
| --- | --- | --- | --- |
| `dummy` | `DummySimulator` | [`dummy_simulator.py`](dummy_simulator.py) | no-op; publishes a synthetic `/clock` for testing |
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
