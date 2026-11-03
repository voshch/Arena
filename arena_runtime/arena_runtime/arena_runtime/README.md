# arena: multi-env orchestration layer

`arena_runtime_node` ([`arena_runtime_node.py`](arena_runtime_node.py)) is the single process-level lifecycle
node that lets multiple simulation environments coexist in one ROS graph. It owns
two primitives (`EnvRegistry`, `HoldRegistry`), serves the
services that envs and callers use to register and tear down, and publishes latched
state topics that everyone reads instead of polling.

Launched by `arena_runtime.launch.py` (namespace `/arena`, node name `arena`).

## The two primitives

### `EnvRegistry`

[`registry.py`](registry.py)

Allocates integer env IDs and translates world extents into non-overlapping
simulator slots. State per record: `env_id`, `fqn`, `placed`, `reference`,
`slot_extent`, `prespawn`, `ready`, `draining`, `last_heartbeat`, `reserved_at`.

Two-phase allocation:

1. `reserve()`: assigns an ID and namespace without touching spatial layout.
   Free IDs are recycled (lowest first), skipping any that are still draining.
2. `place()` / `confirm_world` service: runs a first-fit shelf packer over the
   requested `WorldExtent` padded by `slot_buffer` (default 5 m). Shelves grow
   along +x, new rows stack along +y, and the row width budget stays roughly
   square so the layout does not degenerate into a strip. Returns a `reference`
   offset (applied to all entity coordinates), a `prespawn` anchor in the buffer
   ring east of the bbox, and `slot_extent`. Idempotent on identical extent.

Removing an env (eviction or free) triggers a `_reflow` that rebuilds all shelf
positions for remaining placed records in `env_id` order, keeping the layout
compact after gaps appear.

Invariant: draining records are excluded from `snapshot()` and from ID recycling.

### `HoldRegistry`

[`holds.py`](holds.py)

Ref-counted pause gate. Each caller acquires holds keyed by `(caller_id, reason)`;
`total_count()` is the sum across all entries. When count transitions 0→1 the sim
is paused; when it transitions 1→0 the sim is unpaused (unless an unpause window
is active). Callers can also hold a one-at-a-time **unpause window** via
`LifecycleUnpauseWindow`: acquiring it force-unpauses the sim for the window
holder regardless of other holds, restoring pause state on release. While a
lockstep run is active the sim stays paused and the run steps it for the window.

Invariant: evicting an env calls `release_all(fqn)` so stale holds never block
the sim permanently.

## `ArenaRuntimeNode`: services and topics

Namespace prefix for all names below: `/arena/`.

### Services

| Service | Type | Purpose |
| --- | --- | --- |
| `register_env` | [`RegisterEnv.srv`](../../arena_runtime_msgs/srv/RegisterEnv.srv) | Reserve an ID+namespace; env calls this when it launches without managed mode |
| `spawn_env` | [`SpawnEnv.srv`](../../arena_runtime_msgs/srv/SpawnEnv.srv) | Reserve + launch the env launch file as a child process; waits for ACTIVE. The file is [`env.launch.py`](../launch/env.launch.py) of arena_runtime, started after the `arena_runtime.task_layer` entry point resolves ([task_layer.py](env/task_layer.py)) |
| `despawn_env` | [`DespawnEnv.srv`](../../arena_runtime_msgs/srv/DespawnEnv.srv) | Publish a `ShutdownRequest` asking an env to self-shutdown via lifecycle |
| `confirm_world` | [`ConfirmWorld.srv`](../../arena_runtime_msgs/srv/ConfirmWorld.srv) | Run the shelf packer for an env's extent; returns reference/slot/prespawn |
| `sim_lifecycle/hold` | [`LifecycleHold.srv`](../../arena_runtime_msgs/srv/LifecycleHold.srv) | Acquire or release a pause hold |
| `sim_lifecycle/unpause_window` | [`LifecycleUnpauseWindow.srv`](../../arena_runtime_msgs/srv/LifecycleUnpauseWindow.srv) | Acquire or release the exclusive unpause window |
| `cleanup_env` | [`CleanupEnv.srv`](../../arena_runtime_msgs/srv/CleanupEnv.srv) | Delete sim entities under an env's namespace prefix |

### Topics (all latched, `TRANSIENT_LOCAL`)

| Topic | Type | Content |
| --- | --- | --- |
| `state/paused` | `std_msgs/Bool` | `true` when `hold_count > 0` |
| `state/holders` | [`HoldRegistry.msg`](../../arena_runtime_msgs/msg/HoldRegistry.msg) | All active `(caller_id, reason, count)` entries |
| `state/envs` | [`EnvRegistry.msg`](../../arena_runtime_msgs/msg/EnvRegistry.msg) | Snapshot of all non-draining env records |
| `state/sim` | [`SimState.msg`](../../arena_runtime_msgs/msg/SimState.msg) | `alive`/`reason` for the sim lifecycle (`_lifecycle` pause/unpause/step), `header.stamp` is the wall time of the transition |
| `shutdown_request` | [`ShutdownRequest.msg`](../../arena_runtime_msgs/msg/ShutdownRequest.msg) | Broadcast asking `env_id` to shut down |

`ArenaRuntimeNode` also subscribes per env to `/<ns>/transition_event` (to track
ACTIVE/INACTIVE/FINALIZED state) and `/<ns>/state/heartbeat` (to detect
stale envs). A periodic timer applies `sweep_verdict` per record: a managed env
whose wrapper process exited is evicted immediately, a pre-ready env (still
bootstrapping, e.g. loading its world) is evicted after `bootstrap_timeout_sec`
when one is set (`env.bootstrap_timeout:=<s>` at launch, default 0 = never),
measured from `reserve()` time rather than heartbeat silence, so an env that
keeps heartbeating while stuck in setup is still evicted. A ready env is evicted after
`heartbeat_timeout_sec` (`env.heartbeat_timeout:=<s>` at launch, default 5 s), stretched to `reset_hold_timeout_sec`
(default 30 s) while it holds a "reset" hold. Evicting a still-spawning env
fails its pending `spawn_env` call, since `dispose` cancels `spawn_ready`.

When `env_n > 0` at startup the node self-orchestrates an initial fleet via
`_spawn_initial_envs`; each spawned env has `headless=false` (stdout visible).

`state/sim` flips to `alive=false` after two consecutive lifecycle failures
(pause/unpause returning `False` or raising, a step raising, or a
`cleanup_namespace` raising `SimUnavailable`) and stays there: `arena_runtime_node`
never self-heals a dead sim, it just fails every pending `spawn_env` and stops
the lockstep scheduler. Restarting the runtime is the benchmark runner's job.

## Env lifecycle sequence

1. **Register**: env calls `register_env` (or is pre-reserved by `spawn_env`).
   Gets back `env_id`, `ns`, and the active `sim` name.
2. **Launch**: env starts `arena_env_node` under the allocated `ns`;
   `managed=true` skips re-registration.
3. **Confirm world**: once the env has rendered its world map, it calls
   `confirm_world` with the map bounds, which cover every level as laid out
   side by side; the shelf packer places the slot and returns `reference`.
   `reallocated=true` in the response means the env must translate all entity
   coordinates by the new `reference` offset and republishes its latched
   `map -> env_<n>/map` transform with it.
4. **Heartbeat**: env publishes [`Heartbeat.msg`](../../arena_runtime_msgs/msg/Heartbeat.msg)
   on `<ns>/state/heartbeat`; `arena_runtime_node` resets the timeout clock on each tick.
   The clock starts at `reserve()` time, so an env is covered from registration.
   The env beats from its own thread, so the beat shows a live process, not a responsive executor.
5. **Despawn**: caller sends `despawn_env`; `arena_runtime_node` publishes a
   `ShutdownRequest` the env observes and acts on via its own lifecycle.
6. **Eviction**: on a `sweep_verdict` reason or lifecycle FINALIZED,
   `arena_runtime_node` calls `start_eviction` (marks draining, releases holds),
   logs a warning with the reason and the age of the last heartbeat,
   publishes a `ShutdownRequest`, awaits env disposal (reaping the wrapper
   process for managed envs), and only then cleans up the namespace via
   `SimLifecycle` and calls `complete_eviction` (frees the ID for reuse).

## MCP domain `runtime`

[mcp.py](mcp.py) contributes the `runtime` domain to the Arena MCP server: the env table and env spawn, despawn and cleanup, sim health, pause holds, stepping, unpause windows and the lockstep scheduler, all through the services and latched topics above. `python3 -m arena_mcp ls runtime` lists its tools, `arena://runtime/state` and `arena://runtime/envs` serve the state and the env table as resources.

- `runtime_state` answers whether a runtime is up at all (`running: false` when `/arena` publishes no `state/sim`) and whether its sim is alive. Sim death is terminal: spawns are refused until the runtime restarts.
- One runtime drives one simulator, so a hold, a step, an unpause window or a lockstep run pauses or advances every env at once.
- The server's holds are the entries of `state/holders` under its node's `caller_id` (`/arena_mcp_<pid>`), counted per reason and released on request or when the server exits. Releasing never touches another caller's holds.
- `runtime_spawn_env` waits for the env to reach ACTIVE, or with `wait_ready: false` returns a job (see the `jobs` domain) that finishes once it does. The env's output lands in the returned `log_path`.
- Lockstep follows `arena lockstep`: `runtime_lockstep_start` with `seconds` is `run for:=` (a job that pauses the run when it ends or is cancelled), without it is `on`. `runtime_lockstep_gate` runs `arena lockstep gate` with the given presets or raw `name|topic|type|period_s|role` specs, an empty list clears the `cli` registration.

## MCP domain `entities`

[env/entities_mcp.py](env/entities_mcp.py) contributes the `entities` domain, a client of the node of one env: spawn
static and dynamic obstacles and robots, despawn robots, move entities,
semantics, runtime sounds (the `sounds` module serves them), the map server,
pedestrians (`arena_peds`, `human/move`, `human/stream`) and, when the env
runs `human:=arena`, the humansim behavior services (waypoints, flow,
sources and sinks, stimuli, agents, profile). `python3 -m arena_mcp ls entities` lists its tools.

## `static_tf.py`

[`static_tf.py`](static_tf.py)

`StaticTransformBroadcaster`: the node's latched `/tf_static` publisher. It
keeps one transform per child frame and a resend replaces it, so the
`map -> env_<n>/map` anchor follows a world swap (the tf2_ros broadcaster
keeps the first transform it sent for a child frame).

## See also

- [Simulator interface](sim/README.md): `BaseSim` and its adapters.
- [Human simulators](humans/README.md): pedestrian backends and their task-mode prompts.
- [Human launch axis](../launch/human/README.md), [acoustics launch axis](../launch/acoustics/README.md) and [hearing launch axis](../launch/hearing/README.md): the launch trees per axis.
- [Constants and parameters](env/constants/README.md): `Constants` and `Configuration`.
- [Robot adapters](env/adapters/README.md): per-cap adapters.
- [task_generator](../../../task_generator/README.md): task-mode registry.
- [Env layer](env/README.md): `EnvNode`, `EpisodeNode`, episode loop, `episode` MCP domain, `RobotsManager`, `WorldManager`, `EnvironmentManager`, `Realizer`.
- [arena_runtime_msgs](../../arena_runtime_msgs): all message and service definitions.
