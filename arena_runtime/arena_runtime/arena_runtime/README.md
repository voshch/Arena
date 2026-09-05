# arena: multi-env orchestration layer

`arena_node` ([`arena_node.py`](arena_node.py)) is the single process-level lifecycle
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
holder regardless of other holds, restoring pause state on release.

Invariant: evicting an env calls `release_all(fqn)` so stale holds never block
the sim permanently.

## `ArenaNode`: services and topics

Namespace prefix for all names below: `/arena/`.

### Services

| Service | Type | Purpose |
| --- | --- | --- |
| `register_env` | [`RegisterEnv.srv`](../../arena_runtime_msgs/srv/RegisterEnv.srv) | Reserve an ID+namespace; env calls this when it launches without managed mode |
| `spawn_env` | [`SpawnEnv.srv`](../../arena_runtime_msgs/srv/SpawnEnv.srv) | Reserve + launch `task_generator.launch.py` as a child process; waits for ACTIVE |
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

`ArenaNode` also subscribes per env to `/<ns>/transition_event` (to track
ACTIVE/INACTIVE/FINALIZED state) and `/<ns>/state/heartbeat` (to detect
stale envs). A periodic timer applies `sweep_verdict` per record: a managed env
whose wrapper process exited is evicted immediately, a pre-ready env (still
bootstrapping, e.g. loading its world) is evicted after `bootstrap_timeout_sec`
when one is set (`env.bootstrap_timeout:=<s>` at launch, default 0 = never),
measured from `reserve()` time rather than heartbeat silence, so an env that
keeps heartbeating while stuck in setup is still evicted. A ready env is evicted after
`heartbeat_timeout_sec` (default 5 s), stretched to `reset_hold_timeout_sec`
(default 30 s) while it holds a "reset" hold. Evicting a still-spawning env
fails its pending `spawn_env` call, since `dispose` cancels `spawn_ready`.

When `env_n > 0` at startup the node self-orchestrates an initial fleet via
`_spawn_initial_envs`; each spawned env has `headless=false` (stdout visible).

`state/sim` flips to `alive=false` after two consecutive lifecycle failures
(pause/unpause returning `False` or raising, a step raising, or a
`cleanup_namespace` raising `SimUnavailable`) and stays there: `arena_node`
never self-heals a dead sim, it just fails every pending `spawn_env` and stops
the lockstep scheduler. Restarting the runtime is the benchmark runner's job.

## Env lifecycle sequence

1. **Register**: env calls `register_env` (or is pre-reserved by `spawn_env`).
   Gets back `env_id`, `ns`, and the active `sim` name.
2. **Launch**: env starts `task_generator_node` under the allocated `ns`;
   `managed=true` skips re-registration.
3. **Confirm world**: once the env knows its world extent, it calls
   `confirm_world`; the shelf packer places the slot and returns `reference`.
   `reallocated=true` in the response means the env must translate all entity
   coordinates by the new `reference` offset.
4. **Heartbeat**: env publishes [`Heartbeat.msg`](../../arena_runtime_msgs/msg/Heartbeat.msg)
   on `<ns>/state/heartbeat`; `arena_node` resets the timeout clock on each tick.
   The clock starts at `reserve()` time, so an env is covered from registration.
5. **Despawn**: caller sends `despawn_env`; `arena_node` publishes a
   `ShutdownRequest` the env observes and acts on via its own lifecycle.
6. **Eviction**: on a `sweep_verdict` reason or lifecycle FINALIZED,
   `arena_node` calls `start_eviction` (marks draining, releases holds),
   publishes a `ShutdownRequest`, awaits env disposal (reaping the wrapper
   process for managed envs), and only then cleans up the namespace via
   `SimLifecycle` and calls `complete_eviction` (frees the ID for reuse).

## See also

- [task_generator](../../../task_generator/README.md): episode loop, task-mode registry, manager overview.
- [Managers](../../../task_generator/task_generator/manager/README.md): `RobotsManager`, `WorldManager`, `EnvironmentManager`, `Realizer`.
- [arena_runtime_msgs](../../arena_runtime_msgs): all message and service definitions.
