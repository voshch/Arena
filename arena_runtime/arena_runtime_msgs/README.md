# arena_runtime_msgs

rosidl interfaces for `arena_node` (the runtime). Env registry, lifecycle holds, world placement, namespace cleanup, env shutdown, env purge.

Per-env episode types (`EpisodeRecord`, `RunEpisode`, query/spawn services) live in [`task_generator_msgs`](../../utils/msgs/task_generator_msgs/README.md) instead.

## Services (`srv/`)

| File | Purpose |
|---|---|
| `RegisterEnv.srv` | Reserve an env_id, normalized namespace, and lease for an external environment. |
| `ClaimEnv.srv` | Atomically bind one launch instance to a reserved environment lease. |
| `SpawnEnv.srv` | Reserve and launch `task_generator.launch.py` as a child process, then wait for ACTIVE. |
| `DespawnEnv.srv` | Publish a `ShutdownRequest` asking an env to self-shutdown via lifecycle. |
| `ConfirmWorld.srv` | Run the shelf packer for an env's extent; returns `reference`, `slot_extent`, `prespawn`. |
| `LifecycleHold.srv` | Acquire or release a pause hold (ref-counted across callers). |
| `LifecycleUnpauseWindow.srv` | Acquire or release the exclusive unpause window. |
| `CleanupNamespace.srv` | Delete sim entities under a validated `env_<id>{_,/}` prefix. |

## Messages (`msg/`)

| File | Purpose |
|---|---|
| `EnvRecord.msg` | Single environment entry including its normalized FQN, lease, owner, placement, and liveness state. |
| `EnvRegistry.msg` | Snapshot of all non-draining `EnvRecord`s. Published latched on `state/envs`. |
| `Heartbeat.msg` | Lease-bound liveness ping from a `task_generator_node`. |
| `HoldEntry.msg` | One `(caller_id, reason, count)` triple. |
| `HoldRegistry.msg` | All active `HoldEntry`s. Published latched on `state/holders`. |
| `ShutdownRequest.msg` | Ownership-targeted request asking one environment instance to shut down. |
| `WorldExtent.msg` | World bounding box submitted with `ConfirmWorld`. |

## Actions (`action/`)

| File | Purpose |
|---|---|
| `PurgeEnv.action` | Delete entities under an arbitrary prefix and stream `purging` / `done` feedback. |
