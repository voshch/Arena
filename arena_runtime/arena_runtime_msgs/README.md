# arena_runtime_msgs

rosidl interfaces for `arena_node` (the runtime). Env registry, lifecycle holds, world placement, namespace cleanup, env shutdown, env purge.

Per-env episode types (`EpisodeRecord`, `RunEpisode`, query/spawn services) live in [`task_generator_msgs`](../../utils/msgs/task_generator_msgs/README.md) instead.

## Services (`srv/`)

| File | Purpose |
|---|---|
| `RegisterEnv.srv` | Reserve an env_id and namespace; called by an env that launches without managed mode. |
| `SpawnEnv.srv` | Reserve + launch `task_generator.launch.py` as a child process; waits for ACTIVE. |
| `DespawnEnv.srv` | Publish a `ShutdownRequest` asking an env to self-shutdown via lifecycle. |
| `ConfirmWorld.srv` | Run the shelf packer for an env's extent; returns `reference`, `slot_extent`, `prespawn`. |
| `LifecycleHold.srv` | Acquire or release a pause hold (ref-counted across callers). |
| `LifecycleUnpauseWindow.srv` | Acquire or release the exclusive unpause window. |
| `LifecycleStep.srv` | Advance the held sim by an exact sim-time delta (quantized to physics steps). |
| `LockstepRegister.srv` | Register (or replace, or clear) a caller's set of lockstep data channels. |
| `LockstepStart.srv` | Start or reconfigure the lockstep scheduler, paced to a target RTF, gated or ungated. |
| `CleanupNamespace.srv` | Delete sim entities under a validated `env_<id>{_,/}` prefix. |

## Messages (`msg/`)

| File | Purpose |
|---|---|
| `EnvRecord.msg` | Single env entry: `env_id`, `fqn`, `placed`, `reference`, `slot_extent`, `extent`, `draining`. |
| `EnvRegistry.msg` | Snapshot of all non-draining `EnvRecord`s. Published latched on `state/envs`. |
| `Heartbeat.msg` | Liveness ping from a managed env's `task_generator_node`. |
| `HoldEntry.msg` | One `(caller_id, reason, count)` triple. |
| `HoldRegistry.msg` | All active `HoldEntry`s. Published latched on `state/holders`. |
| `LockstepChannel.msg` | One registered data channel: `name`, `topic`, `type`, `period_s`, `hard`. |
| `LockstepHeartbeat.msg` | Coverage stamp republished by producers with no natively stamped output (task_server beats). |
| `LockstepRegistration.msg` | One caller's channel set: `caller`, `env`, `channels[]` (`LockstepChannel`). |
| `LockstepStatus.msg` | Scheduler state: run flags, target/measured RTF, gated tick, registrations, stall lists. Published latched on `state/lockstep`. |
| `ShutdownRequest.msg` | Broadcast asking `env_id` to shut down (reason carried as a string). |
| `WorldExtent.msg` | World bounding box submitted with `ConfirmWorld`. |

## Actions (`action/`)

| File | Purpose |
|---|---|
| `PurgeEnv.action` | Delete entities under an arbitrary prefix and stream `purging` / `done` feedback. |
