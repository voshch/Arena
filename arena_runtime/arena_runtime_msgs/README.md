# arena_runtime_msgs

rosidl interfaces for `arena_node` (the runtime). Env registry, lifecycle holds, world placement, env cleanup, env shutdown, lockstep.

Per-env episode types (`EpisodeRecord`, `RunEpisode`, query/spawn services) live in [`task_generator_msgs`](../../utils/msgs/task_generator_msgs/README.md) instead.

## Services (`srv/`)

| File | Purpose |
|---|---|
| `RegisterEnv.srv` | Reserve an env_id and namespace; called by an env that launches without managed mode. |
| `SpawnEnv.srv` | Reserve + launch arena_runtime's `env.launch.py` as a child process; waits for ACTIVE. |
| `DespawnEnv.srv` | Publish a `ShutdownRequest` asking an env to self-shutdown via lifecycle. |
| `ConfirmWorld.srv` | Run the shelf packer for an env's extent; returns `reference`, `slot_extent`, `prespawn`. |
| `LifecycleHold.srv` | Acquire or release a pause hold (ref-counted across callers). |
| `LifecycleUnpauseWindow.srv` | Acquire or release the exclusive unpause window. |
| `LifecycleStep.srv` | Advance the held sim by an exact sim-time delta (quantized to physics steps). |
| `LockstepRegister.srv` | Register (or replace, or clear) a caller's set of lockstep data channels. |
| `LockstepStart.srv` | Start or reconfigure the lockstep scheduler, paced to a target RTF, gated or ungated. |
| `CleanupEnv.srv` | Delete the sim entities of `env_id`, under the prefix the active simulator derives from it. |

## Messages (`msg/`)

| File | Purpose |
|---|---|
| `EnvRecord.msg` | Single env entry: `env_id`, `fqn`, `extent`, `reference`, `slot_extent`, `prespawn`, `placed`, `ready`, `last_heartbeat`. |
| `EnvRegistry.msg` | Snapshot of all non-draining `EnvRecord`s. Published latched on `state/envs`. |
| `Heartbeat.msg` | Liveness ping from a managed env's `task_generator_node`. |
| `HoldEntry.msg` | One `(caller_id, reason, count)` triple. |
| `HoldRegistry.msg` | All active `HoldEntry`s. Published latched on `state/holders`. |
| `LockstepChannel.msg` | One registered data channel: `name`, `topic`, `type`, `period_s`, `hard`. |
| `LockstepChannelCount.msg` | Per-channel counters of one scheduler run: `name`, `topic`, ticks `due`, messages `received`. |
| `LockstepHeartbeat.msg` | Coverage stamp republished by producers with no natively stamped output (task_server beats). |
| `LockstepRegistration.msg` | One caller's channel set: `caller`, `env`, `channels[]` (`LockstepChannel`). |
| `LockstepStatus.msg` | Scheduler state: run flags, target/measured RTF, gated tick, registrations, stall lists, gated tick count and per-channel counters (`LockstepChannelCount`), both cumulative per run. Published latched on `state/lockstep`. |
| `ShutdownRequest.msg` | Broadcast asking `env_id` to shut down (reason carried as a string). |
| `SimState.msg` | Sim lifecycle liveness: `alive`, `reason`, wall-time `header.stamp` of the transition. Published latched on `state/sim`. |
| `WorldExtent.msg` | World bounding box submitted with `ConfirmWorld`. |
