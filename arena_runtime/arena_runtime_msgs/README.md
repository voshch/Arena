# arena_runtime_msgs

rosidl interfaces of arena_runtime: the runtime's (`arena_runtime_node`: env registry, lifecycle holds, world placement, env cleanup, env shutdown, lockstep), and every env node's (`arena_env_node`: episodes, robots, semantics, spawns, the episode queue and robot tasks, see [Env node interfaces](#env-node-interfaces)).

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
| `Heartbeat.msg` | Liveness ping from a managed env's `arena_env_node`. |
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

## Env node interfaces

Served by the env node of every env, whichever task runtime it runs.

rosidl interfaces consumed and published by the env node (the per-env episode loop): episode lifecycle, task-mode and config queries, per-env spawn/reset, robot fleet descriptors.

Audio frames and sound detections live in `arena_robots_msgs`. Types of the arena acoustics backend (sound sources, events and receptions, room impulses, runtime microphone services) live in `arena_auditory_msgs` (the `arena_auditory` submodule).

### Services (`srv/`)

| File | Purpose |
|---|---|
| `ResetEpisode.srv` | Advance to a new episode; accepts optional world and seed for replay. Resolves any in-flight `RunEpisode` goal with `Result.SKIPPED` (reason="reset"). |
| `QueueEpisode.srv` | Stage the next episode (modes, world, robots, per-mode params, human backend params). Applied at the next reset. `modes` names the task modes per axis, `params` their parameters by full name. |
| `GetTaskModes.srv` | Return the live task modes per axis in `modes`. |
| `QueryWorlds.srv` / `QueryScenarios.srv` / `QueryEnvironments.srv` / `QueryParametrizeds.srv` / `QueryRobots.srv` / `QueryStaticObstacles.srv` / `QueryDynamicObstacles.srv` / `QueryTaskModes.srv` | Listing of available shortnames for the corresponding asset class. |
| `SpawnStatic.srv` / `SpawnDynamic.srv` / `SpawnRobot.srv` | Inject a static obstacle / dynamic pedestrian / additional robot into the running episode via `TM_Obstacles.extend` / `TM_Robots.extend`. `SpawnRobot` accepts an optional `args` (`diagnostic_msgs/KeyValue[]`) forwarded to `Robot.parse` (e.g. `mobile`, `mobile.local_planner`, `mobile.agent`), and an `immediate` flag that provisions the robot into the live world now (idle) instead of committing on the next reset. |
| `SpawnSound.srv` / `RemoveSound.srv` | Spawn an environment sound of a kind (default asset, or `asset_id` and playback when `customize_playback`) at a pose via `runtime/spawn_sound`, optionally attached to the pose's frame, and remove it by the returned `entity` via `runtime/remove_sound`. |
| `MoveEntity.srv` | Move a runtime-spawned static obstacle or sound by the handle its spawn returned, or any robot by name, to a map-frame pose via `runtime/move`. The same path the rviz drag handles use. |
| `DespawnRobot.srv` | Single fleet-removal surface: stages a live robot for teardown on the next reset, un-stages a queued despawn, or cancels a queued spawn (toggles `state/robots/pending`). |
| `SetSemantic.srv` | Write one semantic field value on an entity via `semantics/set`; one of three writer paths into semantics state (timeline, modules, external). |
| `SubmitTask.srv` | Hand one robot a task via `runtime/submit_task`: `task_yaml` is a `phases:` list or a `{phases, conditions}` mapping in the scenario format, parsed by `TaskRequest.parse` and passed to `RobotManager.submit_task`, goto poses map-frame and shifted into scenario coordinates like rviz goals. Unfinished phases of the robot's previous request are dropped. An empty phases list only stops the robot: its unfinished phases are dropped, its in-flight action goal is cancelled and `phase_indices` is empty. Returns the episode-wide `phase_indices` the request was appended at, elevator legs included. |

### Messages (`msg/`)

| File | Purpose |
|---|---|
| `EpisodeRecord.msg` | One episode: id, world, seed, task modes (`modes` per axis), the selected modes' `params` under full names, `robots[]`, `outcome_state` (`QUEUED` / `RUNNING` / `SUCCESS` / `FAILED` / `SKIPPED` / `FATAL`), `outcome_info` (live status string, may be republished mid-episode via `EpisodeNode.set_episode_info`), `integrity` (why the episode is compromised, one reason per entry, empty when intact), `params` (the effective params of the selected modes under full names, the staged dict overlaid on a queued record) and `human_params` (namespaced human backend params the active backend accepted, same overlay). Published latched on `state/episode` and `state/queue`. `conditions` is a JSON list of `{op, p, q, text}` episode conditions, empty when none. `goal_dist_start` / `goal_dist_min` / `path_length` track the robot that closed the least of its start-to-goal distance, sampled every 0.5s. Zero when no GoTo goal was active. `phases` is a JSON map robot name -> {phases, conditions, map_poses, goal_inputs, instructions} of every phase submitted this episode, resolved, `map_poses` being the realized map-frame goto pose per phase, `goal_inputs` what the robot's planner receives (`pose`, `instruction`) and `instructions` the `{source, text}` it was told per dispatched phase (null without one). |
| `RobotDescriptor.msg` | Lean per-robot identity (name, model, ns, frame); shared with `RobotQueue`, whose pending entries have no resolved caps. |
| `RobotCap.msg` | One resolved, effective cap on a live robot: cap name, bound adapter kind, mount instance, morphology variant. |
| `RobotState.msg` | A resolved, live fleet member: `RobotDescriptor` + resolved `RobotCap[]` + resolved morphology `params`. |
| `RobotFleet.msg` | All currently-active `RobotState`s in the env. Published latched on `state/robots`. |
| `RobotQueue.msg` | Robots staged for spawn/despawn (the pending fleet delta, as lean `RobotDescriptor`s), applied on the next reset. Published latched on `state/robots/pending`. |
| `AdapterVizManifest.msg` | Per-env viz manifest: env-level `AdapterDisplay[]`, per-robot `AdapterEntry[]` and the rviz panels and tools backends contribute as `AdapterPlugin[]` (`role`, pluginlib `class_name`, panel `name`, `properties_json`). Published latched on `state/viz_manifest`. |
| `RecordedTopics.msg` | Topics the recorder subscribes to beyond its own, as `RecordedTopic[]` rows (`key`, `topic` template with `{ns}` and `{tg}`, `msg_type`, `robot_scoped`, `throttled`, `qos_transient_local`, `reliable`, `depth` (0 = the recorder default), `recorded`). Published latched (`TRANSIENT_LOCAL`, depth 1) on `state/recorded_topics`. |
| `SemanticSnapshot.msg` | Full latched semantic state of the env: stamp, world, `SemanticEntityState[]`. Published on `state/semantics` (`TRANSIENT_LOCAL`), republished on any quantum-passing change (attach/detach/reset/write). |
| `SemanticEntityState.msg` | One semantic entity: `kind`, index-aligned discrete/continuous/predicate name-value arrays, `members` (committed occupant ids). |

### Actions (`action/`)

| File | Purpose |
|---|---|
| `RunEpisode.action` | Single-flight episode runner: goal carries optional world; result `state` is one of `QUEUED` / `RUNNING` / `SUCCESS` / `FAILED` / `SKIPPED` / `FATAL` (FATAL = env never reached a runnable state, do not retry). A concurrent `ResetEpisode` resolves the in-flight goal with `SKIPPED`. |
| `RunTask.action` | Assign a task to one robot on `runtime/run_task`, the action form of `SubmitTask.srv`: feedback names each phase as it ends, the result is `met`, `failed` or `dropped`. A later goal for the same robot replaces the running one, cancel stops the robot. |
