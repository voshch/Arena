# task_generator_msgs

rosidl interfaces consumed and published by `task_generator` (the per-env episode loop). Episode lifecycle, task-mode and config queries, per-env spawn/reset, robot fleet descriptors.

Runtime types (env registry, holds, world confirm, cleanup, purge) live in [`arena_runtime_msgs`](../../../arena_runtime/arena_runtime_msgs/README.md) instead.

## Services (`srv/`)

| File | Purpose |
|---|---|
| `ResetEpisode.srv` | Advance to a new episode; accepts optional world and seed for replay. Resolves any in-flight `RunEpisode` goal with `Result.SKIPPED` (reason="reset"). |
| `QueueEpisode.srv` | Stage the next episode (modes, world, robots, per-mode params); applied at the next reset. |
| `Pause.srv` | Toggle pause from external callers. |
| `GetTaskModes.srv` | Return currently active task-mode strings. |
| `QueryWorlds.srv` / `QueryScenarios.srv` / `QueryEnvironments.srv` / `QueryParametrizeds.srv` / `QueryRobots.srv` / `QueryStaticObstacles.srv` / `QueryDynamicObstacles.srv` / `QueryTaskModes.srv` | Listing of available shortnames for the corresponding asset class. |
| `SpawnStatic.srv` / `SpawnDynamic.srv` / `SpawnRobot.srv` | Inject a static obstacle / dynamic pedestrian / additional robot into the running episode via `TM_Obstacles.extend` / `TM_Robots.extend`. `SpawnRobot` accepts an optional `args` (`diagnostic_msgs/KeyValue[]`) forwarded to `Robot.parse` (e.g. `mobile`, `mobile.local_planner`, `mobile.agent`), and an `immediate` flag that provisions the robot into the live world now (idle) instead of committing on the next reset. |
| `SpawnMicrophone.srv` | Place an episode-local acoustic listener at a stamped point. The auditory runtime derives and validates its authored world zone and returns its stable listener ID. |
| `SetAudioSystem.srv` | Start or stop one scenario- or launch-defined radio or multi-speaker alarm system. |
| `DespawnRobot.srv` | Single fleet-removal surface: stages a live robot for teardown on the next reset, un-stages a queued despawn, or cancels a queued spawn (toggles `state/robots/pending`). |
| `SetSemantic.srv` | Write one semantic field value on an entity via `semantics/set`; one of three writer paths into semantics state (timeline, modules, external). |

## Messages (`msg/`)

| File | Purpose |
|---|---|
| `EpisodeRecord.msg` | One episode: id, world, seed, task modes, `robots[]`, `outcome_state` (`QUEUED` / `RUNNING` / `SUCCESS` / `FAILED` / `SKIPPED` / `FATAL`), `outcome_info` (live status string, may be republished mid-episode via `Task.set_info`), integrity flag, plus `obstacles_params` / `robots_params` (effective per-mode params, with staged dict overlay for queued records). Published latched on `state/episode` and `state/queue`. `conditions` is a JSON list of `{op, p, q, text}` episode conditions, empty when none. |
| `RobotDescriptor.msg` | Lean per-robot identity (name, model, ns, frame); shared with `RobotQueue`, whose pending entries have no resolved caps. |
| `RobotCap.msg` | One resolved, effective cap on a live robot: cap name, bound adapter kind, mount instance, morphology variant. |
| `RobotState.msg` | A resolved, live fleet member: `RobotDescriptor` + resolved `RobotCap[]` + resolved morphology `params`. |
| `RobotFleet.msg` | All currently-active `RobotState`s in the env. Published latched on `state/robots`. |
| `RobotQueue.msg` | Robots staged for spawn/despawn (the pending fleet delta, as lean `RobotDescriptor`s), applied on the next reset. Published latched on `state/robots/pending`. |
| `SemanticSnapshot.msg` | Full latched semantic state of the env: stamp, world, `SemanticEntityState[]`. Published on `state/semantics` (`TRANSIENT_LOCAL`), republished on any quantum-passing change (attach/detach/reset/write). |
| `SemanticEntityState.msg` | One semantic entity: `kind`, index-aligned discrete/continuous/predicate name-value arrays, `members` (committed occupant ids). |
| `ContinuousAudioSourceState.msg` | Persistent source state for robot drivetrains and scenario-defined WAV emitters. Environment fields identify the logical system, asset, loop behavior, and shared program epoch. |
| `ContinuousHeardSoundState.msg` | Listener-specific propagation result for a persistent source, including its route, delay, received level, and environment playback metadata. |
| `AudioSystemState.msg` | Transient-local active state and emitter membership for one scenario- or launch-defined radio or alarm system. An empty emitter list removes a system from live controls. |

## Actions (`action/`)

| File | Purpose |
|---|---|
| `RunEpisode.action` | Single-flight episode runner: goal carries optional world; result `state` is one of `QUEUED` / `RUNNING` / `SUCCESS` / `FAILED` / `SKIPPED` / `FATAL` (FATAL = env never reached a runnable state, do not retry). A concurrent `ResetEpisode` resolves the in-flight goal with `SKIPPED`. |
