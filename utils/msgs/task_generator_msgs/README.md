# task_generator_msgs

rosidl interfaces consumed and published by `task_generator` (the per-env episode loop). Episode lifecycle, task-mode and config queries, per-env spawn/reset, robot fleet descriptors.

Runtime types (env registry, holds, world confirm, cleanup, purge) live in [`arena_runtime_msgs`](../../../arena_runtime/arena_runtime_msgs/README.md) instead.

Audio frames and sound detections live in `arena_robots_msgs`. Types of the arena acoustics backend (sound sources, events and receptions, room impulses, runtime microphone services) live in `arena_auditory_msgs` (the `arena_auditory` submodule).

## Services (`srv/`)

| File | Purpose |
|---|---|
| `ResetEpisode.srv` | Advance to a new episode; accepts optional world and seed for replay. Resolves any in-flight `RunEpisode` goal with `Result.SKIPPED` (reason="reset"). |
| `QueueEpisode.srv` | Stage the next episode (modes, world, robots, per-mode params, human backend params). Applied at the next reset. |
| `Pause.srv` | Toggle pause from external callers. |
| `GetTaskModes.srv` | Return currently active task-mode strings. |
| `QueryWorlds.srv` / `QueryScenarios.srv` / `QueryEnvironments.srv` / `QueryParametrizeds.srv` / `QueryRobots.srv` / `QueryStaticObstacles.srv` / `QueryDynamicObstacles.srv` / `QueryTaskModes.srv` | Listing of available shortnames for the corresponding asset class. |
| `SpawnStatic.srv` / `SpawnDynamic.srv` / `SpawnRobot.srv` | Inject a static obstacle / dynamic pedestrian / additional robot into the running episode via `TM_Obstacles.extend` / `TM_Robots.extend`. `SpawnRobot` accepts an optional `args` (`diagnostic_msgs/KeyValue[]`) forwarded to `Robot.parse` (e.g. `mobile`, `mobile.local_planner`, `mobile.agent`), and an `immediate` flag that provisions the robot into the live world now (idle) instead of committing on the next reset. |
| `SpawnSound.srv` / `RemoveSound.srv` | Spawn an environment sound of a kind (default asset, or `asset_id` and playback when `customize_playback`) at a pose via `runtime/spawn_sound`, optionally attached to the pose's frame, and remove it by the returned `entity` via `runtime/remove_sound`. |
| `MoveEntity.srv` | Move a runtime-spawned static obstacle or sound by the handle its spawn returned, or any robot by name, to a map-frame pose via `runtime/move`. The same path the rviz drag handles use. |
| `DespawnRobot.srv` | Single fleet-removal surface: stages a live robot for teardown on the next reset, un-stages a queued despawn, or cancels a queued spawn (toggles `state/robots/pending`). |
| `SetSemantic.srv` | Write one semantic field value on an entity via `semantics/set`; one of three writer paths into semantics state (timeline, modules, external). |

## Messages (`msg/`)

| File | Purpose |
|---|---|
| `EpisodeRecord.msg` | One episode: id, world, seed, task modes, `robots[]`, `outcome_state` (`QUEUED` / `RUNNING` / `SUCCESS` / `FAILED` / `SKIPPED` / `FATAL`), `outcome_info` (live status string, may be republished mid-episode via `Task.set_info`), integrity flag, plus `obstacles_params` / `robots_params` (effective per-mode params, with staged dict overlay for queued records) and `human_params` (namespaced human backend params the active backend accepted, same overlay). Published latched on `state/episode` and `state/queue`. `conditions` is a JSON list of `{op, p, q, text}` episode conditions, empty when none. `goal_dist_start` / `goal_dist_min` / `path_length` track the robot that closed the least of its start-to-goal distance, sampled every 0.5s. Zero when no GoTo goal was active. `phases` is a JSON map robot name -> {phases, conditions, map_poses, goal_inputs, instructions} of every phase submitted this episode, resolved, `map_poses` being the realized map-frame goto pose per phase, `goal_inputs` what the robot's planner receives (`pose`, `instruction`) and `instructions` the `{source, text}` it was told per dispatched phase (null without one). |
| `RobotDescriptor.msg` | Lean per-robot identity (name, model, ns, frame); shared with `RobotQueue`, whose pending entries have no resolved caps. |
| `RobotCap.msg` | One resolved, effective cap on a live robot: cap name, bound adapter kind, mount instance, morphology variant. |
| `RobotState.msg` | A resolved, live fleet member: `RobotDescriptor` + resolved `RobotCap[]` + resolved morphology `params`. |
| `RobotFleet.msg` | All currently-active `RobotState`s in the env. Published latched on `state/robots`. |
| `RobotQueue.msg` | Robots staged for spawn/despawn (the pending fleet delta, as lean `RobotDescriptor`s), applied on the next reset. Published latched on `state/robots/pending`. |
| `AdapterVizManifest.msg` | Per-env viz manifest: env-level `AdapterDisplay[]`, per-robot `AdapterEntry[]` and the rviz panels and tools backends contribute as `AdapterPlugin[]` (`role`, pluginlib `class_name`, panel `name`, `properties_json`). Published latched on `state/viz_manifest`. |
| `RecordedTopics.msg` | Topics the recorder subscribes to beyond its own, as `RecordedTopic[]` rows (`key`, `topic` template with `{ns}` and `{tg}`, `msg_type`, `robot_scoped`, `throttled`, `qos_transient_local`, `reliable`, `depth` (0 = the recorder default), `recorded`). Published latched (`TRANSIENT_LOCAL`, depth 1) on `state/recorded_topics`. |
| `SemanticSnapshot.msg` | Full latched semantic state of the env: stamp, world, `SemanticEntityState[]`. Published on `state/semantics` (`TRANSIENT_LOCAL`), republished on any quantum-passing change (attach/detach/reset/write). |
| `SemanticEntityState.msg` | One semantic entity: `kind`, index-aligned discrete/continuous/predicate name-value arrays, `members` (committed occupant ids). |

## Actions (`action/`)

| File | Purpose |
|---|---|
| `RunEpisode.action` | Single-flight episode runner: goal carries optional world; result `state` is one of `QUEUED` / `RUNNING` / `SUCCESS` / `FAILED` / `SKIPPED` / `FATAL` (FATAL = env never reached a runnable state, do not retry). A concurrent `ResetEpisode` resolves the in-flight goal with `SKIPPED`. |
