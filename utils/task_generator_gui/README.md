# task_generator_gui

RViz2 plugins for the Arena-Rosnav task generator. Ships:

- **`TaskGeneratorPanel`** (`rviz_common::Panel`): episode management, task-mode selection, world/robot configuration, live episode history playlist.
- **`SpawnPedestrianTool`** (`rviz_common::Tool`): toolbar tool that click+drags a pose and calls `runtime/spawn_dynamic` to spawn a dynamic obstacle (pedestrian).
- **`AuditoryPanel`** (`rviz_common::Panel`): propagation and workstation playback switches, microphone routing, motor playback and tuning, environment audio sources.
- **`SpawnMicrophoneTool`** (`rviz_common::Tool`): toolbar tool that places a fixed or TF-attached acoustic listener.
- **`SpawnAudioSourceTool`** (`rviz_common::Tool`): toolbar tool that places a configurable radio or alarm.

## Service contract

All service paths are relative to the task_generator node namespace (default `/task_generator_node`, configurable via the RViz panel `Target` config key).

| Path | Type | Usage |
|---|---|---|
| `lifecycle/reset_episode` | `task_generator_msgs::srv::ResetEpisode` | Next button (after queue flush) |
| `lifecycle/pause` | `std_srvs::srv::Empty` | Pause button |
| `lifecycle/unpause` | `std_srvs::srv::Empty` | Unpause button |
| `lifecycle/wait_for_world` | `std_srvs::srv::Empty` | Called after world param change |
| `query/worlds` | `task_generator_msgs::srv::QueryWorlds` | Populate world combobox |
| `query/robots` | `task_generator_msgs::srv::QueryRobots` | Populate robot combobox |
| `query/environments` | `task_generator_msgs::srv::QueryEnvironments` | Populate environment list |
| `query/parametrizeds` | `task_generator_msgs::srv::QueryParametrizeds` | Populate parametrized list |
| `query/static_obstacles` | `task_generator_msgs::srv::QueryStaticObstacles` | Populate static model list |
| `query/dynamic_obstacles` | `task_generator_msgs::srv::QueryDynamicObstacles` | Populate dynamic model list |
| `query/scenarios` | `task_generator_msgs::srv::QueryScenarios` | Populate scenario list for current world |
| `query/task_modes` | `task_generator_msgs::srv::QueryTaskModes` | Populate mode comboboxes (obstacles, robots, modules) |
| `config/queue_episode` | `task_generator_msgs::srv::QueueEpisode` | Queue / Next buttons (modes, world, robots, per-mode params staged for next reset) |
| `runtime/spawn_dynamic` | `task_generator_msgs::srv::SpawnDynamic` | Spawn pedestrian tool (click+drag pose) |
| `runtime/spawn_microphone` | `task_generator_msgs::srv::SpawnMicrophone` | Spawn microphone tool (clicked position and configured height) |
| `runtime/remove_microphone` | `task_generator_msgs::srv::RemoveMicrophone` | Remove a runtime-spawned microphone |
| `runtime/spawn_audio_source` | `task_generator_msgs::srv::SpawnAudioSource` | Spawn a radio, alarm, or custom catalog asset |
| `runtime/set_audio_system` | `task_generator_msgs::srv::SetAudioSystem` | Start or stop a static radio or multi-speaker alarm |
| `runtime/remove_audio_system` | `task_generator_msgs::srv::RemoveAudioSystem` | Remove a runtime-spawned radio or alarm |
| `runtime/spawn_robot` | `task_generator_msgs::srv::SpawnRobot` | Spawn Robot button (mid-episode spawn) |

### Latched topics consumed

| Topic | Type | Usage |
|---|---|---|
| `state/episode` | `task_generator_msgs::msg::EpisodeRecord` | Current episode. Subscribers dedup by `episode_id` (same id may be republished as outcome resolves) and feed the panel's local `history_buffer_` (ring of 50, oldest-first) plus the current/bold row. |
| `state/queue` | `task_generator_msgs::msg::EpisodeRecord` | Queued (next) episode. On arrival the panel populates widgets via `populateFromQueue` (signal-blocked) and clears dirty flags. Drives the "Next:" preview row when it differs from current. |
| `state/paused` | `std_msgs::msg::Bool` | Drives the pause button label authoritatively. The pause button is fire-and-forget; UI reflects the published state, not the service-call return. |
| `/parameter_events` | `rcl_interfaces::msg::ParameterEvent` | Filters on `node == task_generator_node`; if any changed/new/deleted parameter starts with `task.<active_mode>.`, rebuilds the matching family's param tree on the Qt thread. |

## AuditoryPanel

Sibling panel with the same `Target` config key. Its parameter clients target
`<Target>/robot_sound_node`, `<Target>/human_sound_playback`,
`<Target>/environment_sound_playback`, and `<Target>/sound_propagation_node`;
every group stays disabled until the matching node's parameter service appears,
so the panel is inert under `auditory:=none`.

`Play robot motor audio on this workstation` controls the live
`enable_motor_playback` parameter on `<Target>/robot_sound_node`. It mutes
only the motor bus in the local audio mixer. Motor source messages, propagation,
RIR updates, and robot hearing continue unchanged. The panel reads the initial
value when the playback parameter service appears, follows `/parameter_events`,
and rechecks the value on each `state/episode` update.

`Motor Sound Tuning` exposes live volume, frequency, gear-tone level,
mechanical-noise level, velocity response, and response smoothing controls.
Edits are applied to active procedural drivetrain voices without restarting an
episode. `Reset motor tuning` restores the quieter procedural defaults.

## SpawnPedestrianTool

Subclass of `rviz_default_plugins::tools::PoseTool`. Click+drag in the 3D view to set position and yaw; the tool then calls `<Target>/runtime/spawn_dynamic` with `use_pose=true`, the clicked `PoseStamped` (in the rviz Fixed Frame), and the `Model` string. Both `Target` and `Model` are exposed as Tool Properties; `Model` defaults to `arenian`. Shortcut key: `p`.

## SpawnMicrophoneTool

Subclass of `rviz_default_plugins::tools::PoseTool`. Select **Spawn
Microphone** in the toolbar, then click in the 3D view. The tool calls
`<Target>/runtime/spawn_microphone` with the clicked point in the RViz Fixed
Frame and the `Height` tool property, which defaults to 1.5 m. The acoustic
runtime registers the clicked frame and position immediately. It assigns
`microphone1`, `microphone2`, and later increasing IDs for the episode. The
green triangular cone and ID label therefore appear without waiting for room
geometry to load.

Set `Attach TF Frame` to a frame from the RViz TF tree to make the listener
follow that frame. The clicked point is converted into an offset in the named
frame. Leaving the property empty creates a fixed listener.

The new ID appears in the Auditory panel's **Audio Playback Microphone**
dropdown. Choose it under **Listen through** to route propagation and playback
through that microphone only. Runtime-spawned microphones are cleared on an
episode or world change. Every live robot also contributes
`<robot_name>_mic`, attached to its base TF frame.

The **Spawn Radio** toolbar button places an environment source (radio or alarm). The Auditory panel's
**Environment Audio Sources** table lists every scenario- or
launch-defined source. Check a row to start the whole system and
uncheck it to stop it. A system can contain several speakers. The listener
routing control is applied to human, robot, and environment playback, so the
selected microphone also applies to radios and alarms.

The Auditory panel's **Auditory Runtime** controls independently enable simulated
propagation and local environment playback. Disabling local playback does
not stop propagation or robot hearing. Runtime sources can be selected and
removed from the environment-sources table.

`Spawn Radio` defaults to the bundled looping music or alarm asset and starts
immediately. `Custom Playback` exposes the catalog asset ID, source volume,
loop flag, and initial active state.

## Discard / Queue / Next buttons

The button row is `Pause | Discard | Queue | Next`. Discard and Queue are enabled only when widget values differ from the latched `state/queue` snapshot (any of the per-family dirty flags set).

- **Discard**: re-populates widgets from `last_queued_episode_` via `populateFromQueue` (signal-blocked) and clears dirty. No service call.
- **Queue**: builds a `QueueEpisode::Request` from current widget state and sends `config/queue_episode`. Does not trigger a reset. The latched `state/queue` topic round-trip then refreshes widgets and clears dirty.
- **Next**: if dirty, calls Queue first; once the response arrives, calls `lifecycle/reset_episode` (`world` left empty so the server resolves from queued overrides). If a `RunEpisode` goal is in flight, the server resolves it with `Result.SKIPPED` (reason="reset").

## Pause / Unpause button

A single Pause/Unpause button. The label is driven by the latched `state/paused` topic, not by the service-call return. Clicking is fire-and-forget: calls `lifecycle/pause` or `lifecycle/unpause` based on the current state, and the label flips when the server publishes the new value. This keeps the UI in sync even if pause/unpause is triggered externally.

## Playlist (episode history)

The panel maintains a local `history_buffer_` (`std::deque<EpisodeRecord>`, max 50) deduped by `episode_id`. On every `state/episode` arrival the record is upserted into the buffer and the table is rebuilt:

- Each finalized record is shown oldest-first: `episode_id`, `world`, `seed`, `outcome_state` (QUEUED/RUNNING/SUCCESS/FAILED/SKIPPED/FATAL), `outcome_info`. The `outcome_info` cell is live: tasks may republish it mid-episode (e.g. `"timeout after 120s"`, `"50% complete"`) via `Task.set_info(text)` and the panel will refresh on the next latched message.
- The current episode (`last_current_episode_`) appears as the last (bottom) row in bold.
- A "Next:" preview row is appended (italic) when `last_queued_episode_` differs meaningfully from current (world, modes, robots, or any param).

History is held only in the panel.

## World combobox staging

The world combobox stages a world value into the queued slot via `QueueEpisode`. World is bound to the episode and changes at the next reset boundary; nothing happens to the live world until Queue or Next fires.

## Task-mode tabs

Two tabs (Obstacles, Robots) hold the only TM controls:

- **Obstacles tab** has a task-mode combobox (`Environment` / `Parametrized` / `Random` / `Scenario` / `Prompt`) and per-mode parameter widgets. Combobox change marks `tm_obstacles_dirty_`, rebuilds the param tree, and updates Discard/Queue enabled state. No service call until Queue / Next fires.
- **Robots tab** has a task-mode combobox (`Explore` / `Guided` / `Random` / `Scenario`). Combobox change marks `tm_robots_dirty_`, rebuilds the param tree, and updates Discard/Queue enabled state. No service call until Queue / Next fires.

Server validates the string values against the Python enum on the eventual `QueueEpisode` call. On rejection, a `WARN` is logged with the server's `error_msg`.

### Per-mode parameter staging

All edits are batched locally (no live push). On Queue / Next, `buildQueueEpisodeRequest` packages current widget state into a single `config/queue_episode` request: `tm_robots`, `tm_obstacles` (lowercased), `world`, `robots = [selected_robot_model]`, plus per-mode `obstacles_params` and `robots_params` (each a `rcl_interfaces/Parameter` with a leaf-keyed name relative to the mode, e.g. `scenario.file`, `static.n`). `QueueEpisode` is per-field merge with `action = MERGE`: empty scalar fields preserve previously-queued overrides; `robots` unions with prior queued set (dedup, insertion order). The server stages everything and applies at the next `lifecycle/reset_episode` boundary.

### Mode combobox population

The GUI uses `query/task_modes` to populate the Obstacles and Robots comboboxes at startup. The comboboxes stay in sync automatically when new task modes are registered on the server side. Since `walk_schemas` declares all `task.<mode>.*` params at node startup, the param tree can be rebuilt against any mode without first switching the active mode.

## Cold start

`load()` returns immediately after building the empty UI shell. Every discovery call is async:

- `query/robots`, `query/worlds`, `query/task_modes` are gated through a `whenReady` polling helper that waits for service readiness off the rviz executor, then fires `async_send_request` with a response callback. Each callback marshals back to the Qt thread via `QMetaObject::invokeMethod(..., Qt::QueuedConnection)` and uses `QSignalBlocker` while populating combos to suppress spurious `currentTextChanged` signals.
- The first `state/queue` arrival drives `populateFromQueue`, which seeds the task-mode combos, world combobox, and robot combobox (signal-blocked), then triggers the initial `rebuildParamTree` for both families and writes any param values from the queued record into the widgets via `setWidgetValueFromParam`. `loading_from_queue_` is set during population so widget signals don't bump dirty flags.
- `rebuildParamTree` carries a per-family generation counter through a shared `RebuildState` struct; stale callbacks (older generation) drop their results. The chain is `list_parameters` → parallel `describe_parameters` + `get_parameters` → catalog fan-out → Qt-thread widget construction.
