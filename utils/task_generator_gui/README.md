# task_generator_gui

RViz2 plugins for the Arena-Rosnav task generator. Ships:

- **`TaskGeneratorPanel`** (`rviz_common::Panel`): episode management, task-mode selection, world/robot configuration, live episode history playlist.
- **`SpawnPedestrianTool`** (`rviz_common::Tool`): toolbar tool that click+drags a pose and calls `runtime/spawn_dynamic` to spawn a dynamic obstacle (pedestrian).

The auditory panel and the spawn microphone and spawn sound tools live in the
`arena_auditory_viz` package of the auditory feature, see the
[arena_auditory README](../../arena_auditory/README.md#rviz-plugins). The
`acoustics:=arena` backend declares them in the viz manifest's `plugins`, and
`rviz_config.py` adds every declared panel and tool to the generated config.

## Service contract

All service paths are relative to the env node namespace (default `/arena_env_node`, configurable via the RViz panel `Target` config key).

| Path | Type | Usage |
|---|---|---|
| `lifecycle/reset_episode` | `arena_runtime_msgs::srv::ResetEpisode` | Next button (after queue flush) |
| `lifecycle/wait_for_world` | `std_srvs::srv::Empty` | Called after world param change |
| `query/worlds` | `arena_runtime_msgs::srv::QueryWorlds` | Populate world combobox |
| `query/robots` | `arena_runtime_msgs::srv::QueryRobots` | Populate robot combobox |
| `query/environments` | `arena_runtime_msgs::srv::QueryEnvironments` | Populate environment list |
| `query/parametrizeds` | `arena_runtime_msgs::srv::QueryParametrizeds` | Populate parametrized list |
| `query/static_obstacles` | `arena_runtime_msgs::srv::QueryStaticObstacles` | Populate static model list |
| `query/dynamic_obstacles` | `arena_runtime_msgs::srv::QueryDynamicObstacles` | Populate dynamic model list |
| `query/scenarios` | `arena_runtime_msgs::srv::QueryScenarios` | Populate scenario list for current world |
| `query/task_modes` | `arena_runtime_msgs::srv::QueryTaskModes` | Populate the mode comboboxes from `modes` (axes `obstacles` and `robots`) |
| `config/queue_episode` | `arena_runtime_msgs::srv::QueueEpisode` | Queue / Next buttons (`modes`, world, full-name `params` staged for next reset) |
| `runtime/spawn_dynamic` | `arena_runtime_msgs::srv::SpawnDynamic` | Spawn pedestrian tool (click+drag pose) |
| `runtime/spawn_robot` | `arena_runtime_msgs::srv::SpawnRobot` | Spawn Robot button (mid-episode spawn) |
| `/arena/sim_lifecycle/hold` (absolute) | `arena_runtime_msgs::srv::LifecycleHold` | Pause button: acquires and releases the panel's own hold |

### Latched topics consumed

| Topic | Type | Usage |
|---|---|---|
| `state/episode` | `arena_runtime_msgs::msg::EpisodeRecord` | Current episode. Subscribers dedup by `episode_id` (same id may be republished as outcome resolves) and feed the panel's local `history_buffer_` (ring of 50, oldest-first) plus the current/bold row. |
| `state/queue` | `arena_runtime_msgs::msg::EpisodeRecord` | Queued (next) episode. On arrival the panel populates widgets via `populateFromQueue` (signal-blocked) and clears dirty flags. Drives the "Next:" preview row when it differs from current. |
| `/arena/state/holders` (absolute) | `arena_runtime_msgs::msg::HoldRegistry` | Drives the pause button label: paused while the panel's own hold is listed. |
| `state/resetting` | `std_msgs::msg::Bool` | True while `_run_reset_cycle` holds the reset lock. The world generator panel disables "+Deploy" while it is set. |
| `/parameter_events` | `rcl_interfaces::msg::ParameterEvent` | Filters on `node == arena_env_node`; if any changed/new/deleted parameter starts with the active mode's namespace, rebuilds the matching family's param tree on the Qt thread. |

## SpawnPedestrianTool

Subclass of `rviz_default_plugins::tools::PoseTool`. Click+drag in the 3D view to set position and yaw; the tool then calls `<Target>/runtime/spawn_dynamic` with `use_pose=true`, the clicked `PoseStamped` (in the rviz Fixed Frame), and the `Model` string. Both `Target` and `Model` are exposed as Tool Properties; `Model` defaults to `arenian`. Shortcut key: `p`.

## Discard / Queue / Next buttons

The button row is `Pause | Discard | Queue | Next`. Discard and Queue are enabled only when widget values differ from the latched `state/queue` snapshot (any of the per-family dirty flags set).

- **Discard**: re-populates widgets from `last_queued_episode_` via `populateFromQueue` (signal-blocked) and clears dirty. No service call.
- **Queue**: builds a `QueueEpisode::Request` from current widget state and sends `config/queue_episode`. Does not trigger a reset. The latched `state/queue` topic round-trip then refreshes widgets and clears dirty.
- **Next**: if dirty, calls Queue first; once the response arrives, calls `lifecycle/reset_episode` (`world` left empty so the server resolves from queued overrides). If a `RunEpisode` goal is in flight, the server resolves it with `Result.SKIPPED` (reason="reset").

## Pause / Unpause button

A single Pause/Unpause button backed by the runtime's ref-counted holds. Pause acquires a hold on `/arena/sim_lifecycle/hold` under the panel's own caller id (reason `panel_pause`), Unpause releases it, and the panel releases a hold it still owns when it closes. The label follows the latched `/arena/state/holders`, so it shows whether the panel's hold is in effect. The sim stays paused while other callers hold it, since a hold is shared by every env of one sim.

## Playlist (episode history)

The panel maintains a local `history_buffer_` (`std::deque<EpisodeRecord>`, max 50) deduped by `episode_id`. On every `state/episode` arrival the record is upserted into the buffer and the table is rebuilt:

- Each finalized record is shown oldest-first: `episode_id`, `world`, `seed`, `outcome_state` (QUEUED/RUNNING/SUCCESS/FAILED/SKIPPED/FATAL), `outcome_info`. The `outcome_info` cell is live: tasks may republish it mid-episode (e.g. `"timeout after 120s"`, `"50% complete"`) via `EpisodeNode.set_episode_info(text)` and the panel will refresh on the next latched message.
- The current episode (`last_current_episode_`) appears as the last (bottom) row in bold.
- A "Next:" preview row is appended (italic) when `last_queued_episode_` differs meaningfully from current (world, modes, robots, or any param).

History is held only in the panel.

## World combobox staging

The world combobox stages a world value into the queued slot via `QueueEpisode`. World is bound to the episode and changes at the next reset boundary; nothing happens to the live world until Queue or Next fires.

## Task-mode tabs

Two tabs (Obstacles, Robots) hold the only TM controls:

- **Obstacles tab** has a task-mode combobox (`Environment` / `Parametrized` / `Random` / `Scenario` / `Prompt`) and per-mode parameter widgets. Combobox change marks `tm_obstacles_dirty_`, rebuilds the param tree, and updates Discard/Queue enabled state. No service call until Queue / Next fires.
- **Robots tab** has a task-mode combobox (`Explore` / `Guided` / `Random` / `Scenario`). Combobox change marks `tm_robots_dirty_`, rebuilds the param tree, and updates Discard/Queue enabled state. No service call until Queue / Next fires.

The runtime validates the selected mode names on the eventual `QueueEpisode` call. On rejection, a `WARN` is logged with the server's `error_msg`.

### Per-mode parameter staging

All edits are batched locally (no live push). On Queue / Next, `buildQueueEpisodeRequest` packages current widget state into a single `config/queue_episode` request: `modes` (axes `obstacles` and `robots`, mode names lowercased), `world`, `robots = [selected_robot_model]`, plus `params` (each a `rcl_interfaces/Parameter` named by its full name, the mode's namespace followed by the leaf, e.g. `task.scenario.file`, the namespace as `query/task_modes` reports it in `namespaces`). The panel strips the same namespace from `EpisodeRecord.params` to fill the widgets. `QueueEpisode` is per-field merge with `action = MERGE`: empty scalar fields preserve previously-queued overrides; `robots` unions with prior queued set (dedup, insertion order). The server stages everything and applies at the next `lifecycle/reset_episode` boundary.

### Mode combobox population

The GUI uses `query/task_modes` to populate the Obstacles and Robots comboboxes at startup and keeps its `namespaces` (`<axis>/<mode>` to param namespace): param trees are built under, and leaves queued under, the selected mode's namespace. The comboboxes stay in sync automatically when new task modes are registered on the server side. Since the env declares every registered mode's params at node startup, the param tree can be rebuilt against any mode without first switching the active mode.

## Cold start

`load()` returns immediately after building the empty UI shell. Every discovery call is async:

- `query/robots`, `query/worlds`, `query/task_modes` are gated through a `whenReady` polling helper that waits for service readiness off the rviz executor, then fires `async_send_request` with a response callback. Each callback marshals back to the Qt thread via `QMetaObject::invokeMethod(..., Qt::QueuedConnection)` and uses `QSignalBlocker` while populating combos to suppress spurious `currentTextChanged` signals.
- The first `state/queue` arrival drives `populateFromQueue`, which seeds the task-mode combos, world combobox, and robot combobox (signal-blocked), then triggers the initial `rebuildParamTree` for both families and writes any param values from the queued record into the widgets via `setWidgetValueFromParam`. `loading_from_queue_` is set during population so widget signals don't bump dirty flags.
- `rebuildParamTree` carries a per-family generation counter through a shared `RebuildState` struct; stale callbacks (older generation) drop their results. The chain is `list_parameters` → parallel `describe_parameters` + `get_parameters` → Qt-thread widget construction → catalog fan-out. Catalog combos render immediately, seeded with the current value(s) plus any ids memoized from earlier fetches, and are refilled in place (signal-blocked, selection preserved) as each catalog arrives.
