# task_generator_mcp

MCP (Model Context Protocol) server that wraps `task_generator_node` services as tools and resources. Provides LLM-friendly access to episode lifecycle, world/robot/obstacle queries, task-mode configuration, and runtime spawning.

Transport: stdio (v0). Works with `mcp-cli` and any stdio-MCP client.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `TASK_GENERATOR_NODE_NAME` | `/task_generator_node` | Full ROS node name used to construct service paths (e.g. `/task_generator_node/lifecycle/reset_episode`). |

---

## Tools

### Lifecycle

| Tool | Args | Returns | Purpose |
|---|---|---|---|
| `lifecycle_reset_episode` | `world: str = ""`, `seed: int = -1` | `{success, error_msg}` | Advance to a new episode. Empty world inherits current; seed >= 0 for bit-perfect replay. |
| `lifecycle_pause` | - | `{success}` | Pause simulation clock. |
| `lifecycle_unpause` | - | `{success}` | Resume simulation clock. |
| `lifecycle_wait_for_world` | - | `{success}` | Block until the active world is fully loaded. |
| `lifecycle_run_episode` | `world: str = ""` | `{state, reason, episode_id}` | Run a single episode to completion via the action server. `state` is `"SUCCESS"`, `"FAILED"`, or `"SKIPPED"`. |

### Query

| Tool | Args | Returns | Purpose |
|---|---|---|---|
| `query_worlds` | - | `list[str]` | All available world ids. |
| `query_scenarios` | `world: str = ""` | `list[str]` | Scenario shortnames for world (empty = current world). |
| `query_robots` | - | `list[str]` | Available robot model shortnames. |
| `query_static_obstacles` | - | `list[str]` | Available static obstacle model shortnames. |
| `query_dynamic_obstacles` | - | `list[str]` | Available dynamic obstacle model shortnames. |
| `query_environments` | - | `list[str]` | Available environment configuration shortnames. |
| `query_parametrizeds` | - | `list[str]` | Available parametrized obstacle configuration shortnames. |
| `query_episode` | - | `{current: {...}, queued: {...}}` | Current episode record (live) and queued (next) episode record. |

### Config

| Tool | Args | Returns | Purpose |
|---|---|---|---|
| `config_queue_episode` | `tm_robots`, `tm_obstacles`, `tm_modules`, `keep_modules`, `world`, `robots` | `{success, error_msg}` | Queue overrides for the next episode. Per-field merge (action=MERGE): empty scalar fields preserve previously-queued values; `robots` unions with prior queued set (dedup, insertion order). Applied at next `lifecycle/reset_episode`. |
| `config_get_task_modes` | - | `{tm_robots, tm_obstacles, tm_modules}` | Read current task mode `.value` strings. |
| `config_set_episode_params` | `timeout`, `goal_tolerance_radius`, `robot_safe_dist`, `auto_reset`, `episodes`, `record_data_dir` | `{results: [...]}` | Write only the supplied (non-null) episode params via `SetParameters`. |
| `config_get_episode_params` | - | `{timeout, goal_tolerance_radius, ...}` | Read current episode-shaping params. |
| `config_get_static_config` | - | `{sim, human, robot.mobile_adapter, robot.arm_adapter}` | Read read-only startup configuration params. |

### Runtime spawn

| Tool | Args | Returns | Purpose |
|---|---|---|---|
| `runtime_spawn_static` | `model: str`, `pose?: {position, orientation, frame_id}` | `{id, success, error_msg}` | Spawn a static obstacle. Omit pose for random placement. |
| `runtime_spawn_dynamic` | `model: str`, `pose?` | `{id, success, error_msg}` | Spawn a dynamic obstacle / pedestrian. |
| `runtime_spawn_robot` | `model: str`, `name?: str`, `pose?`, `immediate?: bool` | `{name, success, error_msg}` | Spawn a robot (experimental). `immediate` provisions it live and idle now, else it commits on the next reset. |

Task-mode enum values for `tm_robots` / `tm_obstacles` / `tm_modules` are compiled from `task_generator.constants.Constants.TaskMode.*` at server startup, so the tool schema `enum` field always reflects the live codebase.

**Robot goal-setting is not exposed here.** Drive robots through `<robot_namespace>/goto_pose` (arena_robots task_server action).

---

## Resources

| URI | MIME | Description |
|---|---|---|
| `task_generator://state/world` | `text/plain` | Current world id (cached from latched `state/world` topic). |
| `task_generator://state/episode` | `application/json` | `{current: {...}, queued: {...}}` from latched `state/episode` and `state/queue` topics. |

---

## Setup

### Stdio MCP host

Add to the host's MCP server config:

```json
{
  "mcpServers": {
    "task_generator": {
      "command": "task_generator_mcp",
      "env": {
        "TASK_GENERATOR_NODE_NAME": "/task_generator_node"
      }
    }
  }
}
```

The `task_generator_mcp` console script is installed by `colcon build` into the ROS overlay. Source the overlay (`source ~/arena_ws/install/setup.bash`) before starting the host, or set an absolute path to the installed binary.

### mcp-cli

```bash
source ~/arena_ws/install/setup.bash
mcp tools task_generator_mcp
mcp call task_generator_mcp query_worlds
```

---

## Smoke commands (not run in CI, for manual verification)

```bash
$ mcp tools task_generator_mcp
$ mcp call task_generator_mcp query_worlds
$ mcp call task_generator_mcp query_episode '{"range": 3}'
$ mcp call task_generator_mcp lifecycle_reset_episode '{"world": "empty_map"}'
$ mcp call task_generator_mcp config_get_task_modes
```
