# task_generator utils

Standalone helpers used across the package.

## `arena.py`

[`arena.py`](arena.py)

Two small utilities:

- `get_arena_type() -> Constants.ArenaType`: reads the `ARENA_TYPE` env var
  (defaults to `deployment`).
- `generate_map_inner_border(free_space_indices, map_)`: extracts the four
  bounding-box vertices of the free-space region from an `OccupancyGrid`.
- `update_freespace_indices_maze(map_)`: marks hardcoded wall regions as
  occupied; left from a specific evaluation map, not for general use.

## `gpt.py`

[`gpt.py`](gpt.py)

Lazy shims for `google.genai` and `openai` that redirect API calls to a local
LLM endpoint when `LLM_API_ENDPOINT` is set in the environment. Accessed via
module `__getattr__`: `from task_generator.utils import gpt; gpt.genai` /
`gpt.openai`. When `LLM_API_ENDPOINT` is unset, the shims pass through to the
upstream libraries unchanged.

## `map_generator.py`

[`map_generator.py`](map_generator.py)

Standalone ROS 2 node (`map_generator`) that procedurally generates a
hallway-and-rooms world: random room widths on each side of a central
hallway, door gaps, wall YAML, zones YAML, `map.yaml`, and a PNG image. The
output directory is `$ARENA_WS_DIR/src/arena/simulation-setup/worlds/<map_name>`.
Not used at runtime by the task-generator core; invoked as a standalone tool.

## `taskgen_srvs.py`

[`taskgen_srvs.py`](taskgen_srvs.py)

Empty file (1 line). Reserved for task-generator service definitions.

## `arena_rclpy_mixins`: rclpy↔asyncio bridge

The canonical bridge between rclpy and asyncio is
[`utils/arena_rclpy_mixins/`](../../../utils/arena_rclpy_mixins) in the
workspace root. Use `AsyncNode.await_ros`, `ClientWrapper`, and related
helpers from that package for all async ROS I/O inside task_generator. Do not
hand-roll `asyncio.wrap_future` or equivalent patterns.
